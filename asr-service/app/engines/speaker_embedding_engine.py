"""ReDimNet2 speaker embedding engine.

Offline diarization uses short overlapping windows. Voice identification uses
longer windows through ``embed_identity_clips``. Both paths share one model,
one inference lock, and the existing realtime-priority gate.
"""

from collections.abc import Iterable
from contextlib import nullcontext
from itertools import islice
import logging
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from app.config import MODEL_LOCAL_MAP
from app.utils.model_manager import require_model_file

logger = logging.getLogger(__name__)

DIARIZATION_WIN_SEC = 1.5
DIARIZATION_STEP_SEC = 0.75
IDENTITY_WIN_SEC = 5.0
IDENTITY_STEP_SEC = 2.5

# Existing clustering call sites import these aliases.
WIN_SEC = DIARIZATION_WIN_SEC
STEP_SEC = DIARIZATION_STEP_SEC


def make_windows(
    start_sec: float,
    end_sec: float,
    win: float = DIARIZATION_WIN_SEC,
    step: float = DIARIZATION_STEP_SEC,
) -> list[tuple[float, float]]:
    """Split one speech interval into overlapping diarization windows."""
    out = []
    position = start_sec
    while position + win < end_sec + step:
        out.append((position, min(position + win, end_sec)))
        position += step
    if not out and end_sec > start_sec:
        out.append((start_sec, end_sec))
    return out


def make_identity_windows(
    start_sec: float,
    end_sec: float,
    win: float = IDENTITY_WIN_SEC,
    step: float = IDENTITY_STEP_SEC,
) -> list[tuple[float, float]]:
    """Return complete identity windows, excluding a short trailing fragment."""
    windows = []
    position = start_sec
    while position + win <= end_sec:
        windows.append((position, position + win))
        position += step
    return windows


class SpeakerEmbeddingEngine:
    """ReDimNet2-B6 speaker embedder with normalized 192-d outputs."""

    BACKEND = "pytorch"
    MODEL_TAG = "redimnet2_b6_vb2_vox2_cnc2_v0_lm@v1"
    EMB_DIM = 192
    SAMPLE_RATE = 16000
    _BATCH = 4
    _MODEL_FILE = "b6-vb2+vox2+cnc2_v0-lm.pt"
    _MODEL_SHA256 = (
        "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868"
    )

    def __init__(
        self,
        *,
        device: str = "auto",
        priority_gate=None,
        inference_lock=None,
    ):
        self._device = self._resolve_device(device)
        self._model = None
        self._infer_lock = (
            inference_lock if inference_lock is not None else threading.Lock()
        )
        self._priority_gate = priority_gate

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected == "auto":
            selected = "cpu"
        resolved = torch.device(selected)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ReDimNet2 配置为 CUDA，但当前环境没有可用 GPU")
        return resolved

    def load(self) -> None:
        local_dir = MODEL_LOCAL_MAP["redimnet2_b6"]
        weight_path = os.path.join(local_dir, self._MODEL_FILE)
        require_model_file(
            weight_path,
            sha256=self._MODEL_SHA256,
        )

        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=True,
        )
        from app.engines.redimnet2 import ReDimNet2Wrap

        model = ReDimNet2Wrap(**checkpoint["model_config"])
        result = model.load_state_dict(checkpoint["state_dict"])
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "ReDimNet2 权重与模型结构不一致: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        model.eval()
        self._model = model.to(self._device)
        logger.info(
            "说话人 embedding 模型已加载 "
            f"(ReDimNet2-B6/{self._device.type}, tag={self.MODEL_TAG})"
        )

    @staticmethod
    def _circle_pad(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
        if waveform.shape[0] >= target_len:
            return waveform[:target_len]
        repeats = -(-target_len // waveform.shape[0])
        return torch.cat([waveform] * repeats)[:target_len]

    @staticmethod
    def _raise_if_cancelled(cancelled) -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("说话人处理已取消")

    def _inference_context(self):
        if self._device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        with self._infer_lock, torch.inference_mode(), self._inference_context():
            return self._model(batch.to(self._device, non_blocking=True))

    def _embed_clips(
        self,
        clips: Iterable[np.ndarray],
        *,
        cancelled=None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("说话人 embedding 模型未加载，请先调用 load()")
        self._raise_if_cancelled(cancelled)
        gate = self._priority_gate
        outputs = []
        clip_iterator = iter(clips)
        while True:
            self._raise_if_cancelled(cancelled)
            current_clips = list(islice(clip_iterator, self._BATCH))
            if not current_clips:
                break
            if gate is not None:
                if not gate.wait_realtime_clear(cancelled=cancelled):
                    raise InterruptedError("说话人处理已取消")
            current = []
            for clip in current_clips:
                waveform = np.asarray(clip, dtype=np.float32).reshape(-1)
                if waveform.size == 0:
                    raise ValueError("声纹窗口没有有效采样")
                current.append(
                    torch.from_numpy(np.ascontiguousarray(waveform))
                )
            target_len = max(item.shape[0] for item in current)
            batch = torch.stack([
                self._circle_pad(item, target_len) for item in current
            ])
            outputs.append(self._forward(batch).float().cpu())

        self._raise_if_cancelled(cancelled)
        if not outputs:
            return np.zeros((0, self.EMB_DIM), dtype=np.float32)
        return F.normalize(torch.cat(outputs), dim=1).numpy()

    def _embed_windows(
        self,
        wav: np.ndarray,
        windows: list[tuple[float, float]],
        *,
        cancelled=None,
    ) -> np.ndarray:
        waveform = np.asarray(wav, dtype=np.float32).reshape(-1)

        def iter_clips():
            for start, end in windows:
                self._raise_if_cancelled(cancelled)
                clip = waveform[
                    max(int(start * self.SAMPLE_RATE), 0):
                    min(int(end * self.SAMPLE_RATE), waveform.size)
                ]
                if clip.size == 0:
                    raise ValueError(
                        f"声纹窗口没有有效采样: ({start}, {end})"
                    )
                yield clip

        return self._embed_clips(
            iter_clips(),
            cancelled=cancelled,
        )

    def embed_windows(
        self,
        wav: np.ndarray,
        windows: list[tuple[float, float]],
        *,
        cancelled=None,
    ) -> np.ndarray:
        return self._embed_windows(
            wav,
            windows,
            cancelled=cancelled,
        )

    def embed_identity_clips(
        self,
        clips: Iterable[np.ndarray],
        *,
        cancelled=None,
    ) -> np.ndarray:
        """流式处理已分割的离线身份识别窗口。"""
        return self._embed_clips(
            clips,
            cancelled=cancelled,
        )

    def unload(self) -> None:
        self._model = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("说话人 embedding 模型已卸载")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._device)
