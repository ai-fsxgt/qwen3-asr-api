#!/usr/bin/env bash
set -e

# 构建上下文必须是仓库根目录。
cd "$(dirname "$0")/.."

IMAGE_NAME="registry.cn-hangzhou.aliyuncs.com/ripper/qwen3-asr-service"
VLLM_TAG="${VLLM_TAG:-v0.14.0}"

read -rp "请输入版本号（回车默认 latest）: " input_ver
VER="${input_ver:-latest}"
TAG="${VER}-vllm"

if [ "$VER" = "latest" ]; then
    APP_VER="dev"
else
    APP_VER="$VER"
fi

echo "Building ${IMAGE_NAME}:${TAG} (vLLM, GPU/amd64, base vllm/vllm-openai:${VLLM_TAG}) ..."
docker build \
    -f docker/Dockerfile.vllm \
    --build-arg APP_VERSION="${APP_VER}" \
    --build-arg VLLM_TAG="${VLLM_TAG}" \
    -t "${IMAGE_NAME}:${TAG}" \
    .

echo
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo
echo "Run example:"
echo "  docker run --gpus all -p 8765:8765 -v /path/to/models:/app/models ${IMAGE_NAME}:${TAG}"
