set -e

echo "=== [1/7] NCCL ==="
apt-get install -y libnccl2 libnccl-dev

echo "=== [2/7] CUDA 12.4+ dev libs ==="
apt-get update -q && apt-get install -y -q \
    libcusparse-dev-12-4 \
    libcublas-dev-12-4 \
    cuda-nvcc-12-4 \
    libcudnn9-dev-cuda-12

echo "=== [3/7] Variables CUDA ==="
export CUDA_HOME=/usr/local/cuda
export NVTE_CUDA_INCLUDE_PATH=$CUDA_HOME/include
export PATH=$CUDA_HOME/bin:$PATH
echo "  CUDA_HOME=$CUDA_HOME"
echo "  nvcc : $(nvcc --version | head -1)"

echo "=== [4/7] Ninja ==="
pip install ninja

echo "=== [5/7] Transformer Engine (FP8 — build ~10-20min) ==="
MAX_JOBS=1 pip install --no-build-isolation transformer-engine[pytorch]

echo "=== [6/7] FlashAttention (couches GPT) ==="
pip install flash-attn --no-build-isolation

echo "=== [7/7] Flash Linear Attention (couches GDN — kernel Triton) ==="
pip install flash-linear-attention

echo "=== [8/8] Dépendances Python ==="
pip install -e .
pip install -r requirements.txt

echo ""
echo "Installation terminée !"
echo "  Test : python -c \""
echo "    import transformer_engine; import flash_attn"
echo "    from fla.ops.gated_delta_rule import chunk_gated_delta_rule"
echo "    from naylisgdn import NaylisGDN; print('OK')"
echo "  \""
