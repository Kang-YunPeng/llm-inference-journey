# GPU Behavior Log (RTX 2070 + Qwen1.5-4B-Chat-AWQ)

## 环境
- GPU: NVIDIA RTX 2070 (8GB VRAM)
- Model: ./models/Qwen1.5-4B-Chat-AWQ
- vLLM: 0.4.3 (source install)
- CUDA: 12.1 (nvcc available)

## 测试脚本
test_vllm.py

## 实测性能数据

### 短请求 ("Hello!", generate 32 tokens)
- Prefill latency:     0.020 s  → ~2 tokens processed → 100 tok/s
- Decoding latency:    0.485 s
- Decoding speed:      66.0 tokens/s
- Observation: Decoding dominates total time && low decoding speed → memory-bound

### 长请求 (~300 tokens prompt, generate 16 tokens)
- Prefill latency:     0.184 s  → ~300 tokens processed → 1630 tok/s
- Decoding latency:    0.192 s
- Decoding speed:      83.1 tokens/s
- Observation: Prefill time increases significantly && high prefill speed → compute-bound

## 核心结论
1. Prefill 阶段耗时与 prompt 长度强相关（O(L²) attention）
2. Decoding 阶段速度受 KV Cache 访问模式影响（Decoding 阶段性能强依赖 KV Cache 的读写方式（PagedAttention、块布局、带宽、是否合并访存等））