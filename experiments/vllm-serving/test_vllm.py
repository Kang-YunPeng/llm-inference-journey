# test_vllm.py
from vllm import LLM, SamplingParams

MODEL_PATH = "./models/Qwen1.5-4B-Chat-AWQ"
GPU_MEM_UTIL = 0.80
MAX_MODEL_LEN = 2048
MAX_TOKENS_SHORT = 32
MAX_TOKENS_LONG = 16

# 初始化 LLM
llm = LLM(
    model=MODEL_PATH,
    quantization="awq",
    dtype="half",
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
    enforce_eager=True,
    max_num_seqs=1,
)

# 测试 prompt
short_prompt = "Hello! How are you?"
long_prompt = ("Explain the theory of relativity in simple terms. " * 50).strip()

# 关键：启用 detailed_output
sampling_params_short = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_TOKENS_SHORT,
    detokenize=False,        # 可选：加速（我们只关心 token 和 timing）
    include_stop_str_in_output=False,
)

sampling_params_long = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_TOKENS_LONG,
    detokenize=False,
)

print("Starting inference...\n")

def print_latency_info(prompt_name: str, output):
    metrics = output.metrics
    if metrics:
        # Prefill: 从请求开始到首 token 生成
        prefill_latency = metrics.first_token_time - metrics.arrival_time
        # Decoding: 从首 token 到末 token
        decoding_latency = metrics.finished_time - metrics.first_token_time
        total_latency = metrics.finished_time - metrics.arrival_time
        num_generated = len(output.outputs[0].token_ids)
        
        print(f"\n {prompt_name} Latency Breakdown:")
        print(f"  • Prefill latency:     {prefill_latency:.3f} s")
        print(f"  • Decoding latency:    {decoding_latency:.3f} s")
        print(f"  • Total latency:       {total_latency:.3f} s")
        if decoding_latency > 0:
            print(f"  • Decoding speed:      {num_generated / decoding_latency:.1f} tokens/s")
    else:
        print(f"\n No metrics available for {prompt_name}. (Ensure vLLM >= 0.4.0)")

# === 短请求测试 ===
print("=== Short Prompt (mostly DECODING) ===")
print(f"Prompt: {short_prompt[:50]}...")
outputs = llm.generate(short_prompt, sampling_params_short)
output_text = llm.get_tokenizer().decode(outputs[0].outputs[0].token_ids)
print(f"Generated ({len(outputs[0].outputs[0].token_ids)} tokens): {output_text[:100]}...")
print_latency_info("Short Prompt", outputs[0])

# === 长请求测试 ===
print("\n=== Long Prompt (heavy PREFILL) ===")
print(f"Prompt length: ~300 tokens")
outputs = llm.generate(long_prompt, sampling_params_long)
output_text = llm.get_tokenizer().decode(outputs[0].outputs[0].token_ids)
print(f"Generated {len(outputs[0].outputs[0].token_ids)} tokens.")
print_latency_info("Long Prompt", outputs[0])