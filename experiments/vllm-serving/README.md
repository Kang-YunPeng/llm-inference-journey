# Install PyTorch with CUDA 12.1
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

vllm version: 0.4.3

# fix vllm 0.4.3 rope_scaling issue
config.py:1215: 
rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        rope_type = rope_scaling.get("type")
        scaling_factor = rope_scaling.get("factor")
        if rope_type is not None and scaling_factor is not None and rope_type != "su":
            if disable_sliding_window:
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate.")

            derived_max_model_len *= scaling_factor

            if rope_type == "yarn":
                original_max_len = rope_scaling.get("original_max_position_embeddings")
                if original_max_len is not None:
                    derived_max_model_len = original_max_len * scaling_factor
                else:
                    pass
        else:
            if rope_type is None or scaling_factor is None:
                import warnings
                warnings.warn(
                    f"rope_scaling is present but incomplete (type={rope_type}, factor={scaling_factor}). "
                    "Ignoring rope_scaling and using model's max_position_embeddings.",
                    UserWarning
                )


# fix vllm protocol.py check_logprobs issue
protocol.py:255:
    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        # Handle cases where data is bytes or str (Pydantic 2 compatibility)
        if isinstance(data, bytes):
            import json
            data = json.loads(data.decode('utf-8'))
        elif isinstance(data, str):
            import json
            data = json.loads(data)
        # Now data is dict
        if "top_logprobs" in data and data["top_logprobs"] is not None:
            if "logprobs" not in data or data["logprobs"] is False:
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )
            elif not 0 <= data["top_logprobs"] <= 20:
                raise ValueError(
                    "`top_logprobs` must be a value in the interval [0, 20].")
        return data

# Download Qwen1.5-4B-Chat-AWQ model
huggingface-cli download Qwen/Qwen1.5-4B-Chat-AWQ --local-dir ./models/Qwen1.5-4B-Chat-AWQ

# Install vLLM from source
pip install -e .

# start vllm service 
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen1.5-4B-Chat-AWQ \
  --quantization awq \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.80 \
  --host 0.0.0.0 \
  --port 8000

or python test_vllm.py

# request vllm service short request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./models/Qwen1.5-4B-Chat-AWQ",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# request vllm service long request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./models/Qwen1.5-4B-Chat-AWQ",
    "messages": [{"role": "user", "content": "Explain the theory of relativity in simple terms. Repeat this sentence to make it long enough for testing purposes."}],
    "max_tokens": 50
  }'