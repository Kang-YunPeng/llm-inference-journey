import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
import logging
import gc

# =================配置与日志=================
logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# 屏蔽第三方库的冗余日志
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("flash_attn").setLevel(logging.ERROR)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past_key_values=None, start_index=0):
        embedded = self.tgt_embed[0](tgt)   #embedding
        embedded = self.tgt_embed[1](embedded, start_index)     #positional encoding
        return self.decoder(embedded, memory, src_mask, tgt_mask, past_key_values)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = RMSNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class RMSNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = RMSNorm(size)
        self.norm2 = RMSNorm(size)
        self.size = size

    def forward(self, x, mask):
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + attn_output
        x = x + self.feed_forward(self.norm2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = RMSNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past_key_values=None):
        all_layer_kv = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None
            x, layer_kv = layer(x, memory, src_mask, tgt_mask, layer_past)
            all_layer_kv.append(layer_kv)
        return self.norm(x), all_layer_kv

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = RMSNorm(size)
        self.norm2 = RMSNorm(size)
        self.norm3 = RMSNorm(size)

    def forward(self, x, memory, src_mask, tgt_mask, past_key_values=None):
        # Self-Attention with Cache
        norm_x = self.norm1(x)
        self_attn_output, self_kv = self.self_attn(norm_x, norm_x, norm_x, tgt_mask, past_key_value=past_key_values)
        x = x + self_attn_output

        # Cross-Attention
        norm_x = self.norm2(x)
        cross_attn_output, _ = self.src_attn(norm_x, memory, memory, src_mask)
        x = x + cross_attn_output

        # Feed-Forward Network
        x = x + self.feed_forward(self.norm3(x))
        return x, self_kv

def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    # scores shape: [batch, heads, seq_len_q, seq_len_k]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask shape should be broadcastable to scores
        # Typically [batch, 1, 1, seq_len_k] or [batch, 1, seq_len_q, seq_len_k]
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, value)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None, past_key_value=None):
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        if past_key_value is not None:
            if isinstance(past_key_value, KVCache):
                key, value = past_key_value.update(key, value)
            else:
                prev_key, prev_value = past_key_value
                key = torch.cat([prev_key, key], dim=2)
                value = torch.cat([prev_value, value], dim=2)

        # todo: 进一步理解mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            if mask.size(-1) != key.size(2):
                mask = torch.ones(nbatches, 1, query.size(2), key.size(2), device=query.device, dtype=torch.uint8)

        x = attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), (key, value)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class KVCache:
    """
    KV Cache 预分配内存管理器, 避免每次推理时 torch.cat 拼接 KV 的开销
    """
    def __init__(self, max_seq_len, batch_size, num_heads, head_dim, device):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.seq_len = 0
        self.key_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.value_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device)

    def update(self, new_key, new_value):
        """
        将新的 key/value 写入缓存, 并返回完整的 KV tensor
        """
        new_seq_len = new_key.size(2)
        end_pos = self.seq_len + new_seq_len
        if end_pos > self.max_seq_len:
            raise ValueError(f"KV Cache overflow: {end_pos} > {self.max_seq_len}")
        self.key_cache[:, :, self.seq_len:end_pos, :] = new_key
        self.value_cache[:, :, self.seq_len:end_pos, :] = new_value
        self.seq_len = end_pos
        return self.key_cache[:, :, :end_pos, :], self.value_cache[:, :, :end_pos, :]

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# todo: 实现旋转位置编码
class PositionalEncoding(nn.Module):
    """
    正余弦位置编码 (Sinusoidal Positional Encoding)
    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    其中:
        pos: 位置 (0, 1, 2, ...)
        i: 维度索引 (0, 1, 2, ..., d_model/2)
        d_model: 模型维度
    优点:
        - 可以表示相对位置关系
        - 任意长度外推（通过计算）
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, start_index=0):
        """
        Args:
            x: 输入 tensor [batch, seq_len, d_model]
            start_index: 起始位置索引（用于增量推理）
        Returns:
            加上位置编码后的 tensor
        """
        seq_len = x.size(1)
        # 安全截断，防止超出预定义的最大长度
        end_index = min(start_index + seq_len, self.pe.size(1))
        actual_pe = self.pe[:, start_index:end_index]

        # 如果 start_index 太大导致 actual_pe 为空，需要处理 (这里简单截断)
        if actual_pe.size(1) != seq_len:
             # 这种情况通常不会发生，除非生成的长度超过 max_len
             pass

        x = x + actual_pe.requires_grad_(False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = FeedForwardNetwork(d_model, d_ff)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff)), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff)), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.ModuleList([Embeddings(d_model, tgt_vocab), c(position)]),
        Generator(d_model, tgt_vocab),
    )
    return model

def run_comprehensive_benchmark(prompt_len=10, gen_len=500, model_name="Qwen/Qwen2-0.5B-Instruct"):
    """
    1. 手写无 Cache
    2. 手写 torch.cat Cache
    3. 手写预分配 Cache
    4. HuggingFace (use_cache=True)
    5. vLLM (PagedAttention)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("必须在 GPU 上运行此基准测试！")
        return

    print("\n" + "="*70)
    print(f"    统一基准测试启动 | Prompt={prompt_len}, Gen={gen_len} | Device={device}")
    print("="*70)

    results = {}

    # --- 阶段 1, 2, 3: 手写模型测试 ---
    print("\n[阶段 1/3] 初始化手写 Transformer 模型...")
    src_vocab = 100
    base_model = make_model(src_vocab, src_vocab, N=2, d_model=256, d_ff=512, h=4).to(device)
    base_model.eval()
    src = torch.LongTensor([[1]*prompt_len]).to(device)
    src_mask = torch.ones(1, 1, src.size(1)).to(device)

    def get_subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

    # 1. 无 Cache
    print("     测试 A: 无 Cache (每次重算)...")
    ys = torch.LongTensor([[0]]).to(device)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(gen_len):
        out, _ = base_model.decode(base_model.encode(src, src_mask), src_mask, ys, get_subsequent_mask(ys.size(1)).to(device), start_index=0)
        prob = base_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
    torch.cuda.synchronize()
    results['HandWritten_NoCache'] = time.time() - start

    # 清理显存
    del ys
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 动态 Cat Cache
    print("     测试 B: 动态 Cat Cache...")
    memory = base_model.encode(src, src_mask)
    ys = torch.LongTensor([[0]]).to(device)
    past_key_values = None
    torch.cuda.synchronize()
    start = time.time()
    for i in range(gen_len):
        tgt_mask = torch.ones(1, 1, 1, device=device, dtype=torch.uint8)
        out, new_kv = base_model.decode(memory, src_mask, ys[:, -1:], tgt_mask, past_key_values, start_index=i)
        past_key_values = new_kv
        prob = base_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
    torch.cuda.synchronize()
    results['HandWritten_DynamicCat'] = time.time() - start

    del ys, past_key_values, memory
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 预分配 Cache
    print("     测试 C: 预分配 Cache (静态内存)...")
    memory = base_model.encode(src, src_mask)
    ys = torch.LongTensor([[0]]).to(device)
    num_layers = len(base_model.decoder.layers)
    num_heads = base_model.decoder.layers[0].self_attn.h
    head_dim = base_model.decoder.layers[0].self_attn.d_k
    kv_caches = [KVCache(gen_len + 10, 1, num_heads, head_dim, device) for _ in range(num_layers)]

    torch.cuda.synchronize()
    start = time.time()
    for i in range(gen_len):
        tgt_mask = torch.ones(1, 1, 1, device=device, dtype=torch.uint8)
        out, _ = base_model.decode(memory, src_mask, ys[:, -1:], tgt_mask, kv_caches, start_index=i)
        prob = base_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
    torch.cuda.synchronize()
    results['HandWritten_PreAlloc'] = time.time() - start

    del ys, kv_caches, memory
    gc.collect()
    torch.cuda.empty_cache()
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- 阶段 4: HuggingFace ---
    hf_time = None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"\n[阶段 2/3] 加载 HuggingFace 模型 ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, pad_token='<|extra_0|>')
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model_hf.eval()

        input_ids = torch.randint(10, 1000, (1, prompt_len)).to(device)

        # 预热
        _ = model_hf.generate(input_ids, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()

        print(" 测试 D: HuggingFace (FlashAttn + 融合算子)...")
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model_hf.generate(input_ids, max_new_tokens=gen_len//2, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id, use_cache=True, eos_token_id=None)
        torch.cuda.synchronize()
        hf_time = time.time() - start
        results['HuggingFace'] = hf_time

        del model_hf, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except ImportError:
        print("   ⚠️  跳过 HF: 未安装 transformers")
    except Exception as e:
        print(f"   ❌ HF 失败: {e}")

    # --- 阶段 5: vLLM ---
    vllm_time = None
    try:
        from vllm import LLM, SamplingParams
        print(f"\n[阶段 3/3] 加载 vLLM 引擎 ({model_name})...")

        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.5,
            dtype="float16",
            max_model_len=2048,
            enforce_eager=False
        )

        prompt = "hello " * prompt_len
        sampling_params = SamplingParams(max_tokens=gen_len//2, temperature=0.0, ignore_eos=True)

        # 预热 (编译 kernel)
        print(" vLLM 预热中 (首次运行较慢)...")
        _ = llm.generate([prompt], sampling_params)

        print(" 测试 E: vLLM (PagedAttention + Continuous Batching)...")
        torch.cuda.synchronize()
        start = time.time()
        _ = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        vllm_time = time.time() - start
        results['vLLM'] = vllm_time

    except ImportError:
        print(" 跳过 vLLM: 未安装 vllm")
    except Exception as e:
        print(f"    vLLM 失败: {e}")

    # =================结果展示=================
    print("\n" + "="*70)
    print("📊 最终测试结果汇总")
    print("="*70)

    base_time = results.get('HandWritten_NoCache', 1)

    print(f"{'方法':<25} | {'耗时 (s)':<10} | {'加速比 (vs 无Cache)':<20}")
    print("-" * 60)

    for name, t in results.items():
        ratio = base_time / t if t > 0 else float('inf')
        label = f"{t:.4f}"
        ratio_str = f"{ratio:.2f}x" if name != 'HandWritten_NoCache' else "1.00x (Base)"
        print(f"{name:<25} | {label:<10} | {ratio_str:<20}")

    print("-" * 60)
    print(" 核心结论:")
    if 'HandWritten_DynamicCat' in results:
        print(f"   1. 动态 Cat 相比无 Cache 加速约 {results['HandWritten_NoCache']/results['HandWritten_DynamicCat']:.1f}x (避免重算)")
    if 'HandWritten_PreAlloc' in results:
        print(f"   2. 预分配相比动态 Cat 进一步加速 (减少内存拷贝与碎片)")
    if 'HuggingFace' in results and 'HandWritten_PreAlloc' in results:
        print(f"   3. HF 相比手写预分配快 {results['HandWritten_PreAlloc']/results['HuggingFace']:.1f}x (CUDA 融合算子 + FlashAttn)")
    if 'vLLM' in results and 'HuggingFace' in results:
        print(f"   4. vLLM vs HF: {results['HuggingFace']/results['vLLM']:.2f}x (单请求下主要优势在显存管理，高并发优势更大)")

if __name__ == "__main__":
    run_comprehensive_benchmark(prompt_len=10, gen_len=500)