import torch

def main():
    Out, In, GroupSize = 256, 128, 128
    assert In % 8 == 0
    w_fp16 = torch.rand(Out, In, dtype=torch.float16) * 2 - 1

    # 伪量化
    scale = 0.1
    zero = 8.0
    w_int4 = torch.clamp(torch.round(w_fp16 / scale) + zero, 0, 15).to(torch.int32)
    w_packed = torch.zeros(Out, In // 8, dtype=torch.int32)

    for i in range(8):
        col = w_int4[:, i::8]
        w_packed |= (col << (i * 4))

    
    # 验证解包 (Unpack)
    r, c_pack = 10, 5
    val = w_packed[r, c_pack].item()

    print(f"\n  验证位置: Row={r}, PackCol={c_pack}，Packed Hex: 0x{val:08X}")
    print(f"   Packed Hex: 0x{val:08X}")

    extracted = []
    for i in range(8):
        num = (val >> (i * 4)) & 0xF
        extracted.append(num)

    # 获取真值
    true_vals = w_int4[r, c_pack*8 : c_pack*8+8].tolist()
    
    print(f"   解包结果: {extracted}")
    print(f"   原始真值: {true_vals}")
    
    if extracted == true_vals:
        print("   验证打包成功, Little-Endian 位序正确。")
    else:
        print("   打包失败")

    # 显存计算
    mem_fp16 = w_fp16.element_size() * w_fp16.numel()
    mem_packed = w_packed.element_size() * w_packed.numel()
    # 加上 scale/zero 的开销
    mem_scale = (Out * (In // GroupSize)) * 2 
    ratio = mem_fp16 / (mem_packed + mem_scale)
    print(f"\n  显存分析: 原始 {mem_fp16/1024:.1f}KB -> 量化后 {(mem_packed+mem_scale)/1024:.1f}KB (压缩 {ratio:.2f}x)")

if __name__ == "__main__":
    main()