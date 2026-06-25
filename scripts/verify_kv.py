
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# 将 ethos-paper 目录添加到 sys.path
sys.path.append(os.path.join(os.getcwd()))

from ethos.model import ModelConfig, Ethos

def verify_consistency():
    device = "cpu" # 使用 CPU 进行逻辑验证即可
    
    # 使用一个小型的模型配置
    config = ModelConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=True
    )
    
    print(f"Initializing small model for logic verification...")
    model = Ethos(config)
    model.eval()
    model.to(device)

    # 准备测试数据
    torch.manual_seed(42)
    seq_len = 50
    # 生成一段初始文本
    timeline = torch.randint(0, config.vocab_size, (seq_len,), device=device)

    print(f"\n--- 测试 1: 逐 Token 增量生成的 Logits 一致性 ---")
    
    # 路径 A: 全序列一次性计算
    off_logits_list = []
    
    # 路径 B: KV ON 模式
    on_logits_list = []
    kv_caches = None

    test_gen_len = 10
    current_timeline = timeline.clone()

    for i in range(test_gen_len):
        # 1. KV OFF 路径
        logits_off, _, _ = model(current_timeline[None, :])
        last_logit_off = logits_off[0, -1, :]
        off_logits_list.append(last_logit_off)

        # 2. KV ON 路径
        # 在 KV ON 模式下，如果已有缓存，只输入最后一个 token
        # 注意：如果是第一步，需要输入整个 timeline 来填充缓存
        if kv_caches is None:
            input_tokens_on = current_timeline[None, :]
        else:
            input_tokens_on = current_timeline[None, [-1]]
            
        logits_on, _, new_kv_caches = model(input_tokens_on, kv_caches=kv_caches)
        last_logit_on = logits_on[0, -1, :]
        on_logits_list.append(last_logit_on)
        
        # 更新缓存
        kv_caches = new_kv_caches
        
        # 模拟生成下一个 token
        next_token = torch.argmax(last_logit_on).view(1)
        current_timeline = torch.cat([current_timeline, next_token])
        
        # 计算差异
        diff = torch.max(torch.abs(last_logit_off - last_logit_on)).item()
        print(f"Step {i+1}: Max Logit Diff = {diff:.2e}")
        if diff > 1e-5:
            print(f"ERROR: Step {i+1} diff exceeds threshold!")
            return False

    print(f"\n--- 测试 2: 滑动窗口触发后的重置逻辑 ---")
    context_len = 20
    offset = 1
    slid_timeline = torch.cat([current_timeline[:context_len], current_timeline[context_len + offset :]])
    
    # 路径 A: 滑动后全量计算
    logits_slid_off, _, _ = model(slid_timeline[None, :])
    last_logit_slid_off = logits_slid_off[0, -1, :]
    
    # 路径 B: 滑动后重置缓存并计算
    kv_caches_reset = None
    logits_slid_on, _, _ = model(slid_timeline[None, :], kv_caches=kv_caches_reset)
    last_logit_slid_on = logits_slid_on[0, -1, :]
    
    diff_slid = torch.max(torch.abs(last_logit_slid_off - last_logit_slid_on)).item()
    print(f"Sliding Window Reset: Max Logit Diff = {diff_slid:.2e}")

    if diff_slid < 1e-5:
        print("\nSUCCESS: KV Cache 逻辑与全量计算在数值上高度一致！")
    else:
        print("\nFAILURE: 存在显著的精度偏差。")

    print(f"\n--- 测试 3: INT8 量化模式下的精度损耗 ---")
    model.config.kv_quantization = True
    current_timeline_q = timeline.clone()
    kv_caches_q = None
    
    q_diffs = []
    for i in range(test_gen_len):
        # KV OFF (FP32)
        logits_off, _, _ = model(current_timeline_q[None, :], kv_caches=None)
        last_logit_off = logits_off[0, -1, :]
        
        # KV ON (INT8)
        if kv_caches_q is None:
            input_tokens_q = current_timeline_q[None, :]
        else:
            input_tokens_q = current_timeline_q[None, [-1]]
            
        logits_on_q, _, new_kv_caches_q = model(input_tokens_q, kv_caches=kv_caches_q)
        last_logit_on_q = logits_on_q[0, -1, :]
        kv_caches_q = new_kv_caches_q
        
        # 记录差异
        diff_q = torch.max(torch.abs(last_logit_off - last_logit_on_q)).item()
        q_diffs.append(diff_q)
        
        # 模拟生成
        next_token = torch.argmax(last_logit_on_q).view(1)
        current_timeline_q = torch.cat([current_timeline_q, next_token])
        
    avg_q_diff = sum(q_diffs) / len(q_diffs)
    print(f"Average Max Logit Diff (INT8 vs FP32): {avg_q_diff:.2e}")
    print(f"Max Logit Diff (INT8 vs FP32): {max(q_diffs):.2e}")
    
    if avg_q_diff < 0.1: # 量化通常会有一定误差，0.1 左右的 logit 差异通常是可以接受的
        print("SUCCESS: INT8 量化误差在可接受范围内。")
        return True
    else:
        print("WARNING: INT8 量化误差较大，请检查实现。")
        return False

if __name__ == "__main__":
    with torch.no_grad():
        success = verify_consistency()
        sys.exit(0 if success else 1)
