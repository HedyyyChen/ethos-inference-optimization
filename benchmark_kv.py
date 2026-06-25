import subprocess
import time
import os

# 基础命令配置
base_cmd = [
    "ethos", "infer",
    "--test", "icu_readmission",
    "--model", "out/mimic_layer_6_batch_32_do_0.3/best_model.pt",
    "--data", "tokenized_datasets/mimic_test_timelines_p26758.hdf5",
    "--vocab", "mimic_vocab_t4367.pkl",
    "--n_tokens", "1",
    "--n_jobs", "4",
    "--n_gpus", "2",
    "--device", "cuda",
    "--output", "results_benchmark",
    "--model_name", "ethos_official"
]

def run_benchmark(mode: str, iterations: int = 5):
    """
    运行基准测试。
    mode: "KV_OFF", "KV_ON", "KV_ON_INT8"
    """
    if mode == "KV_OFF":
        flags = ["--no_kv_cache"]
    elif mode == "KV_ON":
        flags = ["--use_kv_cache"]
    elif mode == "KV_ON_FP16":
        flags = ["--use_kv_cache", "--kv_cache_fp16"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results = []

    print(f"\n{'='*20} Starting Benchmark: {mode} {'='*20}")
    
    for i in range(6, iterations + 1):
        suffix = f"{mode.lower()}_iter_{i}"
        cmd = base_cmd + flags + ["--suffix", suffix]
        
        start_time = time.time()
        process = subprocess.run(cmd) # 移除 capture_output=True 从而显示进度条
        end_time = time.time()
        
        duration = end_time - start_time
        status = "SUCCESS" if process.returncode == 0 else f"FAILED (code {process.returncode})"
        results.append({
            "mode": mode,
            "iteration": i,
            "duration": f"{duration:.2f}s",
            "status": status
        })
    return results

if __name__ == "__main__":
    # 结果保存路径
    output_dir = "/home/ligong1/mednlp/ethos/ethos-paper/results"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "benchmark_report_4.txt")
    
    # 运行测试：增加 INT8 量化模式
    all_results = []
    all_results.extend(run_benchmark("KV_ON_FP16", iterations=30))
    
    # 将结果保存到文件
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== ETHOS KV Cache Benchmark Report ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Mode':<10} | {'Iter':<5} | {'Duration':<10} | {'Status'}\n")
        f.write("-" * 50 + "\n")
        for res in all_results:
            f.write(f"{res['mode']:<10} | {res['iteration']:<5} | {res['duration']:<10} | {res['status']}\n")
    
    print(f"Benchmark finished. Results saved to: {os.path.abspath(report_path)}")
