# ETHOS-Enhanced: Optimized Large-scale Healthcare Model Inference

本项目基于 2024 年发表的 **ETHOS (Enhanced Transformer for Health Outcome Simulation)** 模型进行了深度工程优化和任务扩展。旨在提升大模型在资源受限环境（如消费级显卡 RTX 4060Ti）下的推理效率。
## 核心改进 (Core Optimizations)

针对原模型解码器架构中存在的计算开销和显存瓶颈，我们实施了以下三项核心技术优化：

1.  **KV Caching (Key-Value 缓存机制)**
    *   **原理**：解决了原模型在自回归生成时的 **Quadratic attention cost**（平方级注意力计算冗余）。
    *   **效果**：单患者生成 1000 tokens 的平均时间从 **20.05s 降低至 2.86s**，提速约 **85%**。

2.  **Model Quantization (INT8/FP16 量化)**
    *   **优化**：将模型参数从 FP32 压缩至 INT8/FP16，实现 **75% 的显存空间节省**（4 Bytes → 1 Byte）。
    *   **意义**：使大模型能够顺畅运行在 8GB 显存的消费级显卡上，大幅降低了医疗 AI 的部署门槛。

3.  **Generation Truncation (生成截断策略)**
    *   **策略**：针对推理过程中的“长尾效应”，设定 `max_token=2166` 阈值（覆盖 95% 以上患者）。
    *   **效果**：在牺牲极小精度的情况下，将批量处理（200样本）的总耗时缩短了 **40%**。

## 性能表现 (Performance Benchmarks)

基于 MIMIC-IV 数据集的测试结果对比：

| 推理配置 | 200患者平均时间 (秒) | 时间减少率 |
| :--- | :--- | :--- |
| **Original (FP32)** | 1540.88s | - |
| **with KV Cache** | 1218.50s | 21% ↓ |
| **with INT8 Quantization** | 1037.28s | 33% ↓ |
| **with Truncation** | 917.42s | **40% ↓** |
