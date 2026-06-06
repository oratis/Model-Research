# docs · 逐模型深度分析索引

本目录每个文件对一个开源模型做**统一模板**的深度分析。配套源码在仓库根的 [`forks/`](../forks)（git submodule），总表见[根 README](../README.md)。

## 统一文档模板（每篇 9 章）

| 章 | 内容 |
|----|------|
| 1 | **概览** — 出品方、时间线、版本演进、定位、许可证 |
| 2 | **关键链接** — 官方 GitHub / 本地 fork / HuggingFace / 论文 |
| 3 | **模型规格** — 参数 / 层数 / 注意力 / 上下文 / 词表（带来源） |
| 4 | **Benchmark** — 关键分数表（注明版本与来源、是否自报） |
| 5 | **架构剖析** — 注意力 / 位置编码 / 归一化 / MoE / 创新点 |
| 6 | **源码走读** — 关键文件 + `文件:行` 级别的代码讲解 |
| 7 | **训练机制** — 预训练数据/规模 + 后训练（SFT / RLHF / DPO / GRPO …） |
| 8 | **学习要点与横向对比** — 最值得学的点 + 跨模型 `[[链接]]` |
| 9 | **参考** — 带链接的来源清单 |

> 源码走读的 `文件:行号` 引用均已对照 fork 当前提交核对。Qwen / GLM / Phi 的官方仓库未含 `modeling_*.py`，其架构章引用 HuggingFace `transformers` 源码的 URL + 行号（文中已注明）。

## 模型清单

| # | 文档 | 模型 | 一句话亮点 |
|---|------|------|-----------|
| 1 | [01-llama.md](01-llama.md) | **Llama** (Meta) | 经典 decoder 范本；Llama 4 转 MoE + iRoPE + 多模态 |
| 2 | [02-qwen.md](02-qwen.md) | **Qwen** (阿里) | Dense+MoE；QK-Norm；混合思考；四阶段后训练 + 蒸馏 |
| 3 | [03-deepseek.md](03-deepseek.md) | **DeepSeek** | MLA + DeepSeekMoE + MTP + FP8；R1 用 GRPO 纯 RL |
| 4 | [04-mistral.md](04-mistral.md) | **Mistral** | 滑窗注意力 + 环形 KV cache；Mixtral 稀疏 MoE |
| 5 | [05-gemma.md](05-gemma.md) | **Gemma** (Google) | 蒸馏即预训练；局部/全局注意力；soft-capping |
| 6 | [06-phi.md](06-phi.md) | **Phi** (微软) | "Textbooks Are All You Need"，数据为王；LongRoPE |
| 7 | [07-glm.md](07-glm.md) | **GLM** (智谱) | deep&thin 大 MoE；面向 Agent；slime RL 框架 |
| 8 | [08-olmo.md](08-olmo.md) | **OLMo** (AllenAI) | 完全开放；**最佳预训练训练循环阅读样本** |
| 9 | [09-minimax.md](09-minimax.md) | **MiniMax** | Lightning 线性注意力 + Softmax 混合；1M→4M 上下文 |

## 推荐阅读顺序

1. **经典稠密**：[Llama](01-llama.md) → [OLMo](08-olmo.md)（读完整 `train.py`）
2. **高效注意力**：[Mistral](04-mistral.md) → [Gemma](05-gemma.md) → [MiniMax](09-minimax.md)
3. **稀疏 MoE**：[Mistral](04-mistral.md) → [Qwen](02-qwen.md) → [GLM](07-glm.md) → [DeepSeek](03-deepseek.md)
4. **推理与对齐（RL）**：[DeepSeek-R1](03-deepseek.md) → [Qwen](02-qwen.md) → [OLMo](08-olmo.md)
5. **数据与小模型**：[Phi](06-phi.md) → [Gemma](05-gemma.md)
