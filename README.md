# Model-Research · 开源大模型源码与训练机制学习仓库

> 一个用于**系统学习主流开源大语言模型**的研究仓库：把每个模型的官方源码 **fork** 进来（`forks/` 下以 git submodule 形式引用我的 fork），并在 [`docs/`](docs/) 中逐一做**架构剖析 + 源码走读 + 训练机制**的深度分析。
>
> 目标读者：想真正搞懂"这些模型是怎么搭起来、又是怎么训练出来的"的工程师 / 研究者 / 学生。
>
> 📅 数据截至 **2026-06**。Benchmark 与参数均尽量取自**官方来源**（HuggingFace model card、官方 GitHub README、arXiv 论文/技术报告）并标注；标"自报"者为厂商自测，跨模型不可直接横比。

---

## 📦 仓库结构

```
Model-Research/
├── README.md            # 本文件：模型目录 / 参数 / benchmark / 链接总表
├── docs/                # 逐模型深度分析（架构 + 源码走读 + 训练机制）
│   ├── README.md        # 文档索引与推荐学习路线
│   ├── 01-llama.md  ...  09-minimax.md   # 9 个模型分析
│   └── 10-how-to-train.md                # 附：动手训练方案与成本
├── forks/               # 各模型官方源码的 fork（git submodule，浅克隆）
│   ├── llama-models/  Qwen3/  DeepSeek-V3/  DeepSeek-R1/  mistral-inference/
│   └── gemma_pytorch/  PhiCookBook/  GLM-4.5/  OLMo/  MiniMax-01/
└── .gitmodules          # submodule 指向 github.com/oratis/<repo>（fork 自各官方仓库）
```

- **`forks/` 是我（`oratis`）在 GitHub 上对各官方仓库的 fork**，再以 submodule 引用进来；既保留了"fork 归属于你的账号"，又不把他人代码重新托管进本仓库主干。拉取方式见[末尾](#-如何获取源码)。
- 部分仓库（Qwen、GLM、Phi）官方只放了**文档 / 推理脚本 / 微调配方**，模型 `modeling_*.py` 在 HuggingFace `transformers` 里——对应文档的"源码走读"章会**直接引用 transformers 源码的 URL + 行号**，并已注明。

---

## 🗂️ 模型总览

| # | 模型 | 出品方 | 架构类型 | 代表版本 & 参数 | 上下文 | 许可证 | 分析文档 |
|---|------|--------|----------|------------------|--------|--------|----------|
| 1 | **Llama** | Meta | Dense → MoE（L4）+ 多模态 | Llama 3.3 70B；Llama 4 Scout 109B-A17B/16E、Maverick 400B-A17B/128E | 128K；Scout 10M / Maverick 1M | Llama Community（半开放） | [01-llama](docs/01-llama.md) |
| 2 | **Qwen** | 阿里巴巴 | Dense + MoE | Qwen3 dense 0.6–32B；MoE 235B-A22B、30B-A3B | 32K 原生 / 128K（YaRN，2507 系达 256K–1M） | Apache-2.0 | [02-qwen](docs/02-qwen.md) |
| 3 | **DeepSeek** | 深度求索 | MoE（**MLA**）+ RL 推理 | V3 671B-A37B；R1 671B-A37B（蒸馏 1.5–70B） | 128K（V3.2 160K） | MIT（代码）/ 模型协议 | [03-deepseek](docs/03-deepseek.md) |
| 4 | **Mistral / Mixtral** | Mistral AI | Dense（**SWA**）+ MoE | Mistral 7B；Mixtral 8x7B / 8x22B；Small 3 24B；NeMo 12B | 8K → 128K | Apache-2.0 | [04-mistral](docs/04-mistral.md) |
| 5 | **Gemma** | Google DeepMind | Dense（局部/全局 + 蒸馏）+ 多模态 | Gemma 2（2B/9B/27B）；Gemma 3（1B/4B/12B/27B，多模态） | 128K（1B 为 32K） | Gemma Terms | [05-gemma](docs/05-gemma.md) |
| 6 | **Phi** | Microsoft | Dense + MoE（**数据为王**） | Phi-4 14B；Phi-3.5-mini 3.8B；Phi-3.5-MoE 16×3.8B | 128K（Phi-4 base 16K） | MIT | [06-phi](docs/06-phi.md) |
| 7 | **GLM** | 智谱 Z.ai（清华系） | MoE（**deep & thin**）+ Agentic | GLM-4.5 355B-A32B；Air 106B-A12B；GLM-4.6（200K）；GLM-4-9B | 128K（4.6 为 200K） | MIT | [07-glm](docs/07-glm.md) |
| 8 | **OLMo** | AllenAI | Dense（**完全开放**） | OLMo 2：1B / 7B / 13B / 32B | 4K（→ 部分扩展） | Apache-2.0 | [08-olmo](docs/08-olmo.md) |
| 9 | **MiniMax** | MiniMax | **Hybrid 线性+Softmax** MoE | Text-01 456B-A45.9B；M1（推理，CISPO） | 1M 训练 / 4M 推理外推 | 01 自定义 / M1 Apache-2.0 | [09-minimax](docs/09-minimax.md) |

> 参数后缀含义：`A` = 激活参数（MoE 每 token 实际参与计算的量）；`E` = 专家数。例如 `355B-A32B` 表示总参 355B、激活 32B。

---

## 🔗 源码与资源地址

| 模型 | 官方 GitHub | HuggingFace | 论文 / 技术报告 | 本地 fork（submodule） |
|------|-------------|-------------|------------------|------------------------|
| Llama | [meta-llama/llama-models](https://github.com/meta-llama/llama-models) | [meta-llama](https://huggingface.co/meta-llama) | [Llama 2 (2307.09288)](https://arxiv.org/abs/2307.09288) · [Llama 3 Herd (2407.21783)](https://arxiv.org/abs/2407.21783) | `forks/llama-models` |
| Qwen | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | [Qwen](https://huggingface.co/Qwen) | [Qwen3 (2505.09388)](https://arxiv.org/abs/2505.09388) · [Qwen2.5 (2412.15115)](https://arxiv.org/abs/2412.15115) | `forks/Qwen3` |
| DeepSeek | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | [deepseek-ai](https://huggingface.co/deepseek-ai) | [V3 (2412.19437)](https://arxiv.org/abs/2412.19437) · [R1 (2501.12948)](https://arxiv.org/abs/2501.12948) | `forks/DeepSeek-V3`、`forks/DeepSeek-R1` |
| Mistral | [mistralai/mistral-inference](https://github.com/mistralai/mistral-inference) | [mistralai](https://huggingface.co/mistralai) | [Mistral 7B (2310.06825)](https://arxiv.org/abs/2310.06825) · [Mixtral (2401.04088)](https://arxiv.org/abs/2401.04088) | `forks/mistral-inference` |
| Gemma | [google/gemma_pytorch](https://github.com/google/gemma_pytorch) | [google](https://huggingface.co/google) | [Gemma (2403.08295)](https://arxiv.org/abs/2403.08295) · [Gemma 2 (2408.00118)](https://arxiv.org/abs/2408.00118) · [Gemma 3 (2503.19786)](https://arxiv.org/abs/2503.19786) | `forks/gemma_pytorch` |
| Phi | [microsoft/PhiCookBook](https://github.com/microsoft/PhiCookBook) | [microsoft](https://huggingface.co/microsoft) | [Textbooks (2306.11644)](https://arxiv.org/abs/2306.11644) · [Phi-3 (2404.14219)](https://arxiv.org/abs/2404.14219) · [Phi-4 (2412.08905)](https://arxiv.org/abs/2412.08905) | `forks/PhiCookBook` |
| GLM | [zai-org/GLM-4.5](https://github.com/zai-org/GLM-4.5) | [zai-org](https://huggingface.co/zai-org) | [GLM-4.5 (2508.06471)](https://arxiv.org/abs/2508.06471) · [GLM-130B (2210.02414)](https://arxiv.org/abs/2210.02414) | `forks/GLM-4.5` |
| OLMo | [allenai/OLMo](https://github.com/allenai/OLMo) | [allenai](https://huggingface.co/allenai) | [OLMo (2402.00838)](https://arxiv.org/abs/2402.00838) · [OLMo 2 (2501.00656)](https://arxiv.org/abs/2501.00656) | `forks/OLMo` |
| MiniMax | [MiniMax-AI/MiniMax-01](https://github.com/MiniMax-AI/MiniMax-01) | [MiniMaxAI](https://huggingface.co/MiniMaxAI) | [MiniMax-01 (2501.08313)](https://arxiv.org/abs/2501.08313) · [MiniMax-M1 (2506.13585)](https://arxiv.org/abs/2506.13585) | `forks/MiniMax-01` |

---

## 📊 Benchmark 速览

> ⚠️ 以下多为各厂商**自报**分数，且版本/设置不一，**仅供了解量级，不能直接横向排名**。完整表格、版本与来源见各模型文档第 4 章。

| 模型（代表版本） | MMLU / MMLU-Pro | 数学（GSM8K / MATH / AIME） | 代码（HumanEval / LiveCodeBench / SWE-bench） | 备注 |
|------------------|------------------|------------------------------|-----------------------------------------------|------|
| Llama 4 Maverick | MMLU-Pro **80.5** | — / MATH 61.2 / — | — | MMMU 73.4（多模态） |
| Llama 3.3 70B | MMLU 86.0 | GSM8K — / MATH **77.0** | HumanEval **88.4** | |
| Qwen3 235B-A22B（think） | MMLU-Pro **68.2** · GPQA 71.1 | AIME'25 **81.5** | LiveCodeBench 70.7 · BFCL 70.8 | 混合思考模式 |
| DeepSeek-V3 | MMLU **88.5** | MATH-500 90.2 / AIME'24 39.2 | — | 非推理基座 |
| DeepSeek-R1 | MMLU 90.8 | MATH-500 **97.3** / AIME'24 **79.8** | Codeforces 高分 | 推理模型 |
| Mixtral 8x7B | MMLU 70.6 | GSM8K 74.4 / MATH 28.4 | HumanEval 40.2 | 论文数 |
| Gemma 3 27B (PT) | MMLU 78.6 | GSM8K 82.6 / MATH 50.0 | — | 多模态 |
| Phi-4 14B | MMLU 84.8 · MMLU-Pro 70.4 | GSM8K 80.6 / MATH **80.4** | HumanEval 82.6 | 14B 小模型 |
| GLM-4.5 355B-A32B | MMLU-Pro **84.6** | AIME'24 **91.0** | LiveCodeBench 72.9 · **SWE-bench 64.2** · TAU-bench 70.1 | Agentic 强 |
| OLMo 2 32B (base) | MMLU 74.9 | — | — | ARC-C 90.4 / HellaSwag 89.7 |
| MiniMax-Text-01 | MMLU 88.5 | — | — | **RULER-1M 0.910** · LongBench-v2 56.5（超长上下文） |
| MiniMax-M1 | — | AIME'24 **86.0** | **SWE-bench 56.0** | 推理 + 长上下文 |

---

## 📚 逐模型一句话导读

1. **[Llama](docs/01-llama.md)** — 开源生态的"事实标准"。经典 decoder（RMSNorm + RoPE + **GQA** + SwiGLU）的范本；Llama 4 转向 **MoE + iRoPE（NoPE 长上下文）+ 原生多模态**。训练上看 Llama 2 的 **RLHF**（拒绝采样 + PPO）到 Llama 3 的 **DPO** 演进。
2. **[Qwen](docs/02-qwen.md)** — 中文/多语最强开源族之一。Dense 全尺寸 + **MoE**；**QK-Norm 去 QKV-bias**；**混合思考模式**（可设 thinking budget）；36T tokens 预训练 + **四阶段后训练** + 小模型 **strong-to-weak 蒸馏**。
3. **[DeepSeek](docs/03-deepseek.md)** — 算法创新密度最高。**MLA**（KV 低秩压缩）、**DeepSeekMoE**（细粒度 + 共享专家 + **无辅助损失负载均衡**）、**MTP**、**FP8** 训练；**R1** 用 **GRPO** 纯 RL 涌现长链推理并蒸馏到 dense。**首选精读**。
4. **[Mistral](docs/04-mistral.md)** — 高效注意力的代表。**滑动窗口注意力（SWA）+ 环形 KV cache**；Mixtral 把 **Sparse MoE（top-2/8）** 带火。源码最干净、最适合逐行读 MoE。
5. **[Gemma](docs/05-gemma.md)** — **知识蒸馏即预训练**的标杆。**局部/全局交替注意力（5:1）**、**logit soft-capping**、**双重 RMSNorm**、256k 大词表；Gemma 3 加 SigLIP 多模态 + 128K。
6. **[Phi](docs/06-phi.md)** — **"Textbooks Are All You Need"**：用合成 + 重筛数据让小模型对标大模型。看**数据策划**怎样四两拨千斤，以及 **LongRoPE**、pivotal-token **DPO**。
7. **[GLM](docs/07-glm.md)** — **deep & thin** 的大 MoE，**面向 Agent**。partial RoPE + 选择性 QK-Norm + **MTP** + loss-free 路由；自研异步 RL 框架 **slime**；混合推理。
8. **[OLMo](docs/08-olmo.md)** — **完全开放**（数据 Dolma + 代码 + 全部 checkpoint + 训练日志）。**想读真实预训练训练循环 `train.py`、看 FSDP/Z-loss/LR 调度怎么落地，这是最佳样本**。
9. **[MiniMax](docs/09-minimax.md)** — **线性注意力**工程化。**Lightning Attention 与 Softmax 混合（7:1）** 把上下文推到 **1M→4M**；M1 用 **CISPO** 在超长上下文上做高效 RL。

---

## 🧭 推荐学习路线

按"由浅入深、由架构到训练"的顺序：

1. **打底 · 经典稠密解码器**：[Llama](docs/01-llama.md) → [OLMo](docs/08-olmo.md)（顺带把**完整训练循环**读了）
2. **高效注意力**：[Mistral](docs/04-mistral.md)（滑窗）→ [Gemma](docs/05-gemma.md)（局部/全局）→ [MiniMax](docs/09-minimax.md)（线性注意力）
3. **稀疏专家（MoE）**：[Mistral/Mixtral](docs/04-mistral.md) → [Qwen](docs/02-qwen.md) → [GLM](docs/07-glm.md) → [DeepSeek](docs/03-deepseek.md)（MLA + 无辅助损失均衡，集大成）
4. **推理与对齐（RL）**：[DeepSeek-R1](docs/03-deepseek.md)（GRPO）→ [Qwen](docs/02-qwen.md)（四阶段）→ [OLMo](docs/08-olmo.md)（Tülu：SFT/DPO/RLVR）
5. **数据与小模型**：[Phi](docs/06-phi.md)（数据为王）→ [Gemma](docs/05-gemma.md)（蒸馏）

详见 [docs/README.md](docs/README.md)。

---

## 🛠️ 想自己动手训练？

看 **[docs/10-how-to-train.md](docs/10-how-to-train.md)** —— 参考本仓库这些模型，**训练你自己的大模型**的完整方案与成本：

- **路线分层**：A 微调（$几十~几千）/ B 从零小预训练（$几千~几百万）/ C 前沿（$百万~千万+）
- **六大要素**：数据 · 算力 · 框架 · 配方 · 团队 · 时间
- **成本详算**：`训练算力 ≈ 6ND` 公式 + 规模/成本对照表 + 三套完整预算
- **起步包**：1 张 H100 + nanotron + FineWeb-Edu，几百美金跑通从预训练到对齐

---

## 🚀 如何获取源码

各模型源码以 **git submodule（浅克隆）** 引用，clone 本仓库后按需拉取：

```bash
# 克隆本仓库
git clone https://github.com/oratis/Model-Research.git
cd Model-Research

# 拉取全部 fork 源码（浅克隆，省空间）
git submodule update --init --recursive --depth 1

# 或只拉某一个，例如 DeepSeek-V3
git submodule update --init --depth 1 forks/DeepSeek-V3
```

> 💡 **PhiCookBook** 上游含约 10GB 的图片/notebook 历史，本仓库对其用 **blobless + sparse** 仅检出 Markdown 教程；如需完整内容请到[官方仓库](https://github.com/microsoft/PhiCookBook)。
>
> 各 `forks/<repo>` 指向 `github.com/oratis/<repo>`（我对官方仓库的 fork）。

---

## ⚖️ 说明与免责

- `forks/` 下所有源码**版权归各自原作者/组织所有**，遵循其各自许可证（见上表与各仓库 LICENSE）。本仓库仅以 submodule 引用并用于**学习研究**，未修改其内容。
- `docs/` 下分析为基于公开源码与论文的**学习笔记**，可能存在理解偏差；关键结论请以**官方论文 / 源码**为准。文中已对无法核实之处标注"未核实/约"。
- Benchmark 多为厂商自报，设置不一，**请勿据此直接横向排名**。

---

*🤖 本仓库由 [Claude Code](https://claude.com/claude-code) 协助构建：fork 官方源码 + 逐模型源码走读与训练机制分析。*
