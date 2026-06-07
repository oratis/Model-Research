# 附：从零到一 —— 参考开源模型训练你自己的大模型（方案与成本）

> 本文是 Model-Research 的**配套实操指南**，基于本仓库对 9 个开源模型的架构与训练机制分析，给出从「微调现成底座」到「从零预训练」的**路线分层、要素清单、成本详算与起步包**。
>
> ⚠️ 成本为 **2026 年中的规划级估算**（GPU 租赁价随厂商/地区/承诺用量浮动 3–5 倍），用于判断量级，非报价；不构成财务/法律意见。

---

## 1. 先定位路线：你说的"训练"是哪一种？

**最关键的决策。** 它决定了后面一切成本——"训练一个大模型"在花费上能差 6 个数量级。本仓库研究的模型（Llama / Qwen / DeepSeek…）几乎都属于**路线 C**：几百人团队、上万张卡的产物。对个人/小团队，现实的是 A 与小号的 B。

| 路线 | 含义 | 典型规模 | 现实成本 | 适合谁 |
|---|---|---|---|---|
| **A. 微调 / 继续训练** | 拿现成开源底座做 SFT / LoRA / 继续预训练 | 复用 0.5B–70B | **$几十 ～ 几千** | 绝大多数人 ✅ |
| **B. 从零预训练（中小）** | 从随机初始化训 0.1B–8B | 0.1B–8B | **$几千 ～ 几百万** | 教学 / 科研 / 领域专用 |
| **C. 从零训练前沿** | 训 30B–500B+ 通用大模型 | 30B–500B+ | **$数百万 ～ 数千万+** | 有融资的团队 / 大厂 |

**推荐心法**：先 A 跑通流程 → 再用 B 训一个 0.1–1B 完整复现 `pretrain→SFT→对齐`（最好的学习，几百~几千美金）→ 有资源/团队再谈 C。

---

## 2. 训练全景：要覆盖的流水线

```
①数据 → ②预训练 Pretrain → ③中训练 Mid-train/退火 → ④SFT 指令微调
        → ⑤偏好对齐 RLHF / DPO / GRPO → ⑥评测 Eval → ⑦推理部署
```

- 路线 A 通常只做 ④⑤⑥⑦（甚至只 ④）；路线 B/C 从 ① 全做。
- 对应本仓库：完整训练循环看 [[08-olmo]]（`train.py` 逐段）；RL 对齐看 [[03-deepseek]]（GRPO）；数据策略看 [[06-phi]]（数据为王）。

---

## 3. 要准备的六大要素

### 3.1 数据 Data —— 真正的瓶颈，不是 GPU

| 子项 | 要点 | 参考 |
|---|---|---|
| **规模** | Chinchilla 最优 ≈ **20× 参数**；现代模型远超此（Llama 3 ≈ **1875 tokens/参数**，8B 训 15T） | [[01-llama]] |
| **来源** | 开源语料：**FineWeb / FineWeb-Edu（15T+）、Dolma（3–5T）、The Stack v2（代码）、RedPajama、C4**；中文：WuDao、Chinese-FineWeb、CCI | [[08-olmo]] |
| **清洗** | 去重（MinHash / exact）、质量过滤（分类器）、去毒、PII 脱敏、语言/格式过滤——**决定模型上限** | |
| **配比 & 课程** | 网页/代码/数学/多语的混合比例 + 后期高质量"退火"（OLMo2 / Qwen 都这么做） | [[02-qwen]] |
| **合成数据** | Phi 的核心招式：用强模型生成"教科书级"数据，小团队弯道超车 | [[06-phi]] |
| **Tokenizer** | 自训 BPE（128k–256k 词表）或复用现成；大词表利于多语 | [[05-gemma]] |
| **⚠️ 合规** | 数据许可！CC-BY-**NC** 不能商用训练，代码/书籍/网页各有版权风险（见仓库 README 的协议说明） | |

**成本**：语料免费，但清洗/去重要大量 CPU + 存储 + 人力。原始+清洗数据常 **10–100+ TB**（对象存储约 **$15–25/TB·月**）；CPU 清洗几千~几万 core-hours（**$几百~几万**）；**数据工程人力往往是最大隐性成本**。

### 3.2 算力 Compute —— 钱主要烧在这里

**GPU 选型（2026 主流，租赁价 on-demand 约值）**：

| GPU | 显存 | 适合 | 租赁价 |
|---|---|---|---|
| A100 80GB | 80G | 入门 / 微调 | $1–2 /卡·时 |
| **H100 80GB** | 80G | 主力 | **$2–4**（承诺用量 $1.5–2.5） |
| H200 141GB | 141G | 大模型 / 长上下文 | $3–5 |
| B200 / GB200 | 180G+ | 2025–26 新一代，最快 | $4–8 |
| 国产（昇腾 910B 等） | — | 合规 / 供应受限 | 经国内云 |

**租 vs 买**：起步**租云**（Lambda / CoreWeave / RunPod / 三大云），无前期投入；**买**：H100 单卡 ~**$25k–35k**，8×H100 整机 ~**$250k–400k**，再加 InfiniBand / 机房 / 电 / 散热——除非长期满载否则租更划算。**多机训练必须 InfiniBand / NVLink 高速互联**，否则并行效率崩。

**算力估算（务必记住）**：

```
训练算力 ≈ 6 × N(参数) × D(tokens)  FLOPs
GPU·小时 = 6ND / (单卡FLOPS × MFU)
```

以 H100（BF16 峰值 ~990 TFLOP/s）、MFU 40%（有效 ~400 TFLOP/s）估：

| 模型 N | tokens D | GPU·小时(H100) | 算力成本 @$2 | 类比 |
|---|---|---|---|---|
| 0.1B | 10B | ~4 | **~$10** | 玩具 / 教学 |
| 0.5B | 100B | ~210 | **~$420** | 周末项目 |
| 1B | 1T | ~4.2k | **~$8k** | 小领域模型 |
| 7B | 1T | ~29k | **~$58k** | 入门可用 |
| 7–8B | 15T | ~440k | **~$0.9M**（实际 $1–2.6M） | Llama 3 8B 级 |
| 70B | 15T | ~4.4M | **~$9M**（实际 $13M+） | Llama 3 70B 级 |
| 405B | 15T | ~25–30M | **~$50–60M+** | 前沿稠密 |

> ⚠️ **实际 ≈ 表中 ×1.5–3**：MFU 达不到、失败重跑、调参实验、数据/存储/人力都没算进去。
> 💡 **省钱杠杆**：**FP8 训练**（DeepSeek-V3 做法，见 [[03-deepseek]]）再降 ~30–50%；**MoE** 同等效果激活参数更少（[[03-deepseek]] / [[09-minimax]] / [[07-glm]]）。

### 3.3 软件栈 Framework

| 用途 | 选型 |
|---|---|
| **预训练框架** | **Megatron-LM**（大规模主力）、**TorchTitan**（PyTorch 原生）、**nanotron**（HF，轻量适合学习）、**DeepSpeed**、NVIDIA **NeMo**；[[08-olmo]] 的训练代码是极好范本 |
| **并行** | DP / 张量并行 TP / 流水线 PP / 专家并行 EP / 上下文并行 CP；显存优化 **ZeRO**、**FSDP**、激活重计算 |
| **微调（路线 A）** | **LLaMA-Factory**、**Axolotl**、HF **TRL / PEFT**（LoRA/QLoRA）、Unsloth |
| **对齐** | TRL（DPO/PPO/GRPO）、OpenRLHF、verl、DeepSpeed-Chat |
| **推理 / 评测** | **vLLM** / SGLang、**lm-evaluation-harness**、OpenCompass（中文）、lighteval |
| **实验管理** | Weights & Biases / TensorBoard、检查点、容错重启 |

软件开源免费，成本在**工程师时间**。

### 3.4 算法 / 配方 Recipe（抄本仓库模型的作业）

| 决策 | 选项 & 参考 |
|---|---|
| **Dense or MoE** | 预算有限/求简单 → Dense（[[01-llama]] / [[08-olmo]]）；求高性价比 → MoE（[[03-deepseek]] / [[02-qwen]] / [[07-glm]] / [[09-minimax]]），工程更复杂 |
| **注意力** | GQA（标配，省 KV cache）；长上下文 → 滑窗（[[04-mistral]]）/ 局部全局（[[05-gemma]]）/ 线性（[[09-minimax]]）/ MLA（[[03-deepseek]]） |
| **位置编码** | RoPE（+ YaRN / LongRoPE 扩长上下文） |
| **归一化 / 激活** | RMSNorm + SwiGLU（几乎所有现代模型） |
| **稳定性** | QK-Norm、Z-loss、reordered-norm（OLMo2 / Gemma2 / Qwen3 的稳训招式） |
| **超参** | 峰值 LR ~3e-4、warmup、cosine/WSD 调度、global batch、weight decay、grad clip——直接抄开源 `config` |

参考实现就在已 fork 的代码里：`forks/OLMo/configs/official/`、`forks/DeepSeek-V3/inference/model.py` 等。

### 3.5 团队 / 人 —— 最被低估的成本
- 路线 B/C 需要：数据工程、训练/基础设施、算法、评测——至少 3–10 人。
- 资深 ML/Infra 工程师 **$15万–40万/人·年**，是大项目里**仅次于算力**的开销。
- 路线 A 一个人即可。

### 3.6 时间
- 微调（A）：几小时 ~ 几天。
- 小模型预训练（1B / 100–300B tokens）：单 8×H100 节点 几天 ~ 两周。
- 8B / 15T：几百卡 数周 ~ 1–2 月。
- 70B+：上千卡 1–3 月。

---

## 4. 成本详算：三套完整预算

### 💚 方案 A：微调一个 8B 开源底座（最现实）
| 项 | 估算 |
|---|---|
| 算力（LoRA 微调 8B，几十万样本，几天） | $50 – $2,000 |
| 数据（采集/标注/合成） | $0 – $10,000 |
| 存储 / 杂项 | ~$100 |
| 人力 | 你自己 / 1 人 |
| **合计** | **$几百 – 1 万级**，1 张租来的 H100 一周内 |

### 💛 方案 B：从零训一个 1–3B 领域模型（教学 / 科研）
| 项 | 估算 |
|---|---|
| 算力（1B×~300B ≈ 1.3k GPU·时；3B×1T ≈ 1.2万 GPU·时） | **$3,000 – $30,000** |
| 数据清洗（CPU + 存储 10–30TB） | $1,000 – $10,000 |
| 调参 / 失败重跑缓冲 | 上面再 +50–100% |
| 人力（1–3 人 × 1–3 月） | 时间成本为主 |
| **合计** | **$1 万 – 10 万级** |

### ❤️ 方案 C：从零训一个 8B（Llama 3 8B 级，"能打"的通用模型）
| 项 | 估算 |
|---|---|
| 预训练算力（8B×15T，~44万 GPU·时，含 overhead） | **$1M – $2.6M** |
| 数据（大规模清洗 / 合成 / 许可） | $50k – $500k |
| 后训练（SFT+RL，算力 + 偏好数据标注） | $50k – $300k |
| 存储 / 网络 / 容错 | $20k – $100k |
| 团队（5–10 人 × 半年） | **$0.5M – $2M+** |
| **合计** | **$2M – $6M+** |
| 70B 级 | 在此 **×5–10**（$1,000–3,000 万） |
| 405B / 前沿 | **$5,000 万 – 上亿** |

---

## 5. 推荐务实路径

```
第1步  微调跑通流程      LLaMA-Factory 对 Qwen3-1.7B/8B 做 LoRA SFT
       ~$50 / 1 天        → 理解数据格式、训练循环、评测

第2步  从零训"小可乐"     nanotron / TorchTitan 训 0.1–0.5B，
       ~$100–1000         喂 FineWeb-Edu 10–50B tokens，
       1 节点几天          完整走 pretrain→SFT→DPO（对照 [[08-olmo]]）

第3步  扩到 1B + 领域数据  验证数据/配方，$几千–几万（可开源/发论文）

第4步  有团队+融资再谈 8B+  这时才考虑路线 C
```

---

## 6. 最低门槛起步包

| 项 | 选择 |
|---|---|
| 算力 | RunPod / Lambda 租 **1× H100**（$2–3/时）或 8×H100 节点按需 |
| 框架 | `nanotron` 或 `TorchTitan`（学习友好） |
| 数据 | `HuggingFaceFW/fineweb-edu`（直接可下） |
| 评测 | `lm-evaluation-harness` |
| 参考代码 | 已 fork 的 `forks/OLMo`（最完整的真实训练代码） |

---

## 7. Checklist

- [ ] **定路线**（A 微调 / B 小预训练 / C 前沿）→ 决定数量级
- [ ] **数据**：来源 + 清洗去重 + 配比 + tokenizer + **许可合规**
- [ ] **算力**：租 vs 买、GPU 型号、互联；用 `6ND` 估 GPU·时 → 估钱
- [ ] **框架**：预训练 + 对齐(TRL) + 评测(harness)
- [ ] **配方**：Dense/MoE、注意力、超参（抄开源 config）
- [ ] **稳定性**：QK-Norm / Z-loss / 检查点容错
- [ ] **后训练**：SFT → DPO / GRPO
- [ ] **评测**：MMLU / GSM8K / HumanEval… + 领域集
- [ ] **缓冲**：预算 ×1.5–3（失败重跑 / 调参 / 人力）

---

## 8. 参考与延伸

- 训练循环实读：[[08-olmo]]（`forks/OLMo/olmo/train.py`）
- 架构选型：[[01-llama]] [[02-qwen]] [[03-deepseek]] [[04-mistral]] [[05-gemma]] [[06-phi]] [[07-glm]] [[09-minimax]]
- 经验法则：Chinchilla（[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)）、Scaling Laws（[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)）
- 开源数据：[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)、[Dolma](https://huggingface.co/datasets/allenai/dolma)、[The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
- 框架：[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[TorchTitan](https://github.com/pytorch/torchtitan)、[nanotron](https://github.com/huggingface/nanotron)、[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[TRL](https://github.com/huggingface/trl)

> ← 返回 [docs 索引](README.md) · [仓库首页](../README.md)
