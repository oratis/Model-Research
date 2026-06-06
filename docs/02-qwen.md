# Qwen 深度分析（GQA / QK-Norm / 混合思考模式 / 36T 预训练 / 四阶段后训练 / strong-to-weak 蒸馏）

> 本文档为 Model-Research 学习仓库的 **Qwen（阿里通义千问）** 专题。架构与训练细节主要来自本地 fork
> `forks/Qwen3/Qwen3_Technical_Report.pdf`（arXiv 2505.09388 的 PDF，本文以「TR p.X」引用其页码）、`forks/Qwen3/README.md`、
> `forks/Qwen3/eval/`、`forks/Qwen3/docs/source/`。
> **重要：本地 fork 内不含 modeling 源码**（Qwen3 仓库是文档 + 评测 + 部署示例库，不是权重/建模库）。因此第 6 章「源码走读」
> 改用 HuggingFace `transformers` 官方实现，所有 `文件:行号` 引用均针对 `transformers` 的 `main` 分支当前内容核对（截至 2026-06-06）。
> 规格 / benchmark 数字尽量给官方来源 URL，无法核实者标注「未核实 / 约」。撰写日期：2026-06-06。
>
> 交叉链接：[[03-deepseek]]（MoE / MLA / GRPO 对照）、[[01-llama]]（dense GQA 基线）、[[07-glm]]（同期中国开源 + 混合思考对照）。

---

## 1. 概览

**出品方**：阿里巴巴通义千问团队（Qwen Team, Alibaba Cloud）。定位「全尺寸 + 全模态的开源基座」——从 0.6B 边端模型到 235B 旗舰 MoE 全覆盖，绝大多数权重以 **Apache-2.0** 释出（可商用、可蒸馏），是当前下游微调 / 蒸馏生态被引用最广的开源底座之一（DeepSeek-R1、众多社区模型都以 Qwen 为蒸馏 student/teacher）。

**一句话定位**：Qwen3 把「**思考模式（thinking，用于复杂多步推理）**」与「**非思考模式（non-thinking，用于快速通用对话）**」融进**同一份权重**，并提供 **thinking budget（思考预算）** 让用户在推理时按 token 数权衡延迟与质量；同时用 **strong-to-weak 蒸馏**把旗舰模型的能力低成本地灌进小模型——以「一份模型替代 chat 模型 + 专用推理模型两套」的产品形态（TR Abstract, p.1；TR p.2）。

**版本演进表**：

| 版本 | 时间 | 关键变化 | 来源 |
|---|---|---|---|
| Qwen (Qwen1) | 2023-08起 | 首代，1.8B/7B/14B/72B dense；中英为主 | [QwenLM/Qwen](https://github.com/QwenLM/Qwen) |
| **Qwen1.5** | 2024-02-05 | 0.5B/1.8B/4B/7B/14B/32B/72B/110B 全系；统一 HF 接口 | [blog qwen1.5](https://qwenlm.github.io/blog/qwen1.5/) |
| Qwen1.5-MoE-A2.7B | 2024-03-28 | Qwen 家族**首个 MoE**（A2.7B 激活） | [blog qwen-moe](https://qwenlm.github.io/blog/qwen-moe/) |
| **Qwen2** | 2024-06-06 | 0.5B/1.5B/7B/57B-A14B/72B；**全系 GQA**；29 语；最长 128K | [arXiv 2407.10671](https://arxiv.org/abs/2407.10671) |
| **Qwen2.5** | 2024-09-19 | 0.5B–72B（新增 3B/14B/32B）；**预训练 18T tokens**；128K（YaRN）；专用 Coder/Math 线 | [arXiv 2412.15115](https://arxiv.org/abs/2412.15115) |
| **Qwen3 (2504)** | 2025-04-29 | 6 dense（0.6–32B）+ 2 MoE（30B-A3B、235B-A22B）；**混合思考 + thinking budget**；**36T tokens**；119 语；**QK-Norm，去 QKV-bias**；Apache-2.0 | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) |
| Qwen3-235B-A22B-Instruct/Thinking-2507 | 2025-07 | 旗舰升级：思考/非思考拆成两条独立线；长上下文 256K（可扩到 1M） | [HF 2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Qwen3-30B-A3B-2507 / Qwen3-4B-2507 | 2025-07~08 | 中小尺寸同步出 Instruct/Thinking-2507 | [README.md:67-72](../forks/Qwen3/README.md) |

> 说明：本仓库 fork 的 `forks/Qwen3` 是 **Qwen3 的文档 / 评测 / 部署仓库**（README + docs + eval + examples），随附 `Qwen3_Technical_Report.pdf`。它**不含 modeling 权重代码**——这一点贯穿全文，第 6 章用 `transformers` 源码补足架构走读。2507 系列为联网核实到的后续小版本（见第 3 章脚注）。

**为什么值得研究（学习要点预告）**：

1. **混合思考一体化**：不再「chat 模型 + reasoning 模型」两套权重，而是一份权重 + chat-template flag（`/think`、`/no_think`）切换，并能「思考到一半被截断仍给答案」——thinking budget（TR §4.3, p.11）。
2. **strong-to-weak 蒸馏 > 直接 RL**：小模型用旗舰 teacher 的 logits 蒸馏，仅 **1/10 GPU 小时**就超过直接做 RL 的效果，且 Pass@64 也更高（TR §4.5/§4.7, p.12/p.20-21）。
3. **架构上的「减法」**：相对 Qwen2 **去掉 QKV-bias**、相对 Qwen2.5-MoE **去掉 shared expert**，并引入 **QK-Norm** 稳定大模型训练（TR §2, p.3）。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| 官方 GitHub（Qwen3，文档/评测/部署） | https://github.com/QwenLM/Qwen3 |
| 官方 GitHub（Qwen2.5） | https://github.com/QwenLM/Qwen2.5 |
| 本仓库 fork（Qwen3） | `forks/Qwen3`（`Qwen3_Technical_Report.pdf`、`README.md`、`eval/`、`docs/source/`） |
| HuggingFace 组织 | https://huggingface.co/Qwen |
| HF: Qwen3-235B-A22B（旗舰 MoE） | https://huggingface.co/Qwen/Qwen3-235B-A22B |
| HF: Qwen3-30B-A3B（小 MoE） | https://huggingface.co/Qwen/Qwen3-30B-A3B |
| HF: Qwen3 collection | https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f |
| 官方博客（Qwen3-2504） | https://qwenlm.github.io/blog/qwen3/ |
| 官方文档（readthedocs） | https://qwen.readthedocs.io/ |
| 论文：Qwen3 Technical Report | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) |
| 论文：Qwen2.5 Technical Report | [arXiv 2412.15115](https://arxiv.org/abs/2412.15115) |
| 论文：Qwen2 Technical Report | [arXiv 2407.10671](https://arxiv.org/abs/2407.10671) |
| transformers `modeling_qwen3.py`（dense 架构源码） | https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py |
| transformers `modeling_qwen3_moe.py`（MoE 架构源码） | https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py |

---

## 3. 模型规格

Qwen3-2504 系列共 **6 dense + 2 MoE**。下表 dense 部分来自 **TR Table 1（p.3）**，MoE 部分来自 **TR Table 2（p.3）**，并与 HF model card 交叉核对（标 ✅）。

### 3.1 Dense 模型（TR Table 1, p.3）

| 模型 | Layers | Heads (Q / KV) | Tie Embedding | 上下文 | 来源 |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 16 / 8 | Yes | 32K | TR Table 1 |
| Qwen3-1.7B | 28 | 16 / 8 | Yes | 32K | TR Table 1 |
| Qwen3-4B | 36 | 32 / 8 | Yes | 128K | TR Table 1 |
| Qwen3-8B | 36 | 32 / 8 | No | 128K | TR Table 1 |
| Qwen3-14B | 40 | 40 / 8 | No | 128K | TR Table 1 |
| Qwen3-32B | 64 | 64 / 8 | No | 128K | TR Table 1 |

> 注：小模型（0.6B/1.7B/4B）**tie embedding**（输入输出词嵌入共享），节省参数；8B 起不共享。KV head 全系固定为 8（dense），是 GQA 的体现。表里 128K 是「YaRN 外推后」的推理上限；S1/S2 预训练序列长度仅 4,096，长上下文阶段扩到 32,768（TR §3.2, p.4），见第 5/7 章。

### 3.2 MoE 模型（TR Table 2, p.3 + HF model card ✅）

| 维度 | Qwen3-30B-A3B | Qwen3-235B-A22B | 来源 |
|---|---|---|---|
| 总参数 | 30.5B（非嵌入 29.9B） | 235B（非嵌入 234B） | [HF 30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) ✅ / [HF 235B](https://huggingface.co/Qwen/Qwen3-235B-A22B) ✅ |
| 激活参数 / token | 3.3B | 22B | HF model card ✅ |
| Layers | 48 | 94 | TR Table 2 / HF ✅ |
| Heads (Q / KV) | 32 / 4 | 64 / 4 | TR Table 2 / HF ✅ |
| 专家数（总 / 激活） | 128 / 8 | 128 / 8 | TR Table 2 / HF ✅ |
| 上下文（native / YaRN） | 32,768 / 131,072 | 32,768 / 131,072 | HF model card ✅ |
| 许可证 | Apache-2.0 | Apache-2.0 | HF model card ✅ |

### 3.3 全系通用规格

| 维度 | 取值 | 来源 |
|---|---|---|
| Tokenizer | Qwen BBPE（byte-level BPE） | TR §2, p.3 |
| 词表大小 | **151,669** | TR §2, p.3 |
| 位置编码 | RoPE，long-context 阶段 base 频率 10,000 → **1,000,000**（ABF） | TR §3.2, p.4 |
| 长上下文外推 | **YaRN + Dual Chunk Attention (DCA)**，推理期 4 倍序列扩展 | TR §3.2, p.4 |
| 归一化 | RMSNorm（pre-norm） + **QK-Norm** | TR §2, p.3 |
| 激活 / FFN | SwiGLU | TR §2, p.3 |
| 注意力 | GQA（去 QKV-bias） | TR §2, p.3 |
| 多语言 | **119 种语言和方言**（Qwen2.5 为 29） | TR Abstract, p.1 |
| 许可证 | Apache-2.0（开源权重全系） | [README.md:397](../forks/Qwen3/README.md) |

> **2507 后续版本（联网核实，截至 2026-06）**：旗舰把混合思考拆成两条独立线 `Qwen3-235B-A22B-Instruct-2507`（仅非思考）与 `Qwen3-235B-A22B-Thinking-2507`（仅思考），long-context 提到 **256K，并可扩到 1M tokens**（[README.md:30-42, 67](../forks/Qwen3/README.md)；[HF 2507 card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)）。注意：2507 拆线是产品取舍，**与本技术报告（2504）描述的「一体混合思考」是两个阶段**。

---

## 4. Benchmark

> 口径提醒：Qwen3 同一模型有 **thinking / non-thinking 两套分数**，下文均注明。报告评测设置（TR §4.6, p.13）：thinking 用 temp=0.6 / top-p=0.95 / top-k=20；non-thinking 用 temp=0.7 / top-p=0.8 / top-k=20；AIME 每题采样 64 次取平均，max 输出 32K（AIME 放宽到 38,912）。

### 4.1 旗舰 Qwen3-235B-A22B（Thinking 模式）vs 推理模型（TR Table 11, p.14）

| Benchmark（版本） | OpenAI-o1 | DeepSeek-R1 | Gemini2.5-Pro | **Qwen3-235B-A22B** |
|---|---|---|---|---|
| MMLU-Redux | 92.8 | 92.9 | 93.7 | 92.7 |
| GPQA-Diamond | 78.0 | 71.5 | 84.0 | 71.1 |
| AIME'24 | 74.3 | 79.8 | 92.0 | **85.7** |
| AIME'25 | 79.2 | 70.0 | 86.7 | **81.5** |
| LiveCodeBench v5 | 63.9 | 64.3 | 70.4 | **70.7** |
| CodeForces (Rating) | 1891 | 2029 | 2001 | **2056** |
| BFCL v3 | 67.8 | 56.9 | 62.9 | **70.8** |
| Arena-Hard | 92.1 | 92.3 | 96.4 | 95.6 |

要点：thinking 模式下 Qwen3-235B-A22B **以 35% 总参 / 60% 激活参在 17/23 项上超过 DeepSeek-R1**，尤其数学 / agent / 代码（TR p.15）；与闭源 o1 / Gemini2.5-Pro 仍有差距（GPQA、AIME 上 Gemini2.5-Pro 领先）。

### 4.2 旗舰 Qwen3-235B-A22B（Non-thinking 模式）vs 通用模型（TR Table 12, p.14）

| Benchmark | GPT-4o-1120 | DeepSeek-V3 | Qwen2.5-72B-Inst | LLaMA-4-Maverick | **Qwen3-235B-A22B** |
|---|---|---|---|---|---|
| MMLU-Redux | 87.0 | 89.1 | 86.8 | 91.8 | 89.2 |
| GPQA-Diamond | 46.0 | 59.1 | 49.0 | 69.8 | 62.9 |
| AIME'24 | 11.1 | 39.2 | 18.9 | 38.5 | **40.1** |
| LiveCodeBench v5 | 32.7 | 33.1 | 30.7 | 37.2 | 35.3 |
| BFCL v3 | 72.5 | 57.6 | 63.4 | 52.9 | 68.0 |
| Arena-Hard | 85.3 | 85.5 | 81.2 | 82.7 | **96.1** |

要点：即便**不开思考**，Qwen3-235B-A22B 在 18/23 项上超 GPT-4o-1120，并整体超过 DeepSeek-V3 / Qwen2.5-72B-Instruct（TR p.15）。

### 4.3 旗舰 dense Qwen3-32B（Thinking）vs 同档（TR Table 13, p.16）

| Benchmark | DeepSeek-R1-Distill-Llama-70B | QwQ-32B | OpenAI-o3-mini(medium) | **Qwen3-32B** |
|---|---|---|---|---|
| AIME'24 | 70.0 | 79.5 | 79.6 | **81.4** |
| AIME'25 | 56.3 | 69.5 | 74.8 | 72.9 |
| LiveCodeBench v5 | 54.5 | 62.7 | 66.3 | 65.7 |
| BFCL v3 | 49.3 | 66.4 | 64.6 | **70.3** |
| Arena-Hard | 60.6 | 89.5 | 89.0 | **93.8** |

要点：Qwen3-32B（thinking）在 17/23 项上超过 Qwen 自家上一代推理模型 **QwQ-32B**，成为「32B 甜点尺寸」的新 SOTA 推理模型，并与闭源 o3-mini(medium) 互有胜负（TR p.15）。

### 4.4 小 MoE 的性价比（TR Table 15, p.17）

| Benchmark | QwQ-32B (32B激活) | **Qwen3-30B-A3B (3B激活)** |
|---|---|---|
| AIME'24 | 79.5 | **80.4** |
| AIME'25 | 69.5 | **70.9** |
| LiveCodeBench v5 | 62.7 | 62.6 |
| BFCL v3 | 66.4 | 69.1 |

要点：**Qwen3-30B-A3B 仅用约 1/10 激活参就追平 QwQ-32B**（TR p.19-20）——这是 strong-to-weak 蒸馏 + MoE 共同的成果，也是官方主打的「小 MoE 性价比」卖点。

### 4.5 基座（Base）对比：Qwen3-235B-A22B-Base vs 同行（TR Table 3, p.6 摘选）

| Benchmark | Qwen2.5-72B-Base | Llama-4-Maverick-Base (402B) | DeepSeek-V3-Base (671B) | **Qwen3-235B-A22B-Base** |
|---|---|---|---|---|
| MMLU-Pro | 58.07 | 63.91 | 59.84 | **68.18** |
| GPQA | 45.88 | 43.94 | 41.92 | **47.47** |
| MATH | 62.12 | 63.32 | 62.62 | **71.84** |
| EvalPlus | 65.93 | 68.38 | 63.75 | **77.60** |
| MMMLU | 84.40 | 83.09 | 85.88 | **86.70** |

要点：基座层面，Qwen3-235B-A22B-Base **以约 1/3 总参、2/3 激活参在 14/15 项上超过 DeepSeek-V3-Base**（TR p.5）。

### 4.6 长上下文（RULER，TR Table 23, p.23 摘选，non-thinking + YaRN×4）

| 模型 | Avg | 32K | 64K | 128K |
|---|---|---|---|---|
| Qwen2.5-72B-Instruct | 95.1 | 96.5 | 93.0 | 88.4 |
| Qwen3-235B-A22B | 95.0 | 95.1 | 93.3 | **90.6** |
| Qwen3-32B | 93.7 | 94.4 | 91.8 | 85.6 |
| Qwen3-30B-A3B | 91.6 | 92.4 | 89.1 | 79.2 |

要点：non-thinking 下 Qwen3 同档普遍超 Qwen2.5；**thinking 模式在纯检索类长任务上反而略降**（思考内容对检索无益甚至干扰，故 TR 在长上下文测评时把 thinking budget 限到 8192，TR p.23）。

> 更多 benchmark（含 0.6B–14B、逐语言多语言表）见 TR Table 4–37（p.6–35）。fork 内 `forks/Qwen3/eval/` 仅提供了 **ARC-AGI-1（Qwen3-235B-A22B-Instruct-2507，pass@1 = 40.75）** 一项可复现脚本（[eval/README.md](../forks/Qwen3/eval/README.md)），其余分数以技术报告为准。

---

## 5. 架构剖析

Qwen3 dense 与 MoE **共享同一套基础架构**（GQA + RoPE + RMSNorm + SwiGLU），区别只在 FFN 是 dense 还是 MoE（TR §2, p.3）。

### 5.1 与前代的「加减法」（TR §2, p.3）

| 组件 | Qwen2 | Qwen2.5 | **Qwen3** | 说明 |
|---|---|---|---|---|
| 注意力 | GQA + **QKV-bias** | GQA + QKV-bias | GQA，**去 QKV-bias** | 去掉 bias 简化 + 稳定 |
| QK 归一化 | 无 | 无 | **QK-Norm（RMSNorm）** | 对 Q、K 各做 head-dim 上的 RMSNorm，稳定大模型训练 |
| MoE shared expert | （Qwen2-MoE 有）| Qwen2.5-MoE 有 shared expert | **去掉 shared expert** | 改用细粒度专家 + 全局负载均衡 |
| MoE 负载均衡 | aux loss | aux loss | **global-batch load balancing loss** | 鼓励专家专业化 |
| 多语言 | 29 语 | 29 语 | **119 语** | 数据侧大幅扩展 |

> **QK-Norm 是 Qwen3 架构最核心的「新增项」**：在把 Q/K 投影后、施加 RoPE 前，对每个 head 的 head-dim 向量做一次 RMSNorm。它约束了 Q·K 内积的数值范围，缓解大模型 attention logits 爆炸 / 训练不稳（来源思想：Dehghani et al. 2023，TR §2 引用）。这与 [[08-olmo]] 的 QK-Norm（OLMo2 对整层做 norm）实现略有不同——Qwen3 **只在 head_dim 上做**（见第 6 章源码注释）。

### 5.2 MoE 路由（TR §2, p.3）

- 128 个专家，每 token 激活 8 个（top-8）；
- **无 shared expert**（区别于 Qwen2.5-MoE、也区别于 [[03-deepseek]] 的 DeepSeekMoE「shared+routed」结构）；
- 细粒度专家切分（fine-grained expert segmentation，源自 DeepSeekMoE 的 Dai et al. 2024）；
- **global-batch load balancing loss**（Qiu et al. 2025）鼓励专家专业化。

### 5.3 长上下文链路（TR §3.2, p.4）

预训练第三阶段把序列长度从 4,096 扩到 32,768，并：(1) **ABF** 把 RoPE base 频率 10,000 → 1,000,000；(2) 推理期叠加 **YaRN**（频率插值外推）+ **Dual Chunk Attention (DCA)**，实现约 4× 序列扩展，最终对外标称 128K（2507 系列 256K~1M）。

---

## 6. 源码走读（基于 transformers，非本地 fork）

> **必读说明**：本仓库 `forks/Qwen3` **不含 modeling 源码**——它是 Qwen3 官方的「文档 + 评测 + 部署示例」仓库（`README.md`、`docs/source/`、`eval/`、`examples/`、`docker/`），用于演示如何**使用**模型，而非定义模型权重的前向计算。因此架构走读使用 HuggingFace `transformers` 的官方实现。以下行号均针对 `transformers` **main 分支**当前文件内容核对（2026-06-06）；行号会随上游重构漂移，引用时请以 commit 固定为准。
>
> - dense：`transformers/models/qwen3/modeling_qwen3.py`（[GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)）
> - MoE：`transformers/models/qwen3_moe/modeling_qwen3_moe.py`（[GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)）

### 6.1 RMSNorm（`modeling_qwen3.py:50`）

```python
# modeling_qwen3.py:50-68
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        ...
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

标准 RMSNorm（等价 T5LayerNorm，注释见 `:53`），pre-norm 用法。下文 QK-Norm 复用同一个类，但作用在 `head_dim` 上。

### 6.2 Attention：QK-Norm + GQA + RoPE（`modeling_qwen3.py:222`）

`Qwen3Attention.__init__`（`:222-250`）的关键三点：

```python
# modeling_qwen3.py:230-250
self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads      # GQA 分组数
self.scaling = self.head_dim**-0.5
self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads  * self.head_dim, bias=config.attention_bias)  # :236
self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)  # :239
self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)  # :242
self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # :248  unlike olmo, only on the head dim!
self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # :249  thus post q_norm does not need reshape
self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None  # :250
```

走读要点：

1. **去 QKV-bias**：`bias=config.attention_bias`（`:236-243`）。Qwen3 的 config 把 `attention_bias` 默认置为 `False`，这就是 TR §2 所说「去掉 Qwen2 的 QKV-bias」在代码里的落地——通过 config 开关，而非删除参数。
2. **QK-Norm**：`q_norm` / `k_norm` 都是作用在 **`head_dim`** 维上的 `Qwen3RMSNorm`（`:248-249`）。源码注释 `unlike olmo, only on the head dim!` 明确点出与 [[08-olmo]] 的差异。
3. **GQA**：`num_key_value_groups = num_attention_heads // num_key_value_heads`（`:231`）。如 Qwen3-32B 是 64/8=8 组、235B-A22B 是 64/4=16 组——KV head 远少于 Q head，省 KV cache。

`forward`（`:255-290`）里 QK-Norm 与 RoPE 的施加顺序：

```python
# modeling_qwen3.py:261-268
hidden_shape = (*input_shape, -1, self.head_dim)
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # :263  先 q_proj→reshape→QK-Norm
key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # :264
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)               # :265
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)        # :268  再施加 RoPE
```

**顺序关键**：`q_proj → view 成 (…, head, head_dim) → QK-Norm → RoPE`。即 **QK-Norm 在 RoPE 之前**——先把每个 head 的向量归一化，再旋转位置。GQA 的 KV 复制在注意力 kernel 内由 `repeat_kv` 完成：

```python
# modeling_qwen3.py:184-193  +  206-207
def repeat_kv(hidden_states, n_rep):  # (batch, kv_heads, seqlen, head_dim) -> (batch, kv_heads*n_rep, seqlen, head_dim)
    ...
key_states   = repeat_kv(key,   module.num_key_value_groups)   # :206
value_states = repeat_kv(value, module.num_key_value_groups)   # :207
```

`sliding_window`（`:250`、`:285`）是上游通用字段；Qwen3 全注意力（`config.sliding_window=None`），注释 `# diff with Llama`（`:285`）。

### 6.3 SwiGLU FFN（`modeling_qwen3.py:70`）

```python
# modeling_qwen3.py:70-82
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # :76
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]                                            # :79  silu
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))            # :82  SwiGLU
```

经典 SwiGLU：`down(silu(gate(x)) * up(x))`，三个无 bias 线性层（`:76-78`）。

### 6.4 MoE 路由（`modeling_qwen3_moe.py`）

> 上游 `main` 已把 MoE 重构为 **`Qwen3MoeTopKRouter`（路由器）+ `Qwen3MoeExperts`（批量专家）+ `Qwen3MoeSparseMoeBlock`（外壳）** 三件套，与早期版本里「单个 `Qwen3MoeSparseMoeBlock` 内含 gate + 专家循环」略有不同。当前结构如下。

**路由器**（`modeling_qwen3_moe.py:254`）：

```python
# modeling_qwen3_moe.py:254-272
class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok                          # :257  每 token 激活专家数（=8）
        self.num_experts = config.num_experts                           # :258  专家总数（=128）
        self.norm_topk_prob = config.norm_topk_prob                     # :259
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))  # :261  gate（无 bias）
    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight)                          # :265  (seq, num_experts)
        router_probs  = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)  # :266
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)        # :267  top-8
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)            # :268-269  top-k 概率重归一化
        ...
        return router_logits, router_scores, router_indices                          # :272
```

走读要点：

1. **gate 无 bias**：路由就是一个 `weight` 参数的线性投影（`:261`、`:265`）。
2. **先 softmax 再 topk**：对全 128 专家 softmax（`:266`），取 top-8（`:267`）。
3. **norm_topk_prob**：top-8 概率除以其和重新归一化（`:268-269`），让 8 个激活专家权重和为 1——这是「先 softmax 后 topk」路由的标准做法（与 [[03-deepseek]] V3 的 sigmoid 门控 + bias 负载均衡形成对照）。

**批量专家执行**（`modeling_qwen3_moe.py:215`）——用 one-hot + `index_add_` 把 token 散到各自专家再聚回：

```python
# modeling_qwen3_moe.py:215-249（摘）
class Qwen3MoeExperts(nn.Module):
    ...
    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)  # :235
    ...
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])                            # :243
    ...
    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(...))            # :249  聚合各专家输出
```

**外壳**（`modeling_qwen3_moe.py:275`）把两者串起来：

```python
# modeling_qwen3_moe.py:275-285
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        self.gate = Qwen3MoeTopKRouter(config)                                         # :279
        self.experts = Qwen3MoeExperts(config)
    def forward(self, hidden_states):
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)       # :284
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)  # :285
```

**辅助负载均衡 loss**（`modeling_qwen3_moe.py:527`）：`load_balancing_loss_func` 即 Switch-Transformer 式 aux loss，`tokens_per_expert × router_prob_per_expert` 求和后 `× num_experts`（`:563-601`）。这是 HF 的通用实现；**TR §2 实际采用的是 global-batch load balancing loss（Qiu et al. 2025）**——训练侧细节以技术报告为准，HF 推理实现里 aux loss 仅在 `output_router_logits=True` 时计算（`:691-695`）。

> **专家 MLP 也是 SwiGLU**：每个专家等价一个小 `Qwen3MoeMLP`（`:198`），结构与 dense 的 SwiGLU 相同，只是 `intermediate_size` 为 MoE 专属（更小）。

### 6.5 与本地 PDF 的对照小结

| 论点 | TR PDF（本地 fork） | transformers 源码 |
|---|---|---|
| 去 QKV-bias | §2, p.3 文字 | `:236-243` `bias=config.attention_bias`（config 默认 False） |
| QK-Norm 只在 head_dim | §2, p.3 文字 | `:248-249` + 注释 `only on the head dim!` |
| QK-Norm 在 RoPE 前 | （报告未显式说顺序） | `:263-264`（QK-Norm）→ `:268`（RoPE）✅ 源码确认 |
| 128 专家 / top-8 / 无 shared expert | §2, p.3 + Table 2 | `:257-258` top_k/num_experts；外壳无 shared 分支 ✅ |
| global-batch 负载均衡 | §2, p.3 文字 | HF 用 Switch aux loss（`:527`），训练口径以 TR 为准 ⚠️ |

---

## 7. 训练机制

### 7.1 预训练（TR §3, p.3-4）

- **规模**：约 **36 万亿（36T）tokens**，覆盖 **119 种语言和方言**（相比 Qwen2.5 的 18T tokens / 29 语，token 翻倍、语言 3 倍；TR §3.1）。
- **数据扩充手段**（TR p.2/§3.1）：
  - 用 **Qwen2.5-VL** 对海量 PDF 文档做文字识别（OCR），再用 **Qwen2.5** 精修，得到「数万亿」高质量文本 token；
  - 用 **Qwen2.5 / Qwen2.5-Math / Qwen2.5-Coder** 合成「数万亿」token（教科书、问答、指令、代码）；
  - 自研多语言标注系统，对 **30T+ token** 按教育价值 / 领域 / 安全等维度打标，支持 **instance-level 数据配比优化**（用小代理模型做大量消融）。
- **三阶段预训练**（TR §3.2, p.4）：
  1. **General Stage (S1)**：>30T tokens，序列长 4,096，打语言与世界知识底座；
  2. **Reasoning Stage (S2)**：约 +5T 高质量 token，提高 STEM / 代码 / 推理 / 合成数据占比，并加速学习率衰减；
  3. **Long Context Stage**：数千亿 token，序列长扩到 32,768（75% 落在 16K–32K），RoPE base 10,000→1,000,000（ABF），叠加 YaRN + DCA。
- **scaling law**：和 Qwen2.5 一样，针对上述三阶段建 scaling law 预测每个 dense/MoE 模型的最优 lr 与 batch size（TR §3.2）。

### 7.2 后训练总览：四阶段 + 蒸馏（TR §4, p.9 Figure 1）

旗舰模型走 **四阶段** 流水线（前两阶段练「思考」，后两阶段融入「非思考」）；轻量模型走 **strong-to-weak 蒸馏**（不必每个小模型都跑四阶段）。

```
旗舰（235B-A22B / 32B）：
  Stage 1  Long-CoT Cold Start（长链思维冷启动 SFT）
     ↓
  Stage 2  Reasoning RL（推理强化学习，GRPO）
     ↓
  Stage 3  Thinking Mode Fusion（思考/非思考融合 SFT）
     ↓
  Stage 4  General RL（通用强化学习，20+ 任务奖励）

轻量（0.6B/1.7B/4B/8B/14B + 30B-A3B）：
  Strong-to-Weak Distillation（off-policy + on-policy 蒸馏，省 ~90% GPU 小时）
```

### 7.3 四阶段细节

**Stage 1 — Long-CoT Cold Start（TR §4.1, p.10）**
- 构建数学 / 代码 / 逻辑 / STEM 数据集，每题配可验证答案或测试用例；
- 两轮过滤：query 过滤（用 Qwen2.5-72B-Instruct 剔除不可验证 / 无需 CoT 就能答对的题）+ response 过滤（用 **QwQ-32B** 生成 N 个候选，按答案正确性、重复、猜测、思考与总结不一致、语言混杂等 6 条剔除）；
- 目标：**只灌入基础推理范式，不追求即时刷分**，样本与步数都尽量少，给后续 RL 留探索空间。

**Stage 2 — Reasoning RL（TR §4.2, p.10）**
- 仅用 **3,995 个 query-verifier 对**（要求：未在冷启动用过、对冷启动模型可学、尽量难、覆盖广）；
- 算法 **GRPO**（Shao et al. 2024，与 [[03-deepseek]] 同源）；大 batch + 高 rollout 数 + off-policy 提升样本效率；
- 通过控制熵稳步上升 / 维稳来平衡探索-利用；
- 实测 **Qwen3-235B-A22B 的 AIME'24 从 70.1 → 85.1，仅 170 步 RL，无人工调参**。

**Stage 3 — Thinking Mode Fusion（TR §4.3, p.11）**
- 在 Stage 2 模型上做 **continual SFT**，把「非思考」能力融进「思考」模型，并设计 chat template 统一两模式；
- **thinking 数据**：用 Stage 2 模型自己对 Stage 1 query 做 rejection sampling 生成（避免退化）；**non-thinking 数据**：覆盖代码 / 数学 / 指令 / 多语言 / 创作 / 角色扮演，用自动 checklist 评质量；
- **chat template（TR Table 9, p.11）**：用户/系统消息里加 `/think`、`/no_think` flag；non-thinking 样本保留**空 `<think></think>` 块**保证格式一致；多轮对话里随机插多个 flag，模型遵循**最后一个**；
- **Thinking Budget**（思考预算）作为「涌现能力」：因为模型学会了「基于不完整思考也能给答案」，当思考长度达到用户设定阈值时，手动插入 `Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n` 截断并出答案——**此能力非显式训练，而是 Fusion 的副产物**。

**Stage 4 — General RL（TR §4.4, p.12）**
- 覆盖 **20+ 任务** 的奖励系统，针对：指令遵循、格式遵循（正确响应 `/think`、`/no_think` 与 `<think>` token）、偏好对齐、Agent 能力（多轮真实环境工具调用反馈）、RAG 等专门场景；
- 三类奖励：(1) **Rule-based**（规则可验证，防 reward hacking）；(2) **Model-based with reference**（给参考答案，用 Qwen2.5-72B-Instruct 打分）；(3) **Model-based without reference**（训练 reward model 给标量分，提升 engagement/helpfulness）。
- 效果（TR Table 22, p.21）：Stage 4 把 `ThinkFollow`（模式切换正确率）从 88.7 → **98.9**、`ToolUse` 大幅提升；代价是 AIME'24/LiveCodeBench 等高难任务的 thinking 分**略降**——官方明确「接受这个 trade-off 换通用性」（TR p.22）。

### 7.4 Strong-to-Weak 蒸馏（TR §4.5/§4.7, p.12/p.20-21）

适用于 5 个 dense 小模型（0.6B/1.7B/4B/8B/14B）+ MoE 的 30B-A3B。两阶段：
1. **Off-policy 蒸馏**：把 teacher 在 `/think` 与 `/no_think` 两种模式下的输出混合，给 student 做响应蒸馏，打基础推理 + 模式切换；
2. **On-policy 蒸馏**：student 自己产 on-policy 序列，再对齐 teacher（Qwen3-32B 或 235B-A22B）的 **logits**，最小化 KL。

**关键结论（TR Table 21, p.21）**——同从一个 off-policy 蒸馏的 8B checkpoint 出发：

| 方法 | AIME'24 (pass@64) | LiveCodeBench v5 | GPU 小时 |
|---|---|---|---|
| Off-policy Distillation（起点） | 55.0 (90.0) | 42.0 | — |
| + Reinforcement Learning | 67.6 (90.0) | 52.9 | **17,920** |
| + On-policy Distillation | **74.4 (93.3)** | **60.3** | **1,800** |

即 **on-policy 蒸馏比直接 RL 分更高、且只要约 1/10 GPU 小时**，并能提升 pass@64（探索空间，RL 不能；TR p.20-21）。这是 Qwen3 报告里最有「方法论冲击力」的发现之一。

---

## 8. 学习要点与横向对比

### 8.1 设计取舍小结

1. **「一份权重双模式」是产品级创新**：thinking / non-thinking 融进同一模型 + chat-template flag + thinking budget，省去「chat 模型 + reasoning 模型」两套部署。代价是高难推理任务在通用 RL 后略降（§7.3 Stage 4）。
2. **蒸馏 > RL（对小模型）**：旗舰 teacher 的 logits 蒸馏，1/10 成本超过直接 RL——这条结论直接影响整个开源社区的小模型训练范式。
3. **架构做「减法」求稳**：去 QKV-bias、去 shared expert、加 QK-Norm。与 [[03-deepseek]] 走「MLA + 多 shared/routed 专家 + auxiliary-loss-free」的「加法」路线形成鲜明对照。
4. **全尺寸覆盖 + Apache-2.0**：0.6B 到 235B 全开放、宽松许可，是 Qwen 成为「最被广泛 fork/蒸馏的开源底座」的根本原因。

### 8.2 横向对比

| 维度 | **Qwen3** | [[03-deepseek]] V3/R1 | [[01-llama]] 3.x | [[07-glm]] 4.5 |
|---|---|---|---|---|
| 注意力 | GQA + **QK-Norm**，去 QKV-bias | **MLA**（低秩 KV 压缩） | GQA | GQA（部分版本含 partial-RoPE，见 GLM 文） |
| MoE | 128 专家 top-8，**无 shared** | DeepSeekMoE：shared + routed 细粒度 | 主线 dense（Llama-4 才 MoE） | MoE（见 GLM 文） |
| 负载均衡 | global-batch aux loss | **auxiliary-loss-free**（bias 调节） | — | — |
| 推理模型 | **混合思考一体**（budget 可调） | R1 独立推理模型（GRPO 纯 RL/cold-start） | 无官方 reasoning 线 | 混合思考（GLM 也走一体路线，可对照） |
| 小模型策略 | **strong-to-weak 蒸馏** | R1 蒸馏到 Qwen/Llama | — | — |
| 预训练 tokens | **36T**（Qwen3）/ 18T（Qwen2.5） | 14.8T（V3） | 15T+（Llama-3） | — |
| 词表 | **151,669**（BBPE，超大） | ~129K | 128K | — |
| 多语言 | **119 语** | 偏中英 | 多语言 | 中英为主 |
| 许可证 | **Apache-2.0** | V3 模型协议 / R1 MIT | Llama 社区许可（有限制） | 见 GLM 文 |

**给学习者的三条 takeaway**：
- 想理解「**混合思考 + thinking budget** 如何在不改架构的前提下实现」，读 TR §4.3（chat template + 空 think 块 + 截断指令）即可，工程上极轻。
- 想理解「**为什么小模型要蒸馏不要 RL**」，TR Table 21 是最干净的对照实验（同起点、控 GPU 小时、看 pass@64）。
- 想对照「**MoE 路由两条流派**」：Qwen3 的「softmax→topk→重归一化 + aux loss」（§6.4）vs DeepSeek 的「sigmoid 门控 + auxiliary-loss-free bias」（见 [[03-deepseek]]）。

---

## 9. 参考

**本地 fork（`forks/Qwen3`）**
- `Qwen3_Technical_Report.pdf`（= arXiv 2505.09388；本文「TR p.X」指此 PDF 页码）— 架构 Table 1-2(p.3)、预训练 §3(p.3-4)、后训练 §4(p.9-22)、benchmark Table 3-37(p.6-35)、RULER Table 23(p.23)。
- `README.md` — 版本时间线、2507 系列、部署框架、许可证（[README.md:397](../forks/Qwen3/README.md)）、引用 bibtex。
- `eval/README.md` — ARC-AGI-1 可复现脚本（Qwen3-235B-A22B-Instruct-2507，pass@1=40.75）。
- `docs/source/getting_started/thinking_budget.md` — thinking budget 推理代码示例。

**官方来源（联网核实）**
- Qwen3 Technical Report — [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- Qwen2.5 Technical Report（18T tokens / 128K YaRN） — [arXiv 2412.15115](https://arxiv.org/abs/2412.15115)
- Qwen2 Technical Report（全系 GQA / 29 语） — [arXiv 2407.10671](https://arxiv.org/abs/2407.10671)
- Qwen3 blog — https://qwenlm.github.io/blog/qwen3/
- HF: Qwen3-235B-A22B（94 层 / 64Q-4KV / 128-8 专家 / 32K→131K YaRN / Apache-2.0） — https://huggingface.co/Qwen/Qwen3-235B-A22B
- HF: Qwen3-30B-A3B（30.5B-3.3B / 48 层 / 32Q-4KV / 128-8 专家） — https://huggingface.co/Qwen/Qwen3-30B-A3B
- HF 组织 — https://huggingface.co/Qwen
- GitHub QwenLM/Qwen3 — https://github.com/QwenLM/Qwen3
- 历代 blog：[qwen1.5](https://qwenlm.github.io/blog/qwen1.5/)、[qwen-moe](https://qwenlm.github.io/blog/qwen-moe/)、[qwen2](https://qwenlm.github.io/blog/qwen2/)、[qwen2.5](https://qwenlm.github.io/blog/qwen2.5/)

**架构源码（HuggingFace transformers，main 分支，2026-06-06 核对）**
- dense：[`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) — RMSNorm `:50`、Attention/QK-Norm `:222-290`、SwiGLU `:70-82`、repeat_kv `:184-193`。
- MoE：[`modeling_qwen3_moe.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) — TopKRouter `:254-272`、Experts `:215-249`、SparseMoeBlock `:275-285`、load_balancing_loss_func `:527-601`。

> 免责声明：transformers 行号随上游重构会漂移，正式引用请以具体 commit 固定。技术报告中的训练细节（如 global-batch load balancing、GRPO 超参）以 PDF 为准；HF 实现是推理对齐版，二者在训练侧可能不完全一致（已在 §6.4/§6.5 标注）。
