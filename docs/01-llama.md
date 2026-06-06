# Llama 深度分析（经典解码器 / GQA / RoPE / Llama 4 MoE + iRoPE + 原生多模态）

> 本文档为 Model-Research 学习仓库的 **Llama (Meta)** 专题。源码走读基于本地 fork submodule
> `forks/llama-models`（commit `0e0b8c5`，2025-10-09），包含 Llama 3 与 Llama 4 的官方推理实现及各代 MODEL_CARD。
> 所有 `文件:行号` 引用均已对照当前文件内容核对。规格 / benchmark 数字尽量给官方来源 URL（论文、HF model card、Meta 博客、仓库内 MODEL_CARD），无法核实者标注「未核实 / 约」。
> 撰写日期：2026-06-06。

---

## 1. 概览

**出品方**：Meta（Meta AI / GenAI 团队，原 FAIR）。Llama 系列是当代「开放权重（open-weight）」大模型的事实标准之一：架构干净、文档齐全、生态庞大（云厂商、推理框架、微调工具链几乎全部首先适配 Llama），被下载数亿次。

**定位与意义**：
- Llama 1/2/3 是**经典 dense 解码器**（pre-norm RMSNorm + RoPE + GQA + SwiGLU）的「教科书级」参考实现——后来的 Qwen、Mistral、Gemma、DeepSeek dense 部分基本都沿用这套骨架。读懂 `forks/llama-models/models/llama3/model.py` 一个文件，就理解了 2023–2024 主流 LLM 的解码器。
- Llama 4（2025-04）是 Meta 的架构大转向：首次上 **MoE（Mixture-of-Experts）**、首次**原生多模态 early fusion**、首创 **iRoPE**（interleaved RoPE + NoPE 层）做超长上下文（Scout 宣称 10M token）。

**许可证（重要，非纯开源）**：各代均使用 **Llama Community License Agreement**（自定义商用许可，非 OSI 认可的开源协议）。关键限制：(1) 月活 > 7 亿的产品需单独向 Meta 申请商用授权；(2) 必须遵守 Acceptable Use Policy；(3) 衍生模型 / 产品命名需带 "Llama" 前缀并注明 "Built with Llama"。允许商用、微调、蒸馏（用 Llama 输出训练其它模型）。来源：`forks/llama-models/models/llama4/LICENSE`、`README.md:25-33`。

**家族演进时间线**（版本表）：

| 版本 | 时间 | 尺寸 | 上下文 | 关键变化 | 许可证 | 来源 |
|---|---|---|---|---|---|---|
| **LLaMA 1** | 2023-02 | 7B / 13B / 33B / 65B | 2K | 首发；RMSNorm+RoPE+SwiGLU；1.0–1.4T tokens；仅研究许可 | 非商用研究 | [arXiv 2302.13971](https://arxiv.org/abs/2302.13971) |
| **Llama 2** | 2023-07 | 7B / 13B / 70B | 4K | 2T tokens；**RLHF**（拒绝采样+PPO）；**GQA**（仅 70B）；GAtt；首个商用许可 | Llama 2 CL | [arXiv 2307.09288](https://arxiv.org/abs/2307.09288) / `models/llama2/MODEL_CARD.md` |
| **Llama 3** | 2024-04 | 8B / 70B | 8K | 15T tokens；**全系 GQA**；新 128K **tiktoken** 词表；DPO 后训练 | Llama 3 CL | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) / `README.md:28` |
| **Llama 3.1** | 2024-07 | 8B / 70B / 405B | **128K** | RoPE scaling 扩到 128K；405B 前沿级；蒸馏/合成数据来源 | Llama 3.1 CL | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) / `README.md:29` |
| **Llama 3.2** | 2024-09 | 1B / 3B（端侧）；11B / 90B（视觉） | 128K | **端侧小模型**（1B/3B）+ **视觉模型**（cross-attention 接 image encoder） | Llama 3.2 CL | `README.md:30-31` |
| **Llama 3.3** | 2024-12 | 70B（仅 instruct） | 128K | 用更好后训练，让 70B 逼近 405B 质量 | Llama 3.3 CL | `models/llama3_3/MODEL_CARD.md` |
| **Llama 4 Scout** | 2025-04 | 17B 激活 / **16 专家** / 109B 总 | **10M** | MoE + iRoPE + 原生多模态 early fusion | Llama 4 CL | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| **Llama 4 Maverick** | 2025-04 | 17B 激活 / **128 专家** / 400B 总 | 1M | MoE（dense/MoE 层交替）+ iRoPE + 多模态 | Llama 4 CL | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| Llama 4 Behemoth | 训练中（截至发布未放出） | 288B 激活 / 16 专家 / ~2T 总 | — | 教师模型，对 Scout/Maverick 做 **codistillation** | — | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |

> 命名小贴士：早期论文写作 **LLaMA**（全大写中间小写），Llama 2 起官方统一为 **Llama**。本文除引用 1 代论文外统一写 Llama。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| 官方 GitHub（llama-models，推理源码 + model card） | https://github.com/meta-llama/llama-models |
| 本仓库 fork（commit `0e0b8c5`） | `forks/llama-models`（`models/llama3/`、`models/llama4/`、`models/sku_list.py`） |
| HuggingFace 组织 | https://huggingface.co/meta-llama |
| HF: Llama-4-Scout-17B-16E | https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E |
| HF: Llama-3.3-70B-Instruct | https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct |
| HF: Llama-3.1-405B | https://huggingface.co/meta-llama/Llama-3.1-405B |
| Meta 官方博客（Llama 4） | https://ai.meta.com/blog/llama-4-multimodal-intelligence/ |
| HF blog: Welcome Llama 4 | https://huggingface.co/blog/llama4-release |
| 论文：LLaMA: Open and Efficient Foundation Language Models（1 代） | [arXiv 2302.13971](https://arxiv.org/abs/2302.13971) |
| 论文：Llama 2: Open Foundation and Fine-Tuned Chat Models | [arXiv 2307.09288](https://arxiv.org/abs/2307.09288) |
| 论文：The Llama 3 Herd of Models（覆盖 3 / 3.1） | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) |
| Llama Stack（部署工具链） | https://github.com/meta-llama/llama-stack |
| Llama Cookbook（食谱/微调） | https://github.com/meta-llama/llama-cookbook |

---

## 3. 模型规格

### 3.1 经典 dense 系列（Llama 2 / 3 / 3.1 / 3.2 / 3.3）

下表数字以本地 `forks/llama-models/models/sku_list.py` 的 `arch_args`（即生产权重实际超参）为准，与各 MODEL_CARD 交叉验证。注意：`n_kv_heads` 即 GQA 的 KV 头数。

| 模型 | dim | n_layers | n_heads | n_kv_heads (GQA) | vocab | rope_theta | scaled_rope | 上下文 | 来源行 |
|---|---|---|---|---|---|---|---|---|---|
| Llama 2 7B | 4096 | 32 | 32 | 8※ | 32000 | 500000 | False | 4K | sku_list.py:135-146 |
| Llama 2 70B | 8192 | 80 | 64 | 8 | 32000 | 500000 | False | 4K | sku_list.py:171-183 |
| Llama 3 8B | 4096 | 32 | 32 | 8 | 128256 | 500000 | False | 8K | sku_list.py:194-205 |
| Llama 3 70B | 8192 | 80 | 64 | 8 | 128256 | 500000 | False | 8K | sku_list.py:212-224 |
| Llama 3.1 8B | 4096 | 32 | 32 | 8 | 128256 | 500000 | **True** | 128K | sku_list.py:235-246 |
| Llama 3.1 70B | 8192 | 80 | 64 | 8 | 128256 | 500000 | **True** | 128K | sku_list.py:253-264 |
| Llama 3.1 405B | 16384 | 126 | 128 | 8 | 128256 | 500000 | **True** | 128K | sku_list.py:272-284 |
| Llama 3.2 1B | 2048 | 16 | 32 | 8 | 128256 | 500000 | True | 128K | sku_list.py:333-344 |
| Llama 3.2 3B | 3072 | 28 | 24 | 8 | 128256 | 500000 | True | 128K | sku_list.py:348-363 |
| Llama 3.3 70B | 8192 | 80 | 64 | 8 | 128256 | 500000 | True | 128K | 同 3.1 70B 架构 |

> ※ 注 1：sku_list.py 中 Llama 2 7B 的 `arch_args` 写 `n_kv_heads=8`，但 Llama 2 论文与原始权重里 **7B/13B 是 MHA（无 GQA），仅 70B 用 GQA**（`models/llama2/MODEL_CARD.md` 的表格只在 70B 行打 ✓）。此处 sku_list 的 8 是该 fork 统一推理配置的取值，**以 MODEL_CARD 为准**。
> ※ 注 2：3.2-3B 的 `dim=3072 / n_layers=28 / n_heads=24 / ffn_dim_multiplier=1.0` 已逐字段核对（sku_list.py:348-363）。
> ※ 405B 另有 mp16 变体把 `n_kv_heads` 设为 16（sku_list.py:314），便于 16 路张量并行，属部署变体而非模型本体差异。

GQA 节省的 KV cache：以 70B 为例，`n_heads=64` 但 `n_kv_heads=8`，KV cache 体积只有 MHA 的 **1/8**（`n_rep=8`，见 §5）。这是 70B 级模型能做长上下文推理的关键。

### 3.2 Llama 4（MoE + 多模态）

Llama 4 的架构超参**不在 sku_list.py 里**——该文件中 Llama 4 的 `arch_args={}` 为空（sku_list.py:89/96/108/115），真实超参随 checkpoint 的 `params.json` 加载（见 generation.py:67-74「`with open(... / "params.json")`」）。故下表数字来自官方 model card + Meta 博客 + 代码默认值。

| 维度 | Llama 4 Scout | Llama 4 Maverick | 来源 |
|---|---|---|---|
| 激活参数 / token | 17B | 17B | MODEL_CARD.md:26-37 |
| 总参数 | 109B | 400B | MODEL_CARD.md:27/38 |
| 专家数 `num_experts` | **16** | **128** | MODEL_CARD.md:5 / [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| 每 token 激活专家 `top_k` | 1（+1 共享专家常激活） | 1（+1 共享） | args.py:35 默认 top_k=1；moe.py:183 topk |
| MoE 层分布 | 每层均 MoE | **dense / MoE 层交替** | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| 上下文长度 | **10M** | 1M | MODEL_CARD.md:31/42 |
| NoPE 层（iRoPE） | 每 4 层一个 NoPE | 同左 | [HF blog](https://huggingface.co/blog/llama4-release) / model.py:258 |
| QK-Norm | 是（非 NoPE 层） | 是 | model.py:261、216-218 |
| 输入 / 输出模态 | 文本+图像 / 文本+代码 | 同左 | MODEL_CARD.md:29/40 |
| 训练 token 数 | ~40T（多模态） | ~22T（多模态） | MODEL_CARD.md:32/43 |
| 知识截止 | 2024-08 | 2024-08 | MODEL_CARD.md:33/44 |
| 训练精度 | FP8 | FP8 | [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |

> 「17B 激活 / 109B 总」对 Scout 的理解：16 个专家但每 token 路由到 1 个（top-1）+ 1 个共享专家 + 注意力等共享参数，合计激活约 17B；16 专家全部参数 + 共享层合计约 109B。Maverick 用 128 专家、dense/MoE 交替，总参 400B，激活仍 17B。
> Behemoth（教师）：288B 激活 / 16 专家 / 近 2T 总参，对 Scout/Maverick 做 **codistillation**（共蒸馏），截至 2025-04 发布时仍在训练，未放出权重。来源：[Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)。

---

## 4. Benchmark

> 说明：下列数字尽量取**仓库内 MODEL_CARD**（最权威、与放出权重一一对应）。不同来源 / shot 数 / 评测脚本会有出入，已注明版本与 shots。

### 4.1 预训练基座对比（Llama 1 vs 2，5-shot MMLU 等）

来自 `models/llama2/MODEL_CARD.md:64-72`（grouped academic benchmarks）：

| 模型 | MMLU | Code(HumanEval+MBPP 均值) | Math(GSM8K+MATH 均值) | BBH | AGIEval |
|---|---|---|---|---|---|
| Llama 1 65B | 63.4 | 30.7 | 30.8 | 43.5 | 47.6 |
| Llama 2 7B | 45.3 | 16.8 | 14.6 | 32.6 | 29.3 |
| Llama 2 13B | 54.8 | 24.5 | 28.7 | 39.4 | 39.1 |
| Llama 2 70B | **68.9** | **37.5** | **35.2** | **51.2** | **54.2** |

### 4.2 Instruct 对比：Llama 3.1 / 3.3（仓库内 model card）

来自 `models/llama3_3/MODEL_CARD.md:63-73`（该表同时含 3.1 的 8B/70B/405B Instruct，metric/shots 见行内）：

| Benchmark | Shots | Metric | 3.1 8B-Inst | 3.1 70B-Inst | **3.3 70B-Inst** | 3.1 405B-Inst |
|---|---|---|---|---|---|---|
| MMLU (CoT) | 0 | macro_avg/acc | 73.0 | 86.0 | **86.0** | 88.6 |
| MMLU-Pro (CoT) | 5 | macro_avg/acc | 48.3 | 66.4 | **68.9** | 73.3 |
| IFEval | — | — | 80.4 | 87.5 | **92.1** | 88.6 |
| GPQA Diamond (CoT) | 0 | acc | 31.8 | 48.0 | **50.5** | 49.0 |
| HumanEval | 0 | pass@1 | 72.6 | 80.5 | **88.4** | 89.0 |
| MBPP EvalPlus (base) | 0 | pass@1 | 72.8 | 86.0 | **87.6** | 88.6 |
| MATH (CoT) | 0 | sympy_score | 51.9 | 68.0 | **77.0** | 73.8 |
| BFCL v2（工具调用） | 0 | macro_avg/valid | 65.4 | 77.5 | **77.3** | 81.1 |
| MGSM（多语数学） | 0 | em | 68.9 | 86.9 | **91.1** | 91.6 |

> 看点：**Llama 3.3 70B 在 IFEval/HumanEval/MATH 上反超 3.1 405B**（92.1 vs 88.6、88.4 vs 89.0 接近、77.0 vs 73.8），这是「同架构、更强后训练」把 70B 推到 405B 档位的典型案例——也是 3.3 只放 70B 一个尺寸的原因。

### 4.3 Llama 4 Instruct（仓库内 model card）

来自 `models/llama4/MODEL_CARD.md:190-306`（与 3.3 70B / 3.1 405B 对比，均 0-shot）：

| Benchmark | Metric | Llama 3.3 70B | Llama 3.1 405B | **Scout** | **Maverick** |
|---|---|---|---|---|---|
| MMLU-Pro | macro_avg/acc | 68.9 | 73.4 | 74.3 | **80.5** |
| GPQA Diamond | accuracy | 50.5 | 49.0 | 57.2 | **69.8** |
| LiveCodeBench（10/24–02/25） | pass@1 | 33.3 | 27.7 | 32.8 | **43.4** |
| MGSM | em | 91.1 | 91.6 | 90.6 | **92.3** |
| MMMU（图像推理） | accuracy | 无多模态 | 无多模态 | 69.4 | **73.4** |
| ChartQA（图像理解） | relaxed_acc | 无多模态 | 无多模态 | 88.8 | **90.0** |
| DocVQA (test) | anls | 无多模态 | 无多模态 | 94.4 | 94.4 |
| MathVista | accuracy | — | — | 70.7 | **73.7** |

> Llama 4 预训练基座对比见 MODEL_CARD.md:103-182（如 Maverick MMLU 5-shot 85.5、MATH 4-shot 61.2、MBPP pass@1 77.6）。整体定位：Maverick 对标 GPT-4o / Gemini 2.0 Flash，Scout 是「单卡可跑 + 超长上下文」的轻量多模态 MoE。

---

## 5. 架构剖析（Llama 3 经典解码器）

本节逐模块走读 `forks/llama-models/models/llama3/model.py`，这是理解整个 Llama 系列的核心。整体结构：`Transformer` → N×`TransformerBlock` → 每块 = `RMSNorm → Attention(GQA+RoPE) → 残差` + `RMSNorm → FeedForward(SwiGLU) → 残差`，最后 `RMSNorm → output` 线性投影到词表。

### 5.1 pre-norm RMSNorm

`model.py:31-42`。RMSNorm 只做均方根归一化（无均值中心化、无 bias），比 LayerNorm 更省算力：

```python
class RMSNorm(torch.nn.Module):
    def _norm(self, x):
        # 按最后一维求均方根并归一；rsqrt = 1/sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)  # 升 fp32 算稳定，再降回原 dtype
        return output * self.weight                # 仅一个可学习缩放向量
```

「pre-norm」体现在 `TransformerBlock.forward`（model.py:252-253）：归一化在子层**之前**，残差走的是未归一化的 `x`：

```python
h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)  # 先 norm 再 attn，残差加原始 x
out = h + self.feed_forward(self.ffn_norm(h))                                # FFN 同理
```

### 5.2 RoPE（旋转位置编码）与长上下文 scaling

RoPE 把位置信息编码为**复数旋转**，作用在 Q/K 上。预计算频率 `model.py:65-72`：

```python
def precompute_freqs_cis(dim, end, theta=10000.0, use_scaled=False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 每对维度一个频率
    t = torch.arange(end)                  # 位置 0..end
    if use_scaled:
        freqs = apply_scaling(freqs)       # 长上下文：对低频做 NTK-by-parts 缩放
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # e^{iθ}，complex64
```

注意 Llama 实际 `rope_theta=500000`（args.py:48），远大于原始 10000，本身就利于长程。应用旋转 `model.py:83-93`：把 Q/K 每两维看成复数，乘以 `freqs_cis` 即旋转：

```python
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 实数对 -> 复数
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)                 # 复数乘 = 旋转
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**RoPE scaling（128K 长上下文的核心）** `model.py:45-62`。Llama 3.1 起 `use_scaled_rope=True`，用「NTK-by-parts」分频段缩放：高频（短波长，管局部）不动，低频（长波长，管长程）除以 `scale_factor=8`，中间平滑过渡：

```python
def apply_scaling(freqs):
    scale_factor = 8; low_freq_factor = 1; high_freq_factor = 4
    old_context_len = 8192                 # 原 Llama 3 长度
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)  # 长波长缩放
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where((wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
                       (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs, new_freqs)
```

这套「网格搜索得到的」(`model.py:46` 注释) 参数，把原生 8K 上下文外推到 128K 而几乎不掉点。

### 5.3 GQA（Grouped-Query Attention）

GQA 是 Llama 2-70B / Llama 3 全系的标配，介于 MHA（每头独立 KV）与 MQA（所有头共享 1 组 KV）之间。`Attention.__init__`（model.py:108-116）：

```python
self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
self.n_local_heads = args.n_heads // world_size          # Q 头数（如 64）
self.n_local_kv_heads = self.n_kv_heads // world_size    # KV 头数（如 8）
self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每组 KV 被 Q 重复几次 = 8
self.head_dim = args.dim // args.n_heads
```

KV 头通过 `repeat_kv` 复制到与 Q 头同数（model.py:96-105），从而 KV cache 只存 1/8：

```python
def repeat_kv(x, n_rep):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)   # 在新维度上 expand（零拷贝）
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim))
```

### 5.4 SwiGLU 前馈网络

`FeedForward`（model.py:205-225）。SwiGLU = `SiLU(W1·x) ⊙ (W3·x)` 再过 `W2`，是 GLU 的门控变体，比 ReLU/GELU MLP 效果好：

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        hidden_dim = int(2 * hidden_dim / 3)                 # SwiGLU 三矩阵，按 2/3 折算保持参数量
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 取整到 256/1024 倍数
        self.w1 = ColumnParallelLinear(dim, hidden_dim, ...)  # gate
        self.w2 = RowParallelLinear(hidden_dim, dim, ...)     # down
        self.w3 = ColumnParallelLinear(dim, hidden_dim, ...)  # up
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))       # 门控 ⊙ 上投影 -> 下投影
```

### 5.5 KV cache 与因果 mask

Llama 3 用「静态预分配」KV cache（`Attention.__init__` model.py:147-162，按 `max_seq_len` 开 `cache_k/cache_v`）。增量解码时只写入新位置（model.py:183-187）：

```python
self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk    # 写入当前 step 的 K
self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
keys   = self.cache_k[:bsz, : start_pos + seqlen]          # 取全部历史 K
values = self.cache_v[:bsz, : start_pos + seqlen]
```

因果 mask 在 `Transformer.forward`（model.py:287-303）构造为上三角 `-inf`，并对 prefill 阶段拼接历史位置的 0 列（增量解码 `seqlen==1` 时无需 mask）。

### 5.6 tied / untied embedding

Llama 3 用 **untied**（输入 `tok_embeddings` 与输出 `output` 是两套权重，model.py:264/271）。对比：Llama 3.2 的 1B/3B 端侧模型为省参改回 **tied**（共享 embedding），Gemma 等小模型亦如此。本 fork 的 llama3 通用实现是 untied。

### 5.7 Llama 4 架构差异（MoE + iRoPE + 多模态）

走读 `forks/llama-models/models/llama4/model.py` 与 `moe.py`。Llama 4 在 §5.1–5.6 骨架上叠加四项改造：

**(a) iRoPE = 交错 NoPE 层 + 注意力温度调节。** `TransformerBlock.__init__`（model.py:258-261）按层号判定是否为 NoPE 层（不加 RoPE 的层）：

```python
self.is_nope_layer = args.nope_layer_interval is not None and (layer_id + 1) % args.nope_layer_interval == 0
use_rope = not self.is_nope_layer                       # NoPE 层关闭 RoPE
use_qk_norm = args.use_qk_norm and not self.is_nope_layer
```

NoPE 层「没有位置编码」反而能更好泛化到训练时未见过的超长序列；带 RoPE 的层负责局部精度。官方称这种交错布局（HF 博客：**每 4 层一个 NoPE 层**）为 **iRoPE**（interleaved RoPE）。NoPE 层在推理时还做温度调节以增强长度泛化（model.py:223-229）：

```python
if self.attn_temperature_tuning and not self.use_rope:   # 仅 NoPE 层
    seq_positions = torch.arange(start_pos, start_pos + seqlen, ...)
    attn_scales = torch.log(torch.floor((seq_positions + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
    xq = xq * attn_scales.view(1, seqlen, 1, 1)          # 随位置增大放大 Q，对短上下文几乎无影响
```

配套还有「分块局部注意力」`create_chunked_attention_mask`（model.py:431-438）：非 NoPE 层只在固定大小 chunk 内做局部注意力，NoPE 层用全局 mask（model.py:327-330）——这是 10M 上下文得以高效计算的关键。

**(b) QK-Norm。** 非 NoPE 层对 Q/K 各做一次 RMSNorm（model.py:216-218），稳定长上下文注意力。

**(c) MoE 前馈。** `TransformerBlock` 按 `interleave_moe_layer_step` 决定该层用 MoE 还是 dense FFN（model.py:265-283）。MoE 核心 `moe.py:175-210`：

```python
router_scores = torch.matmul(x_aD, self.router_DE).transpose(0, 1)         # 每 token 对每专家打分
router_scores_aK, router_indices_aK = torch.topk(..., self.moe_args.top_k, dim=1)  # 选 top_k 专家
router_scores = torch.sigmoid(router_scores)                                # 门控用 sigmoid（非 softmax）
...
out_aD = self.shared_expert(x_aD)                                           # 共享专家：所有 token 都过
routed_out_eg_D = self.experts(routed_in_EG_D.detach())                     # 路由专家：批量 SwiGLU
out_aD.scatter_add_(dim=0, index=router_indices_EG_D, src=routed_out_eg_D.view(-1, D))  # 回写相加
```

注意两点：**共享专家**（`shared_expert`，moe.py:156）对所有 token 恒激活，与 DeepSeekMoE 思路一致；门控用 **sigmoid** 而非 softmax（moe.py:191）。专家本体是「批量 SwiGLU」`Experts.batched_swiglu`（moe.py:97-99），用 `torch.bmm` 把多个专家的 `w1/w2/w3` 一次算完：

```python
def batched_swiglu(self, x, w1, w3, w2):
    middle_out_egF = F.silu(torch.bmm(x, w1)) * torch.bmm(x, w3)   # 与 dense SwiGLU 同式，只是批量化
    return torch.bmm(middle_out_egF, w2)
```

**(d) 原生多模态 early fusion。** Llama 4 在 `Transformer.forward`（model.py:400-402）把图像 embedding 直接**按位置替换**进文本 token 序列，再走同一套解码器（而非 Llama 3.2 视觉模型那种独立 cross-attention 层）：

```python
if image_embedding := model_input.image_embedding:
    h_image = self.vision_projection(image_embedding.embedding)   # 视觉编码投影到文本 dim
    h = h * ~image_embedding.mask + h_image * image_embedding.mask  # 在 <|image|> 位置替换为图像向量
```

视觉侧用 ViT + **PixelShuffle** 压缩 patch 数（`vision/embedding.py:20-47`），这就是「early fusion」——文本与视觉 token 在输入层即融合，共享全部 Transformer 层。

---

## 6. 源码走读（推理链路 / 对话模板 / tokenizer）

### 6.1 Llama 3 forward 主循环

见 §5.5。`Transformer.forward`（model.py:280-309）：embedding → 逐层 → 末 RMSNorm → `output` 投影到 `vocab_size`，logits 升 fp32。

### 6.2 生成与采样（generation.py）

`Llama3.build`（generation.py:39-145）负责加载：读 `params.json` 构造 `ModelArgs`（generation.py:88-95）、resharding 权重（按 `n_kv_heads`，generation.py:98-101）、按设备选 bf16/fp16（generation.py:126-135）。

核心采样 `generate`（generation.py:153-300）是标准自回归循环 + KV cache，温度>0 时走 top-p（nucleus）采样：

```python
if temperature > 0:
    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    next_token = sample_top_p(probs, top_p)        # 见下
else:
    next_token = torch.argmax(logits[:, -1], dim=-1)
```

`sample_top_p`（generation.py:348-370）：按概率降序累加，截断累积质量超过 p 的尾部，重归一后采样：

```python
probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
probs_sum = torch.cumsum(probs_sort, dim=-1)
mask = probs_sum - probs_sort > p                  # 累积超过 p 的位置丢弃
probs_sort[mask] = 0.0
probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
next_token = torch.multinomial(probs_sort, num_samples=1)
```

默认 `temperature=0.6, top_p=0.9`（generation.py:157-158）。停止 token 为 `<|end_of_text|> / <|eom_id|> / <|eot_id|>`（tokenizer.py:112-116）。

### 6.3 tokenizer（tiktoken / BPE）

Llama 3 起从 Llama 2 的 SentencePiece（32000 词表）换成 **tiktoken BPE**，词表 **128256**（128000 base + 256 特殊/保留 token）。`tokenizer.py:46-116`：

```python
class Tokenizer:
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|...)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|..."  # 预切分正则
    special_tokens = ["<|begin_of_text|>", "<|end_of_text|>", ..., "<|start_header_id|>",
                      "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>", "<|image|>"]
```

`encode`（tokenizer.py:118-172）对超长串分片（避免 tiktoken panic），可选加 BOS/EOS。

### 6.4 chat_format（对话模板 / 特殊 token）

`forks/llama-models/models/llama3/chat_format.py` 是 Llama 3 对话协议的权威实现。消息头格式 `<|start_header_id|>{role}<|end_header_id|>\n\n`（chat_format.py:58、61-67）：

```python
def _encode_header(self, role):
    tokens = [self.tokenizer.special_tokens["<|start_header_id|>"]]
    tokens.extend(self.tokenizer.encode("ipython" if role == "tool" else role, bos=False, eos=False))
    tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
    tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
    return tokens
```

整段对话 `encode_dialog_prompt`（chat_format.py:146-163）：开头 `<|begin_of_text|>`，逐条消息编码，最后补一个空的 assistant 头让模型续写：

```python
tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
for message in messages:
    toks, imgs = self.encode_message(message, tool_prompt_format)
    tokens.extend(toks)
tokens.extend(self._encode_header("assistant"))    # 提示模型「该你说了」
```

每条消息以 `<|eot_id|>`（end of turn）或 `<|eom_id|>`（end of message，工具调用未结束时）收尾（chat_format.py:139-143）。`tool` 角色映射为 `ipython`，工具调用走 `<|python_tag|>`（chat_format.py:42-49、124）——这是 Llama 3 的 agent / tool-use 协议基础。`decode_assistant_message_from_content`（chat_format.py:171-234）负责反解析模型输出里的工具调用。

### 6.5 Llama 4 的 chat_format / tokenizer 差异

Llama 4 换了一套特殊 token 命名（`models/llama4/tokenizer.py:49-110`）：消息头改为 `<|header_start|> / <|header_end|>`，结束符 `<|eot|> / <|eom|>`，工具调用 `<|python_start|> / <|python_end|>`，并新增大量**视觉 token**（`<|image_start|> <|image|> <|patch|> <|tile_x_separator|>` 等，line 69-84）与 **reasoning token**（`<|reasoning_thinking_start|>`，line 96-97），`num_reserved_special_tokens=2048`（line 120，远多于 Llama 3 的 256）。这反映了 Llama 4 对多模态 + 推理 + 工具的原生支持。

---

## 7. 训练机制（预训练数据 / 对齐演变）

### 7.1 LLaMA 1（2023-02）

纯预训练 + 公开数据，无对齐。7B/13B 训 1.0T token，33B/65B 训 1.4T token。核心贡献：证明「**用更多 token 训更小模型**」（突破 Chinchilla 最优点继续训）能在推理成本上完胜。架构即奠定 RMSNorm + RoPE + SwiGLU。来源：[arXiv 2302.13971](https://arxiv.org/abs/2302.13971)。

### 7.2 Llama 2（2023-07）—— RLHF 范式

- **预训练**：2T token（截止 2022-09），全系（`models/llama2/MODEL_CARD.md`）。
- **对齐三步**：(1) SFT（高质量指令数据）；(2) 训 **reward model**（用 chat 模型加回归头，二元排序损失带 margin）；(3) **RLHF** = **拒绝采样（rejection sampling）+ PPO**。先用拒绝采样（采 K 个回答取 reward 最高的做 SFT 数据），后期叠加 PPO；reward 分 helpfulness 与 safety 两个模型。来源：[arXiv 2307.09288](https://arxiv.org/abs/2307.09288)。
- **Ghost Attention（GAtt）**：多轮对话里让 system 指令「贯穿全程」的技巧——把指令拼到合成对话的每条 user 消息上训练（context distillation 变体），解决多轮中模型「忘记最初设定」。来源：[arXiv 2307.09288](https://arxiv.org/abs/2307.09288) / [TDS 解读](https://towardsdatascience.com/understanding-ghost-attention-in-llama-2-dba624901586/)。

### 7.3 Llama 3 / 3.1（2024）—— DPO 取代 PPO

- **预训练**：**15T+ token**（远超 Llama 2 的 2T），数据配比、质量过滤、退火（annealing）都做了系统优化；405B 用 scaling law 指导算力分配。
- **后训练**：改为更简单稳定的 **SFT + 拒绝采样（RS）+ DPO（Direct Preference Optimization）**，**不再用 PPO**。每轮对齐迭代 SFT→RS→DPO。大量使用**合成数据**（如用大模型生成代码/推理/多语数据，再过滤）。
- **3.1 长上下文**：在 128K 上做持续预训练 + RoPE scaling（§5.2），405B 对标 GPT-4。来源：[arXiv 2407.21783](https://arxiv.org/abs/2407.21783) / [Meta blog](https://ai.meta.com/blog/meta-llama-3-1/)。
- **3.2**：视觉模型用 image encoder + cross-attention 适配器接到冻结的文本模型（`args.py:54-57` 的 `vision_num_cross_attention_layers`）；1B/3B 端侧模型用剪枝 + 知识蒸馏从大模型得到。
- **3.3**：仅 70B，靠更强后训练 + 25M+ 合成样本（`models/llama3_3/MODEL_CARD.md:51`）把 70B 推到接近 405B（见 §4.2）。

### 7.4 Llama 4（2025-04）—— MoE + 原生多模态 + codistillation

- **预训练**：多模态数据，Scout ~40T token、Maverick ~22T token（官博另称合计 >30T 多模态 token），200 种语言，截止 2024-08。**FP8 训练**（官博称 32K GPU 上达 390 TFLOPs/GPU）。用 **MetaP** 技术稳定地设置逐层学习率等关键超参。
- **原生多模态 early fusion**：文本与视觉 token 在输入层融合（§5.7d），而非后接适配器。
- **codistillation from Behemoth**：用仍在训练的 **Behemoth（288B 激活 / 16 专家 / ~2T 总参）**作教师，对 Scout/Maverick 做共蒸馏，把超大模型能力压进小激活模型。来源：[Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)（标注：Behemoth 细节以官博为准，权重未放出）。
- **对齐**：延续 Llama 3 的 SFT + RS + DPO 思路，并强调降低 benign 拒绝率、改善 tone/system-prompt 可控性（`models/llama4/MODEL_CARD.md:332-343`）。

> 对齐演变一句话总结：**Llama 2（RLHF=拒绝采样+PPO，重、要训 reward model + RL）→ Llama 3/3.1（SFT+拒绝采样+DPO，轻、用偏好对直接优化，去掉 PPO）→ Llama 4（沿用 DPO 路线 + codistillation + MoE）**。这条「从 PPO 转向 DPO」的路径是 2023→2024 业界后训练的主流迁移。

---

## 8. 学习要点与横向对比

### 8.1 必记要点

1. **一个文件读懂经典解码器**：`models/llama3/model.py`（310 行）= RMSNorm + RoPE + GQA + SwiGLU + KV cache。这是 2023–2024 几乎所有 dense LLM 的母板。
2. **GQA 的本质**：`n_kv_heads < n_heads`，KV cache 缩到 `1/n_rep`。70B 的 `64→8`（1/8）是长上下文可行的前提（§5.3）。
3. **RoPE scaling**：高频不动、低频缩放、中间平滑（NTK-by-parts），是 8K→128K 外推的关键（§5.2）。`rope_theta=500000` 本身已偏大。
4. **SwiGLU 的 2/3 折算**：三矩阵门控 FFN，hidden_dim 乘 2/3 保持参数量（model.py:214）。
5. **Llama 4 三大件**：MoE（共享专家 + sigmoid 门控 + top-1）、iRoPE（每 4 层 NoPE + 温度调节 + 分块局部注意力）、early-fusion 多模态（输入层按位置替换图像向量）。
6. **对齐演变**：RLHF(PPO) → DPO，外加 Llama 4 的 codistillation。
7. **许可证陷阱**：Llama Community License 不是开源；月活 7 亿门槛 + 命名要求 + AUP。

### 8.2 横向对比（与本仓库其它专题）

| 维度 | Llama | 对比 |
|---|---|---|
| MoE 路由 | Llama 4：top-1 + 1 共享专家 + sigmoid 门控 | [[03-deepseek]] DeepSeekMoE 用细粒度专家 + top-8 + auxiliary-loss-free 负载均衡；[[02-qwen]] Qwen3-MoE、[[07-glm]] GLM 亦走 MoE |
| 长上下文 | Llama 4 iRoPE（NoPE 交错层 + 分块注意力）→ 10M | [[03-deepseek]] V3.2 用 DSA 稀疏注意力；[[09-minimax]] MiniMax-01 用 lightning/linear attention 做长上下文 |
| 注意力 | GQA（Llama 2-70B 起） | [[03-deepseek]] 用 **MLA**（KV 压缩潜变量，cache 更省）；[[04-mistral]] 用 GQA + 滑窗注意力 |
| 多模态 | Llama 4 **early fusion**（输入层融合） | [[05-gemma]] Gemma 多模态、Llama 3.2 视觉用 cross-attention 后融合，路线不同 |
| 词表/tokenizer | tiktoken 128K（L3 起） | [[05-gemma]] Gemma 256K 词表；[[02-qwen]] Qwen 用 BPE ~150K |
| 端侧小模型 | Llama 3.2 1B/3B（tied embedding + 蒸馏） | [[06-phi]] Phi 系列主打小模型 + 高质量合成数据；[[05-gemma]] Gemma 2B/9B |
| 后训练 | SFT + RS + DPO（L3 起，去 PPO） | [[03-deepseek]] R1 用大规模 RL（GRPO）做推理；[[08-olmo]] OLMo 全开放训练配方 |
| 开放程度 | 开放权重 + 自定义商用许可（非纯开源） | [[08-olmo]] OLMo 全开放（数据/代码/权重 + Apache）；DeepSeek/Qwen 多为宽松/MIT |

> Llama 的历史角色：用**最干净的实现 + 最齐全的文档 + 商用友好许可**，把前沿 LLM 架构「标准化、平民化」。后来者（Qwen/Mistral/Gemma/DeepSeek 的 dense 部分）几乎都站在 Llama 解码器的肩膀上，再各自在注意力（MLA/滑窗）、MoE（细粒度/无辅助损失）、训练（RL/全开放）上做差异化。

---

## 9. 参考

**论文**
- LLaMA: Open and Efficient Foundation Language Models（1 代）：https://arxiv.org/abs/2302.13971
- Llama 2: Open Foundation and Fine-Tuned Chat Models：https://arxiv.org/abs/2307.09288
- The Llama 3 Herd of Models（覆盖 Llama 3 / 3.1）：https://arxiv.org/abs/2407.21783

**官方仓库 / 模型卡 / 博客**
- GitHub: meta-llama/llama-models：https://github.com/meta-llama/llama-models
- HuggingFace 组织: meta-llama：https://huggingface.co/meta-llama
- Meta 博客（Llama 4 多模态）：https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- Meta 博客（Llama 3.1）：https://ai.meta.com/blog/meta-llama-3-1/
- HF blog: Welcome Llama 4：https://huggingface.co/blog/llama4-release
- 仓库内 model card：`forks/llama-models/models/{llama2,llama3_1,llama3_3,llama4}/MODEL_CARD.md`

**本地 fork 源码（commit `0e0b8c5`，逐行引用见正文）**
- Llama 3 解码器：`forks/llama-models/models/llama3/model.py`
- Llama 3 生成/分词/对话：`forks/llama-models/models/llama3/{generation,tokenizer,chat_format}.py`、`args.py`
- Llama 4 解码器 / MoE / FFN / 视觉：`forks/llama-models/models/llama4/{model,moe,ffn,args,tokenizer}.py`、`vision/embedding.py`
- 规格清单：`forks/llama-models/models/sku_list.py`

> 免责声明：本文用于学习。benchmark 数字优先取仓库内 MODEL_CARD（与放出权重对应），跨来源/shot 数差异已注明；Behemoth 等未放出模型的数字以官方博客口径为准，标注「未核实/约」处请以 Meta 官方为准。
