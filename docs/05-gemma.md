# Gemma (Google DeepMind)：把 Gemini 的研究"蒸馏"成开放小模型

> 本文聚焦 **Gemma**（Google DeepMind）系列：**Gemma 1**（2B/7B）→ **Gemma 2**（2B/9B/27B）→ **Gemma 3**（1B/4B/12B/27B，引入多模态）→ **Gemma 3n**（端侧）。Gemma 与闭源旗舰 **Gemini** 同源——共享研究、技术与 tokenizer，定位是"把大模型的能力以开放权重的形式下放到可在单卡/端侧运行的尺寸"。
>
> Gemma 全系的两条技术主线值得反复咀嚼：(1) **知识蒸馏（knowledge distillation）**——用更大的教师模型的 token 级分布来训练小模型，是 Gemma 2/3 区别于多数同尺寸开源模型的标志；(2) **local-global 交替注意力 + 极致归一化**——靠"大部分层用短滑窗、少数层做全局"来压低长上下文的 KV cache，靠"双重 RMSNorm + QK-norm"稳住训练。本文逐点用**本地源码真实 `文件:行号`** 佐证。
>
> 源码 fork 位置：`forks/gemma_pytorch`（git submodule，对应 `github.com/google/gemma_pytorch`，本地 commit `014acb7`，最后提交 2025-03-12，package version `0.1`）。这是 Google 官方的**推理实现**（inference-only），不含训练 loop，因此本文的"训练机制"章节以**论文**为准、以代码中的结构开关为佐证。
>
> 横向阅读：[[01-llama]]（架构基线）、[[02-qwen]]（蒸馏对照）、[[04-mistral]]（滑窗注意力对照）、[[03-deepseek]]、[[06-phi]]（数据/蒸馏路线）。

---

## 1. 概览

**Gemma** 是 Google DeepMind 2024 年起推出的开放权重（open-weights）语言模型家族，与 **Gemini** 共享底层研究与技术栈，但以"可在消费级硬件、甚至手机上运行"为目标。一句话定位：**Gemini 负责推边界，Gemma 负责把能力开放化、小型化、可部署化**。

与 Llama、Qwen 等"通用尺寸"路线相比，Gemma 的几个鲜明取舍：

- **蒸馏优先**：Gemma 2 的 2B/9B、Gemma 3 全系都用**更大教师模型的概率分布**做预训练目标，而非单纯的 next-token one-hot。这让小模型在远超"compute-optimal"的 token 量上仍能持续吸收信息（详见 §7）。
- **超大词表**：继承 Gemini 的 **256k**（Gemma 1/2）/ **262k**（Gemma 3）SentencePiece 词表，多语言友好，但 embedding 矩阵巨大（小模型里 embedding 占比惊人）。
- **省 KV cache 的注意力**：Gemma 2 起用 **local sliding window + global** 交替；Gemma 3 把比例推到 **5:1**（5 个局部层配 1 个全局层）并把滑窗收到 1024，使 128K 长上下文的显存可控。
- **归一化堆料**：每个 sub-layer 同时有 pre-norm 和 post-norm（**双重 RMSNorm**），Gemma 3 又加 **QK-norm** 替换 Gemma 2 的 logit soft-capping。

### 家族时间线与版本

| 时间 | 版本 | 尺寸 | 关键点 | 论文 |
|---|---|---|---|---|
| 2024-02 | **Gemma 1** | 2B / 7B | 基于 Gemini 研究；2B 用 MQA、7B 用 MHA；GeGLU；256k 词表；RoPE | [arXiv:2403.08295](https://arxiv.org/abs/2403.08295) |
| 2024-06 | **Gemma 2** | 2B / 9B / 27B | **知识蒸馏**（2B/9B）；local-global 交替（1:1）；**logit soft-capping**；GQA；双重 RMSNorm | [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) |
| 2025-03 | **Gemma 3** | 1B / 4B / 12B / 27B | **多模态**（SigLIP 视觉编码器，4B+）；**128K 上下文**；local-global **5:1**；**QK-norm** 取代 soft-capping；全系蒸馏 | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| 2025-06 | **Gemma 3n** | E2B / E4B | **端侧**；MatFormer（一模型多子模型）+ Per-Layer Embeddings（PLE）省内存；音频 + MobileNet-v5 视觉编码器 | [开发者博客](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) |

> 本 fork（`forks/gemma_pytorch`）的 `config.py` 同时容纳了三代模型，通过 `Architecture` 枚举（`config.py:39-43`）和一系列布尔/序列开关区分。Gemma 3n 因架构（MatFormer/PLE）特殊，**不在本 PyTorch 推理库内**，本文对其只做概念性介绍并标注来源。

**许可证（License）**：Gemma 1/2/3 采用 **Gemma Terms of Use**——一份 Google 自撰的**宽松自定义许可证**（custom license），**允许商用、再分发、微调、蒸馏**，但附带一份"禁止用途政策（Prohibited Use Policy）"，且要求把这些限制**向下游传递**给最终用户；衍生模型（fine-tune / distill 自 Gemma）同样受此条款约束。因此它**不是 OSI 认证的开源协议**，常被称为"almost open"。来源：[Gemma Terms of Use](https://ai.google.dev/gemma/terms)、[TechCrunch 报道](https://techcrunch.com/2025/03/14/open-ai-model-licenses-often-carry-concerning-restrictions/)。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| GitHub（官方 PyTorch 推理库，本 fork 对应） | https://github.com/google/gemma_pytorch |
| 本地 fork | `forks/gemma_pytorch`（commit `014acb7`） |
| HuggingFace 组织 | https://huggingface.co/google |
| HF 代表模型 | `google/gemma-2-27b`、`google/gemma-2-9b`、`google/gemma-3-27b-it`、`google/gemma-3-4b-it`、`google/gemma-3-1b-it` |
| Gemma 1 论文 | https://arxiv.org/abs/2403.08295 |
| Gemma 2 论文 | https://arxiv.org/abs/2408.00118 |
| Gemma 3 论文 | https://arxiv.org/abs/2503.19786 |
| Gemma 3n 开发者指南 | https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/ |
| Gemma 3 产品页 | https://deepmind.google/models/gemma/gemma-3/ |
| Gemma Terms of Use | https://ai.google.dev/gemma/terms |
| Google AI for Developers | https://ai.google.dev/gemma |

---

## 3. 模型规格

下表"架构数字"全部来自**本地源码** `forks/gemma_pytorch/gemma/config.py` 的各 `get_config_for_*` 工厂函数（已逐一核对），token / 上下文规模来自对应论文与 HF model card（已核实）。

### Gemma 1（`Architecture.GEMMA_1`）

| 规格 | Gemma 1 2B | Gemma 1 7B | 源码佐证 |
|---|---|---|---|
| 层数 `num_hidden_layers` | 18 | 28 | `config.py:111` / `:54` |
| 隐藏维度 `hidden_size` | 2048 | 3072 | `config.py:114` / `:60` |
| 注意力头 `num_attention_heads` | 8 | 16 | `config.py:112` / `:56` |
| KV 头 `num_key_value_heads` | **1（MQA）** | **16（MHA）** | `config.py:113` / `:58` |
| `head_dim` | 256 | 256 | `config.py:64` |
| MLP `intermediate_size` | 16384 | 24576 | `config.py:115` / `:62` |
| 词表 `vocab_size` | 256000 | 256000 | `config.py:50` |
| 上下文 `max_position_embeddings` | 8192 | 8192 | `config.py:52` |
| 预训练 tokens | 2T | 6T | 论文 2403.08295 |

### Gemma 2（`Architecture.GEMMA_2`）

| 规格 | Gemma 2 2B | Gemma 2 9B | Gemma 2 27B | 源码佐证 |
|---|---|---|---|---|
| 层数 | 26 | 42 | 46 | `config.py:124` / `:142` / `:161` |
| 隐藏维度 | 2304 | 3584 | 4608 | `config.py:128` / `:145` / `:164` |
| 注意力头 | 8 | 16 | 32 | `config.py:126` / `:143` / `:163` |
| KV 头（GQA） | 4 | 8 | 16 | `config.py:127` / `:144` / `:165` |
| `head_dim` | 256 | 256 | **128** | `config.py:132` / `:151` / `:170` |
| MLP | 9216 | 14336 | 36864 | `config.py:129` / `:146` / `:166` |
| 滑窗 `sliding_window_size` | 4096 | 4096 | 4096 | `config.py:134` / `:153` / `:172` |
| local/global 比例 | **1:1** | **1:1** | **1:1** | `config.py:133` / `:152` / `:171` |
| `final_logit_softcapping` | 30.0 | 30.0 | 30.0 | `config.py:130` / `:149` / `:168` |
| `attn_logit_softcapping` | 50.0 | 50.0 | 50.0 | `config.py:131` / `:150` / `:169` |
| `query_pre_attn_scalar` | — | — | **144** | `config.py:173` |
| 词表 | 256000 | 256000 | 256000 | `config.py:50` |
| 上下文 | 8192 | 8192 | 8192 | `config.py:52` |
| 预训练 tokens | 2T | 8T | **13T** | 论文 2408.00118 |
| 训练方式 | **蒸馏** | **蒸馏** | from scratch | 论文 2408.00118 |

> `attn_types=[LOCAL_SLIDING, GLOBAL] * N`（如 27B 的 `* 23`，`config.py:171`）正好铺满 46 层，构成严格 1:1 交替。

### Gemma 3（`Architecture.GEMMA_3`）

| 规格 | Gemma 3 1B | Gemma 3 4B | Gemma 3 12B | Gemma 3 27B | 源码佐证 |
|---|---|---|---|---|---|
| 层数 | 26 | 34 | 48 | 62 | `config.py:181` / `:214` / `:247` / `:281` |
| 隐藏维度 | 1152 | 2560 | 3840 | 5376 | `config.py:184` / `:217` / `:250` / `:284` |
| 注意力头 | 4 | 8 | 16 | 32 | `config.py:182` / `:215` / `:248` / `:282` |
| KV 头（GQA） | 1 | 4 | 8 | 16 | `config.py:183` / `:216` / `:249` / `:283` |
| `head_dim` | 256 | 256 | 256 | **128** | `config.py:187` / `:220` / `:253` / `:287` |
| MLP | 6912 | 10240 | 15360 | 21504 | `config.py:185` / `:218` / `:251`(`3840*8//2`) / `:285`(`5376*8//2`) |
| 滑窗 | **512** | 1024 | 1024 | 1024 | `config.py:197` / `:230` / `:263` / `:298` |
| local/global 比例 | **5:1** | 5:1 | 5:1 | 5:1 | `config.py:189-196` 等 |
| **QK-norm** `use_qk_norm` | ✅ | ✅ | ✅ | ✅ | `config.py:206` / `:237` / `:271` / `:306` |
| soft-capping | ❌（已移除） | ❌ | ❌ | ❌ | 各 config 无 `*_softcapping` 字段 |
| RoPE θ（local/global） | 10k / **1M** | 10k / 1M | 10k / 1M | 10k / 1M | `config.py:198-201` 等 |
| `rope_scaling_factor`（global） | — | 8 | 8 | 8 | `config.py:239` / `:273` / `:308` |
| 视觉编码器（SigLIP） | ❌（纯文本） | ✅ | ✅ | ✅ | `config.py:206`(None) / `:238`/`:272`/`:307` |
| 词表 | 262144 | 262144 | 262144 | 262144 | `config.py:202` 等 |
| 上下文 | **32768** | 131072 | 131072 | 131072 | `config.py:203` / `:269` / `:304`（4B 默认 8192，但 model card/论文为 128K）|
| 预训练 tokens | 2T | 4T | 12T | **14T** | 论文 2503.19786 |

> **数据点说明**：Gemma 3 技术报告（2503.19786）明确写 "**14T** tokens for Gemma 3 27B, 12T for the 12B, 4T for the 4B, and 2T tokens for the 1B"。这与早期 Gemma 2 27B 的 13T 不同，请勿混淆。
>
> 27B（`get_config_for_27b_v3`）的 `query_pre_attn_scalar = 5376 // 32 = 168`（`config.py:289`），即按 `hidden/heads` 而非 `head_dim` 缩放 query。

### Gemma 3n（端侧，**不在本 fork**，概念性）

| 项 | E2B | E4B | 来源 |
|---|---|---|---|
| 原始参数 | 5B | 8B | [开发者博客](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/) |
| 等效内存占用 | ~2B（最低 2GB） | ~4B（最低 3GB） | 同上 |
| 关键技术 | **MatFormer**（嵌套子模型）、**Per-Layer Embeddings（PLE）**、LAuReL、AltUp、音频 + MobileNet-v5 视觉 | 同上 | 同上 |

---

## 4. Benchmark

以下为各代官方技术报告/HF model card 报告的分数（已核实，注明来源与版本口径）。

### Gemma 1（PT，预训练基座）

| Benchmark | Gemma 1 2B | Gemma 1 7B |
|---|---|---|
| MMLU (5-shot) | 42.3 | **64.3** |
| GSM8K | 17.7 | **46.4** |
| HumanEval | 22.0 | 32.3 |
| HellaSwag | 71.4 | 81.2 |

来源：Gemma 1 论文 [2403.08295](https://arxiv.org/abs/2403.08295)（HumanEval 7B=32.3 取自 CodeGemma 报告 2406.11409 的对照表）。

### Gemma 2 vs Gemma 3（预训练基座，论文同口径对照）

| Benchmark | G2 2B | G2 9B | G2 27B | G3 4B | G3 12B | G3 27B |
|---|---|---|---|---|---|---|
| MMLU | 52.2 | 71.2 | 75.2 | 59.6 | 74.5 | **78.6** |
| GSM8K | 25.0 | 70.2 | 74.6 | 38.4 | 71.0 | **82.6** |
| MATH | 16.4 | 36.4 | 42.1 | 24.2 | 43.3 | **50.0** |

来源：Gemma 3 技术报告 [2503.19786](https://arxiv.org/abs/2503.19786)（其中含 Gemma 2 的同口径对照表）；另含 MMLU-Pro、AGIEval、GPQA、MBPP、HumanEval 等。

官方定位（已核实，来自 2503.19786 摘要）：
- "新的后训练配方显著提升数学、对话、指令遵从与多语言能力，使 **Gemma3-4B-IT 与 Gemma2-27B-IT 相当**，**Gemma3-27B-IT 在多项 benchmark 上可比 Gemini-1.5-Pro**。"
- Gemma 2 论文（2408.00118）：在"practical size"上，9B/27B 的表现可与大 2–3 倍的模型竞争。

> 上表是 **base/PT** 分数。Instruct（IT）变体经过 SFT+RLHF/RLAIF，数学与指令遵从分数会显著更高（如 27B-IT 的 GSM8K/MATH 远超 base）。具体逐项见各 model card。

---

## 5. 架构剖析：local/global 注意力、soft-capping、双 RMSNorm、QK-norm、词表

本节所有断言均可在 `forks/gemma_pytorch/gemma/` 中找到行级佐证。三代共用一份 `model.py`，靠 `config` 开关切换。

### 5.1 配置即架构：三代靠枚举与开关区分

`config.py` 用一个枚举区分注意力类型，一个枚举区分代际：

```python
# config.py:34-43
class AttentionType(enum.Enum):
    GLOBAL = 1            # 全局注意力（看到全部历史）
    LOCAL_SLIDING = 2     # 局部滑窗注意力（只看最近 W 个 token）

class Architecture(enum.Enum):
    GEMMA_1 = 1
    GEMMA_2 = 2
    GEMMA_3 = 3
```

构建模型时按代际选择 decoder layer 类，并把 `attn_types` 序列**按层取模铺开**：

```python
# model.py:469-481
if config.architecture == gemma_config.Architecture.GEMMA_1:
    self.layers.append(GemmaDecoderLayer(config))          # 全 global
elif config.architecture in (GEMMA_2, GEMMA_3):
    attn_type = (config.attn_types[i % len(config.attn_types)]  # 交替模式取模
                 if config.attn_types is not None else GLOBAL)
    self.layers.append(Gemma2DecoderLayer(config, attn_type))
```

### 5.2 local-global 交替注意力（省 KV cache 的核心）

**思想**：长上下文里，多数层只需关注"最近的局部窗口"，少数层负责"全局聚合"。局部层的 KV cache 只需保留窗口大小，显著降低长序列显存。

- **Gemma 2**：`[LOCAL_SLIDING, GLOBAL] * N`，即严格 **1:1** 交替，滑窗 **4096**（`config.py:133-134`）。
- **Gemma 3**：把比例推到 **5:1**（5 局部 + 1 全局），滑窗收到 **1024**（4B/12B/27B）甚至 **512**（1B）：

```python
# config.py:189-197（Gemma 3 1B）
attn_types=(
    AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING, AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL,   # 5 局部 + 1 全局
),
sliding_window_size=512,
```

滑窗如何落到 attention 上？在 `GemmaAttention.forward` 里，局部层把传入的全局 causal mask **替换**为更窄的 `local_mask`：

```python
# model.py:317-322
if (self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING
        and self.sliding_window_size is not None
        and local_mask is not None):
    mask = local_mask     # 局部层改用滑窗 mask（更早的 token 被屏蔽）
```

`local_mask` 的构造（生成阶段）：在普通上三角 causal mask 之外，再用一个"下三角偏移 `-W`"的 mask 把窗口之外的更早 token 也屏蔽掉：

```python
# model.py:672-675
local_mask_tensor = mask_tensor + torch.tril(
    torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
    diagonal=-self.config.sliding_window_size,   # 只保留最近 W 个 token
) if self.config.sliding_window_size else None
```

> 对照 [[04-mistral]]：Mistral 7B 用**全层统一**的 4096 滑窗 + rolling-buffer KV cache；Gemma 的差异是**只在部分层**用滑窗、其余层做全局，因此既省显存又保留长程建模能力，是"local-global 混合"而非"全局滑窗"。

### 5.3 RoPE：local/global 双频率（Gemma 3）

Gemma 3 的全局层把 RoPE base frequency 从 **10k 提到 1M**，并对全局层额外做 `rope_scaling_factor=8` 的波长缩放，从而支持 128K 上下文；局部层仍用 10k：

```python
# config.py:198-201（Gemma 3 1B）
rope_wave_length={
    AttentionType.LOCAL_SLIDING: 10_000,
    AttentionType.GLOBAL:        1_000_000,   # 全局层用大 θ 拉长有效上下文
},
```

`model.py` 因此为 Gemma 3 注册**两套** `freqs_cis` 缓冲区（`local_freqs_cis` / `global_freqs_cis`，`model.py:539-546`），前向时按层的 `attn_type` 取用（`model.py:575-588`）。论文原文："We increase RoPE base frequency from 10k to 1M on global self-attention layers, and keep the frequency of the local layers at 10k."（[2503.19786](https://arxiv.org/html/2503.19786v1)）

### 5.4 logit soft-capping（Gemma 2 标志，Gemma 3 移除）

Gemma 2 在**两处**对 logits 做 `tanh` 软上限，防止数值爆炸、稳住训练：

**(a) 注意力分数** soft-cap 到 50：

```python
# model.py:324-327
if self.attn_logit_softcapping is not None:
    scores = scores / self.attn_logit_softcapping      # 50.0
    scores = torch.tanh(scores)                         # 压到 (-1,1)
    scores = scores * self.attn_logit_softcapping       # 再放大回 ±50
```

**(b) 最终输出 logits** soft-cap 到 30（在 Sampler 内）：

```python
# model.py:53-56
if self.config.final_logit_softcapping is not None:
    logits = logits / self.config.final_logit_softcapping   # 30.0
    logits = torch.tanh(logits)
    logits = logits * self.config.final_logit_softcapping
```

**Gemma 3 改回 QK-norm**：技术报告明确 "we replace the soft-capping of Gemma 2 with QK-norm"（[2503.19786](https://arxiv.org/html/2503.19786v1)）。因此 Gemma 3 的 config 里没有 `*_softcapping` 字段，转而置 `use_qk_norm=True`。

### 5.5 QK-norm（Gemma 3）

QK-norm 对每个 head 的 query / key 向量在**进入注意力打分前**各做一次 RMSNorm（维度 = `head_dim`）：

```python
# model.py:248-257
self.query_norm = (RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                   if config.use_qk_norm else None)
self.key_norm   = (RMSNorm(self.head_dim, eps=config.rms_norm_eps)
                   if config.use_qk_norm else None)
```

```python
# model.py:285-287（在 RoPE 之前应用）
if self.query_norm is not None and self.key_norm is not None:
    xq = self.query_norm(xq)
    xk = self.key_norm(xk)
```

> 对照 [[08-olmo]] / [[02-qwen]]：QK-norm 已成为 2024–2025 稳定性"标配"。OLMo 2、Qwen3 也用 QK-norm。Gemma 3 的特别之处是**用它替换**了上一代的 soft-capping，两者解决的是同一类"注意力 logits 过大"问题。

### 5.6 双重 RMSNorm（pre + post，Gemma 2/3 的归一化堆料）

Gemma 的 RMSNorm 有两个细节：(1) 权重以 **`1 + weight`** 形式作用（unit offset，`model.py:186-187`）；(2) 在 float32 下计算再 cast 回去（`model.py:185`）：

```python
# model.py:182-190
def forward(self, x):
    output = self._norm(x.float())          # 用 fp32 求 RMS，数值更稳
    if self.add_unit_offset:
        output = output * (1 + self.weight.float())   # 注意是 (1 + w)
    else:
        output = output * self.weight.float()
    return output.type_as(x)
```

**关键升级**：Gemma 1 每个 sub-layer 只有 input-norm + post-attention-norm（pre-norm 残差）。Gemma 2/3 的 `Gemma2DecoderLayer` 在 MLP 周围**额外**加了 pre-FFW 与 post-FFW 两个 norm，形成"**每个 sub-layer 进、出都归一化**"的双重结构：

```python
# model.py:415-424
self.pre_feedforward_layernorm  = (RMSNorm(...) if config.use_pre_ffw_norm  else None)
self.post_feedforward_layernorm = (RMSNorm(...) if config.use_post_ffw_norm else None)
```

```python
# model.py:446-456（注意力与 MLP 各有 post-norm 再入残差）
hidden_states = self.post_attention_layernorm(hidden_states)   # 注意力 post-norm
hidden_states = residual + hidden_states
residual = hidden_states
if self.pre_feedforward_layernorm is not None:
    hidden_states = self.pre_feedforward_layernorm(hidden_states)   # MLP pre-norm
hidden_states = self.mlp(hidden_states)
if self.post_feedforward_layernorm is not None:
    hidden_states = self.post_feedforward_layernorm(hidden_states)  # MLP post-norm
```

对比 Gemma 1 的 `GemmaDecoderLayer`（`model.py:373-389`）只有 input/post-attention 两个 norm——**这就是"双重归一化"的代际差异**，由 `use_pre_ffw_norm` / `use_post_ffw_norm` 两个开关控制（Gemma 2/3 全置 True）。

### 5.7 GQA / MQA

注意力支持任意 `num_kv_heads ≤ num_heads`，用 `repeat_interleave` 把 KV 头复制到 query 头数：

```python
# model.py:301-306
if self.num_kv_heads != self.num_heads:
    key   = torch.repeat_interleave(key,   self.num_queries_per_kv, dim=2)
    value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)
```

- Gemma 1 2B：`num_kv_heads=1` → **MQA**；7B：`=16=heads` → **MHA**。
- Gemma 2/3：全系 **GQA**（如 27B：32 query / 16 KV）。

### 5.8 GeGLU 激活（注意：不是 SwiGLU）

Gemma 的 MLP 用 **GeGLU**——gate 分支走 **tanh 近似的 GELU**，而非 Llama 系的 SwiGLU（SiLU gate）：

```python
# model.py:206-212
def forward(self, x):
    gate = self.gate_proj(x)
    gate = F.gelu(gate, approximate="tanh")   # GeGLU：GELU(tanh 近似) 作为门控
    up   = self.up_proj(x)
    fuse = gate * up
    return self.down_proj(fuse)
```

> 这是 Gemma 与 Llama/Mistral 的一个常被忽略的区别：**门控激活用 GELU 而非 SiLU**。论文称之为"approximated GeGLU"。

### 5.9 超大词表 + embedding scaling + tied embeddings

- **词表**：Gemma 1/2 = **256000**（`config.py:50`），Gemma 3 = **262144**（`config.py:202`）。继承自 Gemini，多语言覆盖广。
- **embedding scaling**：embedding 查表后乘以 `sqrt(hidden_size)`，是 Gemma 的固定操作：

```python
# model.py:594-598
# Gemma normalizes the embedding by sqrt(hidden_size).
normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, ...)
hidden_states = hidden_states * normalizer
```

- **tied embeddings**：输出 logits 直接用 embedding 矩阵转置（输入/输出权重共享），见 Sampler `logits = torch.matmul(hidden_states, embedding.t())`（`model.py:50`），`forward` 把 `self.embedder.weight` 传给 sampler（`model.py:608-613`）。

> 小模型里这套 256k 词表的代价很大：Gemma 3 1B 隐藏维度仅 1152，但 embedding 矩阵就有 262144 × 1152 ≈ 3 亿参数——**embedding 占总参数的相当大比例**。这也是 Gemma 3n 用 Per-Layer Embeddings 把 embedding 挪到 CPU 的动机之一。

---

## 6. 源码走读

### 6.1 `model.py`：三代共用的文本解码器

整体数据流（`GemmaForCausalLM.forward`，`model.py:558-620`）：

1. **embedding + scaling**（`:593-598`）：查表后乘 `sqrt(hidden)`。
2. **RoPE 频率选择**（`:573-588`）：Gemma 3 用 local/global 两套 `freqs_cis`，按层 `attn_type` 取（`model.py:499`）；Gemma 1/2 单套。
3. **逐层解码**（`GemmaModel.forward`，`:495-505`）：每层拿到对应 `freqs_cis`、`mask`、`local_mask`、本层 KV cache。
4. **final norm**（`:505`）→ **Sampler**（`:612`）：tied embedding 算 logits、可选 soft-cap、temperature/top-p/top-k 采样（`model.py:46-88`）。

两种 decoder layer 的差异是本文件的核心：

| | `GemmaDecoderLayer`（Gemma 1） | `Gemma2DecoderLayer`（Gemma 2/3） |
|---|---|---|
| 行号 | `model.py:342-391` | `model.py:394-458` |
| attn_type | 恒为 GLOBAL（`:349`） | 由参数传入（local/global，`:401`） |
| norm 数量 | 2（input + post-attn） | 最多 4（+ pre/post-FFW） |
| 传 `local_mask` | 否（`:376-382` 调用 attn 时不传） | 是（`:438-445`） |

> 代码里有一句 `# TODO(imayank): Decouple Gemma versions into separate files.`（`model.py:363`）——三代塞在一份文件里确实是阅读负担，但也方便我们**并排对比**代际差异。

KV cache 是预分配的固定张量，按 `kv_write_indices` 原地写入（`model.py:296-297`），生成循环见 `generate`（`model.py:622-735`）：prefill 到 `min_prompt_len` 后逐 token decode。

### 6.2 `gemma3_model.py`：多模态包装层

`Gemma3ForMultimodalLM`（`gemma3_model.py:30`）在文本模型外包了一层视觉融合，核心是**把图像 patch 编码后塞进文本 token 序列的占位符位置**：

1. **图像编码**（`:102-115`）：`image_patches`（B×N×3×896×896）展平后过 SigLIP，再过 `mm_soft_embedding_norm`（RMSNorm）与 `mm_input_projection`（线性投影到 `hidden_size`）：

```python
# gemma3_model.py:106-109
image_embeddings = self.siglip_vision_model(flattened_input)         # (B*N)xUxD
image_embeddings = self.mm_soft_embedding_norm(image_embeddings)     # 归一化
image_embeddings = self.mm_input_projection(image_embeddings)        # 投影到文本维度
```

2. **占位符替换**（`populate_image_embeddings`，`:142-161`）：找到 `input_token_ids` 中等于 image placeholder 的位置，把对应 SigLIP 输出**原地写入** hidden_states：

```python
# gemma3_model.py:156-160
image_placeholder_mask = input_token_ids == self.tokenizer.image_token_placeholder_id
image_placeholder_indices = image_placeholder_mask.flatten().nonzero(...).squeeze()
hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
hidden_states[image_placeholder_indices] = valid_image_embeddings.reshape(...)
```

3. **双向图像注意力**（`create_attention_mask`，`:163-207`）：文本是 causal，但**同一张图的 patch 之间是双向可见**（bidirectional）——通过给每张图的连续 patch 编号、令同号 patch 互相可见来实现：

```python
# gemma3_model.py:195-199
bidirectional_mask = torch.logical_and(
    kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
    q_block_indices.unsqueeze(-1) > 0,
)
attention_mask = torch.logical_or(causal_mask, bidirectional_mask.unsqueeze(1))
```

这正是"图像内部双向、图文之间因果"的混合注意力，符合 Gemma 3 多模态设计。

### 6.3 `siglip_vision/`：视觉编码器（400M SigLIP）

`SiglipVisionModelConfig`（`siglip_vision/config.py:22-58`）：**27 层** encoder，embedding 维度 **1152**，**16 头** × head_dim **72**，patch conv kernel **14**，MLP **4304**，输入 **896×896**（`preprocessor.py:27`）。这对应论文里的 **400M 变体 SigLIP**。

前向流程（`siglip_vision_model.py:186-203`）：

```python
# siglip_vision_model.py:191-203
x = self.patch_embedding(pixel_values)          # Conv2d(14×14, stride14) 切 patch
x = x.flatten(2).transpose(1, 2)                # → (B, 64*64, 1152)
x = x + self.position_embedding(position_ids)   # 加位置编码
for block in self.encoder_blocks:               # 27 个 pre-LN encoder block
    x = block(x)
x = self.final_norm(x)
return self.avg_pool(x)                          # 4×4 平均池化 → 256 token
```

关键的 **token 压缩**在 `AveragePool2D`（`siglip_vision_model.py:23-43`）：896/14 = 64，故 patch 网格是 64×64 = 4096；做 **4×4 平均池化** → 16×16 = **256 个图像 token**，再喂给文本模型。论文原文："square images resized to 896 × 896" 产生 "256 image tokens"（[2503.19786](https://arxiv.org/html/2503.19786v1)）。

> SigLIP encoder 是标准 **pre-LN ViT**：每个 block 先 LayerNorm 再 attention/MLP，残差在 LayerNorm 之后相加（`siglip_vision_model.py:134-147`）。注意它用 **LayerNorm（带 bias）**，与文本侧的 RMSNorm 不同；attention 也是普通 MHA（无 GQA、无 RoPE），用绝对位置 embedding。

`pan_and_scan.py`（未展开）实现"平移扫描"：对大图/非方图先切多个 crop，每个 crop 各编码 256 token，提升高分辨率细节——对应论文的 Pan & Scan 推理增强。

### 6.4 `tokenizer.py` 与 `scripts/run.py`

- **Tokenizer**（`tokenizer.py:25-53`）：SentencePiece 封装。Gemma 3 额外定义图像 token：`_BEGIN_IMAGE_TOKEN = 255999`、`_END_IMAGE_TOKEN = 256000`（`tokenizer.py:22-23`），并把 `image_token_placeholder_id` 设为 `pad_id`（`tokenizer.py:39`）——图像占位符复用 pad id。
- **run.py**（`scripts/run.py:68-94`）：构造 `config.get_model_config(variant)`、加载权重、`model.generate(prompt, ...)`。默认采样 `top_p=0.95, top_k=64`（`model.py:628-629`）。注意 `_VALID_MODEL_VARIANTS`（`run.py:38`）只列了文本变体；多模态走 `scripts/run_multimodal.py`。

---

## 7. 训练机制（蒸馏为核心）

> ⚠️ 本 fork 是**推理库**，不含训练代码。本节内容以三篇技术报告为准，代码侧仅能佐证"结构开关"。

### 7.1 知识蒸馏（Gemma 2/3 的标志）

**核心做法**：不再用 next-token 的 one-hot 标签做交叉熵，而是让小模型（student）去**拟合更大教师模型（teacher）在每个 token 上输出的完整概率分布**。形式上每个位置最小化
`Σ_v  P_teacher(v | context) · ( -log P_student(v | context) )`。

**为什么有效**（论文论证，已核实）：
- 教师的"软标签"携带的信息量远大于 one-hot，等于把每个 token 变成了一个富监督信号；
- 因而可以在**远超 compute-optimal 的 token 数**上持续训练而不过拟合。Gemma 2 论文原文：用 teacher 在比理论 compute-optimal 多 **50×** 的 token 量上训练小模型（[2408.00118](https://arxiv.org/html/2408.00118v1)）。
- **Gemma 2**：2B、9B 用蒸馏；**27B 仍 from scratch**（没有更大的合适教师）。
- **Gemma 3**：技术报告称"**All Gemma 3 models are trained with knowledge distillation**"——**全系蒸馏**（[2503.19786](https://arxiv.org/html/2503.19786v1)）。

> 这是 Gemma 与 [[02-qwen]] 蒸馏路线的对照点：Qwen3 也用"大模型蒸小模型"（strong-to-weak distillation，off-policy + on-policy），但 Qwen3 的蒸馏更偏**后训练**阶段；Gemma 把蒸馏放在**预训练目标**本身，是更彻底的"蒸馏即预训练"。与 [[06-phi]] 的"教科书级数据 + 合成数据"路线也可对比：Phi 赌**数据质量**，Gemma 赌**教师分布**。

### 7.2 预训练数据

- **Gemma 1**：2B=2T，7B=6T tokens，以英文 web/代码/科学文献为主（[2403.08295](https://arxiv.org/abs/2403.08295)）。
- **Gemma 2**：2B=2T，9B=8T，27B=**13T**（主要英文）（[2408.00118](https://arxiv.org/html/2408.00118v1)）。
- **Gemma 3**：1B=2T，4B=4T，12B=12T，27B=**14T**；并显著增加多语言与（4B+）图文数据（[2503.19786](https://arxiv.org/html/2503.19786v1)）。
- 全系做了安全过滤（CSAM、PII、敏感内容过滤），词表/分词继承 Gemini。

### 7.3 后训练（SFT + RLHF/RLAIF）

- **SFT**：在指令/对话数据上监督微调，得到 `-it`（instruction-tuned）变体。
- **RLHF / RLAIF**：用人类反馈 + AI 反馈的奖励信号做强化学习对齐。Gemma 3 论文强调其"**新的后训练配方**"对数学、对话、指令遵从、多语言的提升尤为关键（使 4B-IT ≈ Gemma2-27B-IT）。
- 使用与 Gemini 一致的对话格式（`<start_of_turn>` / `<end_of_turn>` 控制 token）。

---

## 8. 学习要点与横向对比

### 8.1 一句话记住 Gemma 各代

- **Gemma 1**：Gemini 的"开源小弟"。标准 decoder + GeGLU + 256k 词表 + RoPE；2B 用 MQA、7B 用 MHA。
- **Gemma 2**：**蒸馏** + **local/global 1:1** + **logit soft-capping** + **双重 RMSNorm** + GQA。"practical size"路线。
- **Gemma 3**：**多模态**（SigLIP）+ **128K** + **local/global 5:1** + **QK-norm 替换 soft-capping** + 全系蒸馏。
- **Gemma 3n**：端侧。MatFormer + Per-Layer Embeddings，一个模型抽出多个子模型。

### 8.2 横向对比表

| 维度 | Gemma 3 | [[01-llama]] (Llama 3) | [[02-qwen]] (Qwen3) | [[04-mistral]] (Mistral 7B) |
|---|---|---|---|---|
| 门控激活 | **GeGLU**（GELU gate） | SwiGLU | SwiGLU | SwiGLU |
| 归一化 | RMSNorm，**pre+post 双重** + **QK-norm** | RMSNorm（pre）| RMSNorm + QK-norm | RMSNorm（pre）|
| 长上下文注意力 | **local/global 5:1** 混合 | 全 global（GQA） | 全 global（GQA）| **全层统一滑窗 4096** |
| KV 优化 | GQA + 局部层短滑窗 | GQA | GQA | GQA + rolling buffer |
| 词表 | **262k**（超大）| 128k | 151k | 32k |
| 蒸馏 | **预训练即蒸馏（全系）** | 否（主要靠数据规模）| 后训练蒸馏 | 否 |
| 多模态 | **原生（SigLIP，4B+）** | 单独 Llama 3.2-V | Qwen-VL 系列 | Pixtral 单独线 |
| 许可证 | Gemma Terms（自定义宽松）| Llama Community（自定义）| Apache-2.0（多数）| Apache-2.0 |

### 8.3 值得借鉴的工程点

1. **local/global 混合注意力**比"全层滑窗"（Mistral）更灵活：既省 KV cache 又保留全局建模，5:1 是不错的甜点比例（[[04-mistral]] 对照）。
2. **蒸馏让小模型突破数据墙**：当 high-quality token 有限时，用教师分布"放大"监督信号，是 Gemma 2/3 小尺寸强势的根因（[[02-qwen]]、[[06-phi]] 对照）。
3. **归一化要堆够**：双重 RMSNorm + (1+w) offset + fp32 计算 + QK-norm，是 Gemma 稳定训练的"组合拳"。
4. **soft-capping → QK-norm 的演进**：两者都治"注意力 logits 过大"，但 QK-norm 对推理/编译更友好（无需 tanh、可走 Flash Attention），故 Gemma 3 切换（[[08-olmo]] 也用 QK-norm）。
5. **超大词表的代价要心里有数**：256k+ 词表在 1B 这种小模型里 embedding 占比极高，Gemma 3n 的 PLE 正是对此的回应。

---

## 9. 参考

**论文（arXiv）**
- Gemma 1：*Gemma: Open Models Based on Gemini Research and Technology* — https://arxiv.org/abs/2403.08295
- Gemma 2：*Gemma 2: Improving Open Language Models at a Practical Size* — https://arxiv.org/abs/2408.00118
- Gemma 3：*Gemma 3 Technical Report* — https://arxiv.org/abs/2503.19786 ｜ HTML：https://arxiv.org/html/2503.19786v1
- CodeGemma（HumanEval 对照）：https://arxiv.org/abs/2406.11409

**官方**
- GitHub（PyTorch 推理库）：https://github.com/google/gemma_pytorch
- HuggingFace 组织：https://huggingface.co/google
- Gemma 3 产品页：https://deepmind.google/models/gemma/gemma-3/
- Gemma 3n 开发者指南：https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/
- Gemma Terms of Use：https://ai.google.dev/gemma/terms
- Google AI for Developers（Gemma 文档）：https://ai.google.dev/gemma

**本地源码（fork，commit `014acb7`）**
- `forks/gemma_pytorch/gemma/config.py` — 三代超参（`get_config_for_*`）
- `forks/gemma_pytorch/gemma/model.py` — 文本解码器（Gemma 1/2/3 共用）
- `forks/gemma_pytorch/gemma/gemma3_model.py` — 多模态包装层
- `forks/gemma_pytorch/gemma/siglip_vision/` — SigLIP 视觉编码器（config / model / preprocessor / pan_and_scan）
- `forks/gemma_pytorch/gemma/tokenizer.py` — SentencePiece + 图像 token
- `forks/gemma_pytorch/scripts/run.py` — 文本推理入口

> 免责声明：本文 benchmark 与 token 规模均标注来源；架构数字以**本地 fork 源码**为唯一事实来源（已逐行核对）。Gemma 3 各尺寸预训练 token 数以官方技术报告 2503.19786 为准（27B=14T，区别于 Gemma 2 27B 的 13T）。Gemma 3n 不在本 PyTorch 库内，相关数字以官方开发者博客为准。
