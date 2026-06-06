# Mistral / Mixtral 深度分析（SWA / Rolling Buffer KV Cache / Sparse MoE / GQA / SwiGLU）

> 本文档为 Model-Research 学习仓库的 **Mistral AI** 专题。源码走读基于本地 fork submodule
> `forks/mistral-inference`（官方推理库 `mistralai/mistral-inference`）。
> 所有 `文件:行号` 引用均已对照当前文件内容核对；规格 / benchmark 数字尽量给官方或 arXiv 来源 URL，
> 无法核实者明确标注「未核实 / 约」。撰写日期：2026-06-06。
>
> 一句话定位：**Mistral AI 用「工程化的注意力 + 稀疏 MoE」把推理成本压到极致**——Mistral 7B 靠
> Sliding Window Attention (SWA) + GQA 让 7B 打赢 Llama 2 13B；Mixtral 用 Sparse MoE 把「总参数」
> 与「单 token 算力」解耦（8×7B：47B 总参 / 13B 激活）。绝大多数开源权重采用 **Apache-2.0**。

---

## 1. 概览

**出品方**：Mistral AI（法国巴黎，2023 年 4 月成立，创始团队来自 Meta FAIR 与 DeepMind）。
主打「**开放权重 + 推理效率 + 欧洲主权 AI**」。早期旗舰开源权重多以宽松的 **Apache-2.0** 发布（可商用、可微调、可蒸馏），后期部分前沿 / 专用模型一度采用自有的 **MNPL（非生产）/ MRL（研究）许可证**，2026 年起又把多款模型（如 Codestral 2、Mistral 3 系列）重新放回 Apache-2.0。

**核心理念**：不靠堆参数，而靠**架构与系统工程**取得「同等算力下更强、同等能力下更省」的帕累托前沿：

- **Mistral 7B**：Sliding Window Attention（局部窗口注意力，O(W) 而非 O(n) 显存）+ Grouped-Query Attention（GQA，KV 头共享）+ Rolling Buffer KV Cache（环形缓冲，固定显存）+ SwiGLU + byte-fallback BPE。
- **Mixtral（SMoE）**：每层把单个 FFN 换成 **8 个专家 + Top-2 路由**的稀疏 Mixture-of-Experts；每 token 只激活 2/8 专家，于是「47B 总参只花 13B 的算」。

**版本演进表**（联网核实，截至 2026-06；带 † 者为本仓库 fork 直接支持的模型）：

| 模型 | 时间 | 关键点 | 许可证 | 来源 |
|---|---|---|---|---|
| **Mistral 7B v0.1** † | 2023-09 | SWA(W=4096) + GQA；7B 打赢 Llama 2 13B | Apache-2.0 | [arXiv 2310.06825](https://arxiv.org/abs/2310.06825) |
| Mistral 7B v0.2 / v0.3 † | 2024-03/05 | 32k 上下文、rope θ=1e6、**关闭 SWA**；v0.3 加 function calling、vocab→32768 | Apache-2.0 | [HF 7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| **Mixtral 8x7B v0.1** † | 2023-12 | **SMoE**，Top-2/8 专家，47B/13B，32k | Apache-2.0 | [arXiv 2401.04088](https://arxiv.org/abs/2401.04088) |
| **Mixtral 8x22B** † | 2024-04 | SMoE，141B/39B，64k，原生 function calling | Apache-2.0 | [Mistral 8x22B](https://mistral.ai/news/mixtral-8x22b/) |
| Codestral 22B † | 2024-05 | 代码模型，80+ 语言，FIM | MNPL（非生产）| [Codestral](https://mistral.ai/news/codestral/) |
| Mathstral 7B † | 2024-07 | 数学专用，基于 7B | Apache-2.0 | [README](forks/mistral-inference/README.md) |
| Codestral-Mamba 7B † | 2024-07 | **Mamba2** 架构代码模型（非 Transformer）| Apache-2.0 | [Codestral-Mamba](https://mistral.ai/news/codestral-mamba/) |
| **Mistral NeMo 12B** † | 2024-07 | 与 NVIDIA 合作，**128k 上下文**，Tekken tokenizer | Apache-2.0 | [Mistral NeMo](https://mistral.ai/news/mistral-nemo/) |
| **Mistral Large 2 (123B)** † | 2024-07 | dense 123B，128k，MMLU 84 | MRL（研究）| [Large Enough](https://mistral.ai/news/mistral-large-2407/) |
| **Pixtral 12B** † | 2024-09 | 多模态（视觉+语言），基于 NeMo + ViT | Apache-2.0 | [Pixtral 12B](https://mistral.ai/news/pixtral-12b/) |
| Ministral 3B / 8B | 2024-10 | 端侧小模型，128k；8B 权重为研究许可 | 部分 Apache-2.0 / 研究 | [Ministral](https://mistral.ai/news/ministraux/) |
| Pixtral Large † | 2024-11 | 124B 多模态旗舰 | MRL | [Pixtral Large](https://mistral.ai/news/pixtral-large/) |
| **Mistral Small 3 (24B)** | 2025-01 | dense 24B，MMLU>81%，150 tok/s，**无 RL/无合成数据** | Apache-2.0 | [Mistral Small 3](https://mistral.ai/news/mistral-small-3/) |
| **Mistral Small 3.1 (24B)** † | 2025-03 | Small 3 + 多模态 + 128k | Apache-2.0 | [Small 3.1](https://mistral.ai/news/mistral-small-3-1/) |
| Mistral 3 / Ministral 3 系列 | 2025-12 | 10 款开放权重（含 MoE 前沿与端侧），全 Apache-2.0 | Apache-2.0 | 未完全核实（见 §3 备注）|
| Codestral 2 | 2026-04（约）| 22B 代码模型重新 Apache-2.0 化 | Apache-2.0 | 未核实（见 §3 备注）|

> 说明：本仓库 fork 的 `mistral-inference` 是「**最小推理参考实现**」，覆盖上表带 † 的 Transformer 系
> （以及 Mamba 系的 `mamba.py`）。2025-12 之后的 Mistral 3 / Codestral 2 等为联网检索到的后续公告，
> 细节标注「未核实」，不作为本文档的硬性结论。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| 官方 GitHub（推理库） | https://github.com/mistralai/mistral-inference |
| 官方 GitHub（tokenizer / 协议） | https://github.com/mistralai/mistral-common |
| 本仓库 fork（submodule） | `forks/mistral-inference`（`src/mistral_inference/*.py`、`README.md`、`tutorials/`） |
| HuggingFace 组织 | https://huggingface.co/mistralai |
| HF: Mistral 7B Instruct v0.3 | https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 |
| HF: Mixtral 8x7B Instruct v0.1 | https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1 |
| HF: Mixtral 8x22B v0.1 | https://huggingface.co/mistralai/Mixtral-8x22B-v0.1 |
| HF: Mistral NeMo Instruct 2407 | https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407 |
| 官方文档 | https://docs.mistral.ai/ |
| 论文：Mistral 7B | [arXiv 2310.06825](https://arxiv.org/abs/2310.06825) |
| 论文：Mixtral of Experts | [arXiv 2401.04088](https://arxiv.org/abs/2401.04088) |
| 博客：Announcing Mistral 7B | https://mistral.ai/news/announcing-mistral-7b/ |
| 博客：Mixtral of experts | https://mistral.ai/news/mixtral-of-experts/ |

---

## 3. 模型规格

下表「架构超参」列以 **arXiv 论文 Table 1** 为准，并与本地 fork / HF config 交叉验证。

### 3.1 Transformer 主体规格

| 维度 | Mistral 7B (v0.1) | Mixtral 8x7B | Mixtral 8x22B | Mistral NeMo 12B | 来源 |
|---|---|---|---|---|---|
| 总参数 | ~7.24B | ~46.7B（≈47B）| ~141B | ~12B | 论文 / HF |
| 每 token 激活参数 | 全量 (~7B) | **~13B** | **~39B** | 全量 (~12B) | [2401.04088](https://arxiv.org/abs/2401.04088) |
| dim (hidden) | 4096 | 4096 | 6144 | 5120 | 论文 Table 1 / HF |
| n_layers | 32 | 32 | 56 | 40 | 论文 / HF |
| n_heads | 32 | 32 | 48 | 32 | 论文 / HF |
| n_kv_heads (GQA) | 8 | 8 | 8 | 8 | 论文 / HF |
| head_dim | 128 | 128 | 128 | 128 | 论文 / HF |
| hidden_dim (FFN) | 14336 | 14336 | 16384 | 14336 | 论文 / HF |
| num_experts | — | **8** | **8** | — | [2401.04088](https://arxiv.org/abs/2401.04088) |
| experts/tok (top-k) | — | **2** | **2** | — | [2401.04088](https://arxiv.org/abs/2401.04088) |
| 上下文 | 8192（v0.1，可经 SWA 外推）| 32768 | 65536 | **131072 (128k)** | 论文 / [NeMo](https://mistral.ai/news/mistral-nemo/) |
| sliding_window | **4096**（v0.1）| 4096（v0.1）| null | null | 论文 / HF |
| rope θ | 10000（v0.1）→ 1e6（v0.3）| 1e6 | 1e6 | 1e6 | HF config |
| vocab_size | 32000（v0.1）→ 32768（v0.3）| 32000 | 32768（v0.3）| **131072**（Tekken）| HF config |

> 关键纠偏（务必注意版本差异）：
> 1. **SWA 只在 7B v0.1 默认开启（W=4096）**。Mistral 7B **v0.2/v0.3** 的 HF `config.json` 里
>    `sliding_window: null`、`rope_theta: 1000000.0`、`max_position_embeddings: 32768`
>    （[HF 7B-Instruct-v0.3 config](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)）——
>    即新版**关闭了 SWA**、改用全注意力 + 更大 rope θ 直接支持 32k。
> 2. 本地 fork 的默认 rope θ 是 **1e6**（见 §6.4 `transformer.py:115`），与 v0.3+ 一致，不是 v0.1 的 1e4。
> 3. Mixtral「47B」不是 8×7=56B：因为**注意力 / embedding / norm 等非专家参数被 8 个专家共享**，
>    只有 FFN（专家）部分乘以 8，故总参约 46.7B。

### 3.2 后续 / 专用模型一览（联网核实）

| 模型 | 规模 | 上下文 | 定位 | 许可证 | 来源 |
|---|---|---|---|---|---|
| Mistral Large 2 (2407) | dense 123B | 128k | 通用旗舰，MMLU 84 | MRL（研究）| [Large Enough](https://mistral.ai/news/mistral-large-2407/) |
| Mistral Small 3 / 3.1 | dense 24B | 32k / 128k | 高性价比通用，3.1 加多模态 | Apache-2.0 | [Small 3](https://mistral.ai/news/mistral-small-3/) |
| Codestral 22B | dense 22B | 32k | 代码（80+ 语言，FIM）| MNPL（非生产）| [Codestral](https://mistral.ai/news/codestral/) |
| Codestral-Mamba 7B | 7B（Mamba2）| ∞（理论）| 代码，线性复杂度 | Apache-2.0 | [Codestral-Mamba](https://mistral.ai/news/codestral-mamba/) |
| Mathstral 7B | 7B | 32k | 数学 / STEM | Apache-2.0 | [README](forks/mistral-inference/README.md) |
| Ministral 3B / 8B | dense 3B / 8B | 128k | 端侧 / 边缘 | 部分 Apache-2.0 / 研究 | [Ministral](https://mistral.ai/news/ministraux/) |
| Pixtral 12B | 12B + ViT | 128k | 多模态（图文）| Apache-2.0 | [Pixtral 12B](https://mistral.ai/news/pixtral-12b/) |
| Pixtral Large | 124B + ViT | 128k | 多模态旗舰 | MRL | [Pixtral Large](https://mistral.ai/news/pixtral-large/) |

> 备注：2025-12 的 **Mistral 3 / Ministral 3** 系列与 2026 的 **Codestral 2** 公告称「全面 Apache-2.0、
> 含 MoE 前沿模型」，但截至撰写日（2026-06）笔者未取得稳定的官方规格页一手核实，**视为「约 / 未核实」**，
> 不进入上面的硬规格表。

---

## 4. Benchmark

> 口径提醒：不同来源 shot 数 / 解码策略不同，跨模型横比仅供参考；优先采用**论文 / 官方 model card**。

### 4.1 Mistral 7B v0.1（vs Llama 2 / Llama 1，论文 Table 2）

| Benchmark | Mistral 7B | Llama 2 7B | Llama 2 13B | 来源 |
|---|---|---|---|---|
| MMLU (5-shot) | **60.1** | 44.4 | 55.6 | [2310.06825](https://arxiv.org/abs/2310.06825) |
| GSM8K (maj@8) | **52.2** | 16.0 | 34.3 | [2310.06825](https://arxiv.org/abs/2310.06825) |
| MATH (maj@4) | **13.1** | 3.9 | 6.0 | [2310.06825](https://arxiv.org/abs/2310.06825) |
| HumanEval (pass@1) | **30.5** | 11.6 | 18.9 | [2310.06825](https://arxiv.org/abs/2310.06825) |
| MBPP | **47.5** | 26.1 | 35.4 | [2310.06825](https://arxiv.org/abs/2310.06825) |

> 论文核心论断：**Mistral 7B 在全部评测上超过 Llama 2 13B**，并在推理 / 数学 / 代码上超过 Llama 1 34B。

### 4.2 Mixtral 8x7B（vs Llama 2 70B / GPT-3.5，论文）

| Benchmark | Mixtral 8x7B | Llama 2 70B | GPT-3.5 | 来源 |
|---|---|---|---|---|
| MMLU | **70.6** | 69.9 | 70.0 | [2401.04088](https://arxiv.org/abs/2401.04088) |
| GSM8K | **74.4** | 69.6 | 57.1 | [2401.04088](https://arxiv.org/abs/2401.04088) |
| MATH | **28.4** | 13.8 | — | [2401.04088](https://arxiv.org/abs/2401.04088) |
| HumanEval | **40.2** | 29.3 | — | [2401.04088](https://arxiv.org/abs/2401.04088) |
| MT-Bench（Instruct）| **8.30** | 6.86 | 8.32（约）| [2401.04088](https://arxiv.org/abs/2401.04088) |

> 论文核心论断：Mixtral 8x7B 以 **~13B 激活参数**匹配 / 超过 **70B dense** 的 Llama 2 70B，
> 数学 / 代码 / 多语言尤为突出；Instruct 版 MT-Bench 8.30 超过 GPT-3.5 Turbo / Claude-2.1 / Gemini Pro。

### 4.3 其他（官方 / 检索，标注来源）

| 模型 | MMLU | HumanEval | 其他 | 来源 |
|---|---|---|---|---|
| Mixtral 8x22B | ~77.75 | 高（具体值未核实）| 64k 上下文 | [8x22B](https://mistral.ai/news/mixtral-8x22b/) |
| Mistral NeMo 12B | ~68.0 | — | 128k；多语言强 | [NeMo](https://mistral.ai/news/mistral-nemo/) |
| Mistral Large 2 | ~84.0 | ~92（约，未核实）| 128k | [Large Enough](https://mistral.ai/news/mistral-large-2407/) |
| Mistral Small 3 (24B) | >81 | 强（具体未核实）| 150 tok/s | [Small 3](https://mistral.ai/news/mistral-small-3/) |
| Codestral 22B | — | ~81.1 | MBPP 78.2 / RepoBench 34.0 | [Codestral](https://mistral.ai/news/codestral/) |
| Ministral 8B | ~65.0 | — | 端侧 | [Ministral](https://mistral.ai/news/ministraux/) |

---

## 5. 架构剖析

Mistral / Mixtral 的 decoder block 结构是**标准的 pre-norm LLaMA 式**（RMSNorm → Attention → 残差 →
RMSNorm → FFN/MoE → 残差），创新点集中在**注意力的局部化（SWA + 环形 KV cache）**与 **FFN 的稀疏化（SMoE）**。

### 5.1 Block 结构（pre-norm 残差）

走读 `transformer_layers.py:158-169`：

```python
def forward(self, x, freqs_cis, cache=None, mask=None):
    r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)  # 先 RMSNorm 再注意力
    h = x + r                                                             # 残差 1
    r = self.feed_forward.forward(self.ffn_norm(h))                       # FFN 或 MoE（同一接口）
    out = h + r                                                           # 残差 2
    return out
```

> `feed_forward` 在构造时按 `moe is None?` 二选一：dense 用 `FeedForward`，否则用 `MoeLayer`
> （`transformer_layers.py:148-156`）。两者 `forward(x)->x` 签名一致，所以 block 代码对 dense / MoE 完全无感——
> 这正是「Mixtral = Mistral 把 FFN 换成 MoE」在工程上的体现。

### 5.2 Sliding Window Attention（SWA）——原理与收益

**问题**：标准全注意力的算力 O(n²·d)、KV cache 显存 O(n)，长序列时都爆炸。

**SWA 思路**（论文 §2.1）：第 k 层位置 i 的 token **只 attend 上一层 [i−W, i] 窗口内**的 token（W=4096）。

- **单层显存 / 算力**：从 O(n) 降到 **O(W)**（窗口固定，与序列长度脱钩）。论文称 32k 序列上
  **rolling buffer cache 显存降 8×**（[2310.06825](https://arxiv.org/abs/2310.06825)，§2.2）。
- **感受野随层数线性扩张**：虽然单层只看 W，但信息每经过一层就向前传播 W 个位置，**第 k 层的等效感受野 ≈ W×k**。
  W=4096、32 层 → 理论可覆盖 **≈131K（128k）token**（论文 §2.1：「approximately 131K tokens」），
  这就是 7B v0.1 能处理远超 8192 训练长度序列的原因。

> 直觉类比：SWA 像 CNN 的「局部卷积核 + 深层堆叠扩大感受野」，用「深度」换「单层宽度」。
> 在 fork 中，SWA 不在注意力 kernel 里写死，而是通过**给 xformers 传一个 local attention bias mask**
> 来实现（见 §6.5 `cache.py:240`）。

### 5.3 Rolling Buffer KV Cache（环形缓冲）

既然每个 token 最多只需要最近 W 个 KV，cache 就**不必随序列增长**，而是用**固定 W 大小的环形缓冲**：
位置 i 的 K/V 存进 `i mod W` 槽位，新 token 覆盖最老的（已超出窗口、不再被需要的）token。

走读 `cache.py:235`（计算「token 该写进环形 buffer 的哪个槽」）：

```python
cache_positions = positions % cache_size + batch_idx * cache_size  # 绝对位置对 W 取模 → 环形槽位
```

走读 `cache.py:59-67`（读取时把环形 buffer「**展开 unrotate**」回时间顺序）：

```python
def unrotate(cache, seqlen):
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:   # 还没绕满一圈：直接取前 seqlen 个
        return cache[:seqlen]
    elif position == 0:           # 正好整圈：顺序就是对的
        return cache
    else:                         # 绕过一圈：把 [position:] 和 [:position] 拼回时间序
        return torch.cat([cache[position:], cache[:position]], dim=0)
```

> RoPE 用的是**绝对位置** `positions`（`cache.py:229`），而 KV 落盘用的是**取模后的环形位置**，两者解耦——
> 所以即便 KV 被覆盖循环，旋转位置编码依旧正确。注意 fork 的 `BufferCache` 注释（`cache.py:140-144`）
> 明说这是「示例实现，矩形分配有浪费，生产请用 PagedAttention」。

### 5.4 Grouped-Query Attention（GQA）

`n_heads=32` 个 Query 头，但只有 `n_kv_heads=8` 个 K/V 头——**每 4 个 Q 头共享 1 组 KV**
（`repeats = n_heads // n_kv_heads = 4`，`transformer_layers.py:46`）。

收益：KV cache 显存与 KV 投影算力**直接除以 4**（相对 MHA），是长上下文 / 高吞吐的关键。
实现上在注意力里用 `repeat_kv` 把 8 组 KV 复制成 32 组以对齐 Q（`transformer_layers.py:16-19, 84`）。

### 5.5 Sparse Mixture-of-Experts（SMoE，Mixtral 核心）

每层把 1 个 dense FFN 换成 **8 个独立 FFN 专家 + 1 个线性门控（router）**；每 token 由 router 选 **Top-2** 专家，
输出按门控权重加权求和。论文公式（[2401.04088](https://arxiv.org/abs/2401.04088)）：

$$ y = \sum_{i=0}^{n-1} \mathrm{Softmax}\big(\mathrm{Top2}(x \cdot W_g)\big)_i \cdot \mathrm{SwiGLU}_i(x) $$

要点：
- **稀疏激活**：8 个专家里每 token 只跑 2 个 → 算力 ≈ 2 个 FFN，而非 8 个。于是「47B 总参 / 13B 激活」。
- **Top-2 后再 softmax**：softmax 只在**被选中的 2 个 logit** 上做归一化（不是全 8 个），未选中的专家权重恒为 0。
- **token 级路由**：路由在每层、每 token 独立，相邻 token / 不同层可选不同专家。

源码细节见 §6.3。

### 5.6 SwiGLU 与归一化

- **FFN = SwiGLU**：`w2( silu(w1(x)) * w3(x) )`，三个无 bias 线性层（`transformer_layers.py:96-106`）。
  `hidden_dim=14336 ≈ 3.5×dim`（SwiGLU 因门控多一支 `w3`，惯例把隐藏维取 ~2/3×4×dim）。
- **RMSNorm**：`x * rsqrt(mean(x²)+eps) * weight`，且**在 fp32 下算 norm** 再 cast 回（`transformer_layers.py:109-120`），数值更稳。

### 5.7 RoPE（旋转位置编码）

复数实现：预计算 `freqs_cis`（`rope.py:6-10`），施加时把相邻两维当复数旋转（`rope.py:13-23`）。
fork 默认 `theta=1e6`、预计算到 `128_000` 位置（`transformer.py:115-116`），对应 v0.3+ 的大 θ / 长上下文配置。
Pixtral 等视觉模型另有 **2D RoPE**（`rope.py:26-51`）。

---

## 6. 源码走读（逐模块）

> 路径前缀统一为 `forks/mistral-inference/src/mistral_inference/`。行号对照当前 fork 内容核对。

### 6.1 `transformer.py` —— 模型主体与前向

`Transformer.forward_partial`（`transformer.py:163-219`）是核心：

```python
# transformer.py:198-209
freqs_cis = self.freqs_cis[input_metadata[0].positions]   # 按绝对位置取 RoPE 频率
for local_layer_id, layer in enumerate(self.layers.values()):
    if cache is not None:
        cache_metadata = input_metadata[local_layer_id]
        cache_view = cache.get_view(local_layer_id, cache_metadata)  # 每层一个 cache 视图
    else:
        cache_view = None
    h = layer(h, freqs_cis, cache_view)                   # 逐 block 前向
```

- **变长打包（no padding）**：输入是把一个 batch 的多条序列**拼平**成 `(sum_seqlen,)`，配 `seqlens` 列表
  （`transformer.py:178-179`），靠 xformers 的 `BlockDiagonalMask` 区分序列边界——零 padding 浪费。
- **流水线并行**：按 `pipeline_rank` 把层切片到不同 rank（`transformer.py:94-98`），
  rank 间用 `torch.distributed.send/recv` 传激活（`transformer.py:196, 213-214`）。
- **多模态融合**：若配置了 `vision_encoder`，`embed_vision_language_features`
  （`transformer.py:122-161`）把图像 patch 特征按 `image_token_id` 占位符**散射**进文本 embedding 序列——这是 Pixtral 的入口。
- `freqs_cis` 故意**不注册为 buffer**（注释 `transformer.py:110-112`），因为它是 complex64，dtype 与模块不同。

### 6.2 `transformer_layers.py` —— Attention / FFN / Block

注意力前向（`transformer_layers.py:56-93`），SWA 通过 mask 注入：

```python
# transformer_layers.py:66-70
xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)   # KV 头数 = 8（GQA）
xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)     # 先 RoPE

# transformer_layers.py:84-88
key, val = repeat_kv(key, val, self.repeats, dim=1)        # 8 组 KV → 复制成 32 组对齐 Q
xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
output = memory_efficient_attention(xq, key, val, mask if cache is None else cache.mask)  # SWA 在 mask 里
```

> 关键：**SWA 不在这里写死**，而是 prefill / decode 时由 `cache.mask`（局部窗口 bias）决定哪些位置可见
> （见 §6.5）。`cache is None`（无 cache 的纯前向，如训练 / 打分）时走外部传入的 `mask`。

`assert mask is None or cache is None`（`transformer_layers.py:63`）——mask 与 cache 互斥，二者只取其一。

FFN（SwiGLU）：`transformer_layers.py:105-106` `return self.w2(F.silu(self.w1(x)) * self.w3(x))`。

### 6.3 `moe.py` —— SMoE 门控与专家加权（**最关键的 ~30 行**）

整个稀疏 MoE 只有一个 `forward`（`moe.py:24-32`）：

```python
def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    gate_logits = self.gate(inputs)                                  # (T, num_experts=8) 路由打分
    weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)  # 取 Top-2
    weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)  # 只在被选 2 个上 softmax
    results = torch.zeros_like(inputs)
    for i, expert in enumerate(self.experts):                        # 遍历 8 个专家
        batch_idx, nth_expert = torch.where(selected_experts == i)   # 哪些 token 选了专家 i
        results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
    return results
```

逐行注释：
1. `gate`（构造于 `transformer_layers.py:152`，是 `nn.Linear(dim, num_experts, bias=False)`）给出每 token 对 8 专家的 logit。
2. `torch.topk(..., 2)` 选出 Top-2 专家的 logit 与索引（实现公式里的 `Top2`）。
3. `softmax(weights)` **只对这 2 个值归一化** → 两专家权重和为 1（对应公式 `Softmax(Top2(·))`）。
4. 循环里 `torch.where(selected_experts == i)` 找出「**把专家 i 当作其 top-1 或 top-2**」的所有 token，
   仅对这些 token 跑 `expert(inputs[batch_idx])`，再乘以对应门控权重累加 → 实现 `Σ weight_i · SwiGLU_i(x)`。

> 工程注记：这是**最朴素的「按专家循环 + 稀疏 scatter」实现**，可读性最佳但非最优吞吐
> （生产推理用 grouped GEMM / Megablocks 等把稀疏 dispatch 融合）。这里**没有 auxiliary load-balancing loss**——
> 那是训练侧的事，推理库不含。对比 DeepSeekMoE 的「细粒度共享专家 + 无辅助损失负载均衡」见 §8。

`MoeArgs`（`moe.py:10-14`）只有两个字段：`num_experts` 与 `num_experts_per_tok`，
分别对应 Mixtral 的 **8** 与 **2**。

### 6.4 `rope.py` —— 旋转位置编码

```python
# rope.py:6-10
def precompute_freqs_cis(dim, end, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 频率基
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # 复数 e^{iθ}，complex64
```

施加（`rope.py:13-23`）：把 K/Q 相邻两维 `view_as_complex` 成复数、乘 `freqs_cis`、再 `view_as_real` 摊平——
即标准的「成对维度做 2D 旋转」。`theta` 在 `transformer.py:115` 默认 1e6（大 θ 利于长上下文外推）。

### 6.5 `cache.py` —— SWA mask 与环形缓冲（系统精华）

不同推理阶段用不同的 xformers 注意力 bias（`cache.py:236-254`）：

```python
# cache.py:238-254（节选）
if first_prefill:        # 首次 prefill（整段 prompt 一次喂入）
    mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(cache_size)  # 因果 + 局部窗口
elif subsequent_prefill: # 续 prefill（分 chunk 喂长 prompt）
    mask = BlockDiagonalMask.from_seqlens(...).make_local_attention_from_bottomright(cache_size)
else:                    # 逐 token decode
    mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=seqlens, kv_padding=cache_size,
        kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=cache_size).tolist())
```

- `make_local_attention(cache_size)` 把因果 mask **再裁一个宽 W 的局部窗口** → 这就是 SWA 的「窗口」落地处。
- `to_cache_mask`（`cache.py:226-227`）只保留每条序列**最后 W 个 token** 写入 cache（注释示例见 `cache.py:197-206`）——
  与环形 buffer 容量一致。
- `get_cache_sizes`（`cache.py:13-24`）支持**逐层不同窗口**（`sliding_window` 可传 list，按层循环），
  为「部分层全局、部分层局部」的混合注意力（类似 `[[05-gemma]]` 的 local/global 交替）预留了接口。

### 6.6 `generate.py` —— 分块 prefill + 自回归解码

```python
# generate.py:67-76
cache_window = max(seqlens) + max_tokens                 # cache 至少容纳 prompt+生成
cache = BufferCache(model.n_local_layers, max_batch_size, cache_window,
                    model.args.n_kv_heads, model.args.head_dim, model.args.sliding_window)
```

`generate`（`generate.py:44-148`）先**按 `chunk_size` 分块编码 prompt**（`generate.py:91-100`，长 prompt 省峰值显存），
再逐 token 采样。`cache_window` 取 `max_seqlen + max_tokens`；若模型设了 `sliding_window`，
`BufferCache` 会按窗口而非全长分配 KV（`cache.py:160-167`）——这是 SWA 省显存的最终兑现处。

---

## 7. 训练机制

> 诚实声明：**Mistral AI 对预训练数据与训练流程披露极少**（远少于 `[[08-olmo]]` 的全开放）。
> 论文 / model card 基本只给架构与评测，不给数据配比 / 训练 token 数 / 算力。以下区分「已知」与「合理推断」。

### 7.1 已知（来自论文 / 官方）

- **架构层面**：7B = SWA + GQA + SwiGLU + RMSNorm + RoPE；Mixtral = 在此之上把 FFN 换成 Top-2/8 SMoE
  （[2310.06825](https://arxiv.org/abs/2310.06825) / [2401.04088](https://arxiv.org/abs/2401.04088)）。
- **上下文 / tokenizer**：Mixtral 以 32k 上下文训练；NeMo / Pixtral 改用 **Tekken**（基于 tiktoken，>100 语言，
  压缩率优于旧 SentencePiece，vocab 131072）。早期模型用 **byte-fallback BPE**（SentencePiece）——
  任意字节都能编码，**永不 OOV / UNK**，对代码与多语言友好。
- **指令版后训练**：官方多次提到 Instruct 版用 **SFT + DPO（Direct Preference Optimization）**；
  Mixtral 8x7B-Instruct 经 SFT+DPO 后 MT-Bench 8.30（[2401.04088](https://arxiv.org/abs/2401.04088)）。
- **Mistral Small 3 特例**：官方明确「**既未用 RL、也未用合成数据**」训练
  （[Small 3](https://mistral.ai/news/mistral-small-3/)），定位为「干净基座」。

### 7.2 合理推断（标注：推断，非官方确认）

- **数据规模**：7B 级别普遍训练 1–8T token 量级；Mistral 未公布具体数，**推断**与同期 Llama 2（~2T）同量级或更多。**（推断）**
- **MoE 训练**：Mixtral 训练时**几乎必然带 router 的负载均衡 auxiliary loss**（防专家坍塌），
  尽管推理库 `moe.py` 不含该 loss——这是 SMoE 训练通识。**（推断）**
- **专家专精度**：论文分析称专家**未表现出明显的「领域 / 主题」专精**，更多是**句法 / token 级**的路由模式
  （[2401.04088](https://arxiv.org/abs/2401.04088) 的路由分析）——这是少数官方给出的训练侧洞见。

### 7.3 推理侧工程（fork 直接体现训练/部署解耦）

- **流水线并行**（`transformer.py:94-98`）、**变长无 padding 打包**（`transformer.py:178`）、
  **环形 KV cache**（§5.3）、**LoRA 推理**（`lora.py`，`maybe_lora` 在 `transformer_layers.py:22-28`）
  ——这些是把大模型「跑起来 / 跑得省」的系统手段，独立于训练。

---

## 8. 学习要点与横向对比

### 8.1 三个「一句话」记忆点

1. **SWA = 用深度换窗口宽度**：单层只看 W，靠堆层把感受野放大到 W×层数；KV 显存从 O(n) 降到 O(W)。
   （但注意 7B v0.2/v0.3 已**关闭 SWA**改全注意力，SWA 主要是 v0.1 的卖点。）
2. **SMoE = 解耦总参与算力**：Top-2/8 路由让 47B 总参只花 13B 的算；门控「Top-k 后再 softmax」。
3. **Mixtral = Mistral 换个 FFN**：block 代码对 dense / MoE 完全无感（`transformer_layers.py:148-156`）。

### 8.2 SMoE 横向对比

| 维度 | **Mixtral**（本文）| DeepSeekMoE（`[[03-deepseek]]`）| MiniMax-01（`[[09-minimax]]`）|
|---|---|---|---|
| 专家粒度 | 粗粒度，8 大专家 | **细粒度** + 共享专家 | MoE + **Lightning（线性）注意力** |
| 每 token 激活 | Top-2/8 | 多个细专家 + 常驻共享专家 | 见 `[[09-minimax]]` |
| 负载均衡 | 训练用 aux loss（推理库不含）| **auxiliary-loss-free**（bias 调整）| 见 `[[09-minimax]]` |
| 路由专精 | 偏句法 / token 级，非领域级 | 设计上鼓励专家分工 | — |
| 参数账 | 47B 总 / 13B 激活 | 671B 总 / 37B 激活（V3）| — |

> 看点：Mixtral 是「**最干净的 SMoE 教科书实现**」；DeepSeek 在其上做了**细粒度切分 + 共享专家 +
> 免辅助损失负载均衡**等多项改良。读懂 `moe.py` 这 9 行，再看 DeepSeekMoE 会非常顺。

### 8.3 SWA / 注意力横向对比

| 维度 | **Mistral SWA**（本文）| Gemma（`[[05-gemma]]`）| 全注意力（如 `[[01-llama]]`）|
|---|---|---|---|
| 局部窗口 | 全层 W=4096（v0.1）| **local/global 层交替**（部分层滑窗、部分层全局）| 无窗口，全局 |
| KV 显存 | O(W)，环形 buffer | local 层 O(W)，global 层 O(n) | O(n) |
| 长程信息 | 靠层数堆叠传播 | global 层直接保证 | 天然全局 |

> Mistral v0.1 是**全层统一滑窗**；Gemma 2/3 把它改进为**滑窗与全局交替**——既省显存又保留少数全局层兜底长程依赖。
> 有趣的是，fork 的 `cache.py:13-24` `get_cache_sizes` **已支持逐层不同窗口（传 list）**，
> 架构上与 Gemma 式混合注意力同源。

### 8.4 与本仓库其它专题的关系

- 注意力 / 位置编码基础设施（RoPE、GQA、RMSNorm、SwiGLU）几乎与 `[[01-llama]]` 一致——Mistral 是 LLaMA 谱系的工程化分支。
- MoE 主题串读：`[[03-deepseek]]`（细粒度 MoE + MLA）↔ 本文 `[[04-mistral]]`（粗粒度 SMoE）↔ `[[09-minimax]]`（MoE + 线性注意力）↔ `[[07-glm]]`。
- 长上下文 / 高效注意力：本文 SWA ↔ `[[05-gemma]]` 混合注意力 ↔ `[[03-deepseek]]` DSA（稀疏注意力）。
- 完全开放训练的对照系：`[[08-olmo]]`（数据 / 代码 / 中间 checkpoint 全开）——正好反衬 Mistral 训练细节的封闭。

### 8.5 学习者动手建议

- 把 `moe.py:24-32` 抄下来、设 `num_experts=8, num_experts_per_tok=2`，喂随机张量打印
  `selected_experts` 与 `weights`，亲手验证「Top-2 后 softmax、权重和=1」。
- 读 `cache.py:197-206` 的注释示例（`seqlens=[5,7,2]`, `sliding_window=3`），手算 `to_cache_mask` /
  `cache_positions`，彻底理解环形 buffer 的取模映射。
- 对照 `[[01-llama]]`：把 Mistral 的 `transformer_layers.py` 与 LLaMA 的同名实现 diff，会发现差异极小——
  主要就是 SWA mask 与（Mixtral 的）MoE FFN。

---

## 9. 参考

**论文 / arXiv**
- Mistral 7B — [arXiv 2310.06825](https://arxiv.org/abs/2310.06825)（[ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2310.06825)）
- Mixtral of Experts — [arXiv 2401.04088](https://arxiv.org/abs/2401.04088)（[ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2401.04088)）

**官方代码 / 模型**
- GitHub: https://github.com/mistralai/mistral-inference ｜ tokenizer: https://github.com/mistralai/mistral-common
- HuggingFace 组织: https://huggingface.co/mistralai
- 本仓库 fork: `forks/mistral-inference`（`src/mistral_inference/{transformer,transformer_layers,moe,rope,cache,generate,args,model}.py`、`README.md`、`tutorials/`）

**官方博客 / 公告**
- Announcing Mistral 7B — https://mistral.ai/news/announcing-mistral-7b/
- Mixtral of experts — https://mistral.ai/news/mixtral-of-experts/
- Mixtral 8x22B — https://mistral.ai/news/mixtral-8x22b/
- Mistral NeMo — https://mistral.ai/news/mistral-nemo/
- Mistral Large 2 — https://mistral.ai/news/mistral-large-2407/
- Pixtral 12B — https://mistral.ai/news/pixtral-12b/
- Codestral — https://mistral.ai/news/codestral/
- Mistral Small 3 — https://mistral.ai/news/mistral-small-3/ ｜ Small 3.1 — https://mistral.ai/news/mistral-small-3-1/
- 官方文档 — https://docs.mistral.ai/

**横向对比文档**：`[[01-llama]]`、`[[02-qwen]]`、`[[03-deepseek]]`、`[[05-gemma]]`、`[[06-phi]]`、`[[07-glm]]`、`[[08-olmo]]`、`[[09-minimax]]`

> 免责声明：本文档为学习用途的二手整理。规格 / benchmark 以 arXiv 论文与官方 model card 为准；
> 标注「约 / 未核实 / 推断」处请回到一手来源确认。源码行号基于撰写日 fork 内容，后续上游更新可能偏移。
