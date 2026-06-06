# Phi (Microsoft)：用"数据为王"让小模型对标大模型

> 本文聚焦微软 **Phi** 系列小语言模型（Small Language Model, SLM）。Phi 的最大价值不在某一项 benchmark，而在它把一条研究主线推到了极致：**"Textbooks Are All You Need"（教科书就是你需要的全部）**——通过**合成"教科书级"数据 + 重度筛选网页数据**，让 1.3B~14B 的小模型在推理、数学、代码上对标大它好几倍的模型。这是 Phi 最值得学的一点：**模型能力的上限，越来越由"数据质量"而非"参数规模"决定**。
>
> 注意本仓库 fork 的是 **PhiCookBook**（微软官方的"教程 / 微调配方"合集，纯 Markdown），**不含架构源码**。因此本文的"架构剖析 / 源码走读（5、6.1 章）"基于 HuggingFace `transformers` 的 `modeling_phi3.py` / `modeling_phimoe.py`（给出 URL + 行号）；"微调配方走读（6.2 章）"基于本地 `forks/PhiCookBook/md/03.FineTuning/...`。
>
> 源码 fork 位置：`forks/PhiCookBook`（git submodule，对应 `github.com/microsoft/PhiCookBook`，本地 commit `0e5d477`，最后提交 2026-05-26；`md/` 下 82 个 Markdown、约 1.35 万行，体积大的图片/翻译已被 sparse-checkout 过滤）。
>
> 横向阅读：[[08-olmo]]（数据全开放 / 训练全过程）、[[02-qwen]]（蒸馏 / 大小模型族）、[[03-deepseek]]（推理 / RL）、[[04-mistral]]、[[05-gemma]]、[[01-llama]]。

---

## 1. 概览

**Phi** 是微软（Microsoft Research）推出的开源 **小语言模型（SLM）** 系列。它的定位用一句话概括：**"小而强"（small but mighty）+ 数据为王（data-centric）**——在 1.3B 到 14B 这个"能跑在边缘设备 / 单卡"的尺寸区间内，靠**数据策划（data curation）**把性能顶到同尺寸甚至上一档模型的水平。

与多数厂商"堆参数、堆 token"的路线不同，Phi 团队（Sébastien Bubeck 等）从 Phi-1 起就押注一个假说：

> **如果训练数据足够"教科书级"（textbook-quality）——逻辑清晰、循序渐进、几乎无噪声——那么小模型也能学到强推理能力。** 规模不是唯一答案，**数据质量**才是。

这条主线贯穿整个家族：从 Phi-1 用合成"教科书"教会 1.3B 模型写 Python，到 Phi-4 用约 **4000 亿 token 合成数据**（50 类数据集）把 14B 模型的数学/推理顶到与大模型同台竞技。

**许可证**：Phi-3 / Phi-3.5 / Phi-4 全系均为 **MIT License**（已核实，HF model cards 明确标注 "licensed under the MIT license"）——这是 Phi 相对很多"自定义社区许可"模型（如部分 Llama / Gemma 条款）的一大友好之处：商用、改、再分发几乎无限制。

### 时间线与版本表

| 时间 | 版本 | 尺寸 | 要点 | 论文 / 来源 |
|---|---|---|---|---|
| 2023-06 | **Phi-1** | 1.3B | 只做 Python 代码；"教科书数据"假说首发；HumanEval 50.6% pass@1 | [arXiv:2306.11644](https://arxiv.org/abs/2306.11644)《Textbooks Are All You Need》|
| 2023-09 | **Phi-1.5** | 1.3B | 扩到通用文本 / 常识推理（合成"教科书"为主） | [arXiv:2309.05463](https://arxiv.org/abs/2309.05463) |
| 2023-12 | **Phi-2** | 2.7B | 通用文本 / 对话；广泛流行的"小钢炮" | HF `microsoft/phi-2` |
| 2024-04 | **Phi-3**（mini/small/medium） | 3.8B / 7B / 14B | dense decoder（Llama 风格）；**LongRoPE** 把上下文扩到 **128K**；MIT | [arXiv:2404.14219](https://arxiv.org/abs/2404.14219) |
| 2024-08 | **Phi-3.5**（mini / MoE / vision） | 3.8B / 16×3.8B / 4.2B | 加入**稀疏 MoE**（16 专家 top-2，~6.6B 激活）与多模态 vision | HF `microsoft/Phi-3.5-*` |
| 2024-12 | **Phi-4** | 14B | **合成数据为主**（~400B token）、强推理、**pivotal-token DPO**；base 上下文 16K | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)《Phi-4 Technical Report》|
| 2025 | **Phi-4-mini / -multimodal / -reasoning** | 3.8B / 5.6B / 3.8B(+) | mini 上下文回到 128K；multimodal 加 vision+audio；reasoning 走"长思维链 + RL/蒸馏" | HF `microsoft/Phi-4-*` |

> 一句话定位：**OLMo 开放"过程"，Qwen 走"大小模型族 + 蒸馏"，而 Phi 把赌注全押在"数据策划"上**——它要证明的是"小模型 + 好数据"能打"大模型 + 海量脏数据"。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| GitHub（教程仓库，本 fork 对应） | https://github.com/microsoft/PhiCookBook |
| 本地 fork | `forks/PhiCookBook`（commit `0e5d477`） |
| HuggingFace 组织 | https://huggingface.co/microsoft |
| Phi-1 / 1.5 collection | https://huggingface.co/collections/microsoft/phi-1-6626e29134744e94e222d572 |
| Phi-3 collection | https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3 |
| Phi-4 collection | https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4 |
| 论文《Textbooks Are All You Need》(Phi-1) | https://arxiv.org/abs/2306.11644 |
| 论文 Phi-1.5 | https://arxiv.org/abs/2309.05463 |
| 论文 Phi-3 Technical Report | https://arxiv.org/abs/2404.14219 |
| 论文 Phi-4 Technical Report | https://arxiv.org/abs/2412.08905 |
| 架构源码（dense，本文 5/6.1 章引用） | `transformers/.../models/phi3/modeling_phi3.py` |
| 架构源码（MoE） | `transformers/.../models/phimoe/modeling_phimoe.py` |
| Microsoft Foundry 模型目录 | https://ai.azure.com/explore/models?selectedCollection=phi |

> 本地资料导航（`forks/PhiCookBook/md/`）：`01.Introduction/01/01.PhiFamily.md`（家族总览表）、`03.FineTuning/`（微调配方：LoRA / QLoRA / Olive / MLX / Vision / Kaito / AzureML）、`02.QuickStart/`、`01.Introduction/04/`（量化）。

---

## 3. 模型规格

下表"架构数字"以 **HuggingFace model cards** 为准并交叉核对论文；标注**未核实**者表示本次未在一手来源逐字确认（多为论文配置表中的层数/隐藏维度，model card 未逐字给出）。

| 模型 | 参数 | 架构 | 上下文 | 训练 token | 许可证 | 来源 |
|---|---|---|---|---|---|---|
| Phi-1 | 1.3B | dense decoder | 2K | ~7B（6B 筛网页 + 1B 合成） | MIT | [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) |
| Phi-2 | 2.7B | dense decoder | 2K | 1.4T | MIT | HF `microsoft/phi-2` |
| Phi-3-mini | **3.8B** | dense decoder（Llama 式）；**LongRoPE→128K** | 4K / **128K** | 3.3T（论文）/ model card 现标 **4.9T**（refresh） | **MIT** | HF `Phi-3-mini-128k-instruct` |
| Phi-3-small | **7B** | dense decoder；**GQA**；tiktoken 词表(100k+) | 8K / 128K | 4.8T | MIT | [arXiv:2404.14219](https://arxiv.org/abs/2404.14219) |
| Phi-3-medium | **14B** | dense decoder | 4K / 128K | 4.8T | MIT | 同上 |
| Phi-3.5-mini | 3.8B | dense decoder；LongRoPE | **128K** | 3.4T | MIT | HF `Phi-3.5-mini-instruct` |
| **Phi-3.5-MoE** | **16×3.8B（总~42B / 激活~6.6B）** | **稀疏 MoE**：16 专家、**top-2** | **128K** | 4.9T | MIT | HF `Phi-3.5-MoE-instruct` |
| Phi-3.5-vision | 4.2B | dense + 图像编码器(CLIP ViT-L/14) | 128K | — | MIT | HF `Phi-3.5-vision-instruct` |
| **Phi-4** | **14B** | dense decoder-only | **16K**（base） | **9.8T**（含 ~400B 合成） | **MIT** | HF `microsoft/phi-4`、[arXiv:2412.08905](https://arxiv.org/abs/2412.08905) |
| Phi-4-mini | 3.8B | dense decoder | **128K** | — | MIT | HF `Phi-4-mini-instruct` |
| Phi-4-multimodal | 5.6B | dense + vision + audio | 128K | — | MIT | HF `Phi-4-multimodal-instruct` |

**关键事实核对（一手来源逐字确认）**：
- Phi-3-mini = "3.8B parameters … dense decoder-only Transformer"，上下文 "128K tokens"，词表 "32064 tokens"，"licensed under the MIT license"（HF `Phi-3-mini-128k-instruct`）。**训练 token：原论文 3.3T，但当前 model card 已更新为 "4.9T tokens"（2024-05~06 重训/refresh），两者并存，引用时需注明版本。**
- Phi-3.5-MoE = "16x3.8B … 6.6B active parameters when using 2 experts"，"128K"，"4.9 trillion tokens"，512×H100-80G 训练 23 天（HF `Phi-3.5-MoE-instruct`）。
- Phi-4 = "14B parameters, dense decoder-only Transformer"，上下文 **16K**，"9.8T tokens"，"SFT + iterative DPO"，MIT（HF `microsoft/phi-4`）。**注意：很多二手资料把 Phi-4 上下文写成 128K，但官方 base model card 是 16K；128K 是 Phi-3-128k / Phi-3.5 / Phi-4-mini 的特性。**

> 词表演化：Phi-3-mini/medium 沿用 **Llama 的 32064 词表**（便于复用 Llama 生态）；Phi-3-small 改用 **tiktoken（约 10 万词表）**——这是同代内为多语言/效率做的取舍（未在 model card 逐字，源自论文，标**约**）。

---

## 4. Benchmark

下表数字**均取自官方 model card / 技术报告**，并标注 shot 设置与来源；不同来源 shot 数不同时不可直接横比，已逐项标注。

| 模型 | MMLU | GSM8K | HumanEval | MATH | MMLU-Pro | 其它 | 来源 |
|---|---|---|---|---|---|---|---|
| Phi-1 | — | — | **50.6**（pass@1）| — | — | MBPP 55.5 | [2306.11644](https://arxiv.org/abs/2306.11644) |
| Phi-3-mini-128k | **69.7**（5-shot）| 85.3（8-shot CoT）| 60.4（0-shot）| — | — | — | HF card |
| Phi-3.5-mini | 69（5-shot）| 86.2（8-shot CoT）| 62.8（0-shot）| — | — | — | HF card |
| Phi-3.5-MoE | 78.9（5-shot）| **88.7**（8-shot CoT）| 70.7（0-shot）| — | — | MBPP 80.8 / BBH 79.1 | HF card |
| **Phi-4** | **84.8** | MGSM **80.6** | **82.6** | **80.4** | **70.4** | GPQA 56.1 | HF `microsoft/phi-4` |

**读图要点**：
- **同尺寸跨代跃迁**：Phi-3.5-mini → Phi-4（都是 dense，4B vs 14B），MMLU 69 → 84.8、HumanEval 62.8 → 82.6——**主要驱动力是"合成数据 + 后训练（DPO）"，而非单纯放大参数**（Phi-4 才 14B，并未比 Phi-3-medium 大）。
- **MoE 的性价比**：Phi-3.5-MoE 总参 ~42B 但**每 token 只激活 ~6.6B**，GSM8K 88.7 甚至超过更大的 dense 模型——这是稀疏 MoE"用更多参数存知识、用更少算力推理"的典型收益。
- Phi-4 的 **MATH 80.4 / MMLU-Pro 70.4**（都是"难"基准）说明合成数据策略对**复杂推理**的提升尤其显著，而非只是刷简单题。

> ⚠️ benchmark 比较坑：Phi 团队在论文里专门讨论过"数据污染（contamination）"——小模型刷高分容易引来"是否见过测试集"的质疑，因此 Phi-4 报告中用了**去污染流程**和 **AMC/竞赛新题**等"训练截止后"的题来佐证泛化。引用 Phi 的 benchmark 时，最好同时看其去污染说明。

---

## 5. 架构剖析（基于 transformers `modeling_phi3.py`）

Phi-3 系列是一个**标准的、Llama 风格的 dense decoder-only Transformer**，但在工程上做了两处"合并投影"的优化和一处关键的"长上下文"机制（LongRoPE）。以下逐一对照 HuggingFace 实现，URL 统一为：

`https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py`

### 5.1 整体堆叠：Pre-Norm + 残差

`Phi3DecoderLayer`（`modeling_phi3.py:295`）是经典的 **Pre-Norm** 结构：先 RMSNorm 再进子层，子层输出加回残差。前向（`:307`）核心四行：

```python
# modeling_phi3.py:317-334
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)        # Pre-Norm
hidden_states, _ = self.self_attn(...)                     # 注意力
hidden_states = residual + self.resid_attn_dropout(hidden_states)  # main diff with Llama（残差 dropout）

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                    # MLP
hidden_states = residual + self.resid_mlp_dropout(hidden_states)
```

> 与 Llama 唯一可见差异：Phi-3 在两条残差路径上加了 **`resid_pdrop` dropout**（`:304-305`、`:329`、`:334`，源码注释明写 "main diff with Llama"）。其余拓扑（Pre-Norm、两子层、RMSNorm）与 Llama 一致——这也是为什么 Phi-3 能直接复用 Llama 词表和大量生态工具。

### 5.2 RMSNorm

`Phi3RMSNorm`（`modeling_phi3.py:275`）就是标准 RMSNorm（等价于 T5 LayerNorm，无均值中心化、无 bias），关键在于**在 float32 下算方差再转回原 dtype**，保证数值稳定：

```python
# modeling_phi3.py:284-289
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

### 5.3 Attention：合并 QKV（qkv_proj）+ 可选 GQA

`Phi3Attention`（`modeling_phi3.py:208`）的两大工程特征：

**(1) 单矩阵合并 Q/K/V**：不像很多实现分别建 `q_proj/k_proj/v_proj`，Phi-3 用**一个** `qkv_proj` 输出全部三者再切片，减少 kernel launch、对量化/导出更友好：

```python
# modeling_phi3.py:222-224
op_size = config.num_attention_heads * self.head_dim + 2 * (config.num_key_value_heads * self.head_dim)
self.o_proj   = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
self.qkv_proj = nn.Linear(config.hidden_size, op_size, bias=False)
```

前向里按位置切回 Q/K/V（`:237-241`）：

```python
qkv = self.qkv_proj(hidden_states)
query_pos = self.config.num_attention_heads * self.head_dim
query_states = qkv[..., :query_pos]
key_states   = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
```

**(2) GQA（Grouped-Query Attention）支持**：`op_size` 用 `num_key_value_heads` 单独计算 K/V 头数（`:222`），`num_key_value_groups = num_attention_heads // num_key_value_heads`（`:216`）。当 KV 头数 < Q 头数时即为 GQA，推理时由 `repeat_kv`（`:141`）把 KV 头复制到与 Q 对齐。
> 实际配置上：**Phi-3-mini（3.8B）KV 头 = Q 头（即 MHA，不分组）；Phi-3-small/medium、Phi-3.5-MoE 才用 GQA**（这点 model card 未逐行给，源自论文配置，标**约**）。无论哪种，同一份代码用 `num_key_value_heads` 参数即可表达。

注意力计算本身通过 `ALL_ATTENTION_FUNCTIONS` 分发（`:253`），可切 eager / SDPA / FlashAttention；缩放为标准 `head_dim**-0.5`（`:218`）。

### 5.4 MLP：合并 gate_up_proj 的 SwiGLU

`Phi3MLP`（`modeling_phi3.py:49`）是 **SwiGLU**，同样做了"合并投影"：把 SwiGLU 需要的 `gate` 和 `up` 两个投影**合并成一个** `gate_up_proj`（输出维度 `2 * intermediate_size`），前向时 `chunk` 切两半：

```python
# modeling_phi3.py:50-64
self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
self.down_proj    = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
self.activation_fn = ACT2FN[config.hidden_act]          # SiLU
...
up_states = self.gate_up_proj(hidden_states)
gate, up_states = up_states.chunk(2, dim=-1)            # 切成 gate / up
up_states = up_states * self.activation_fn(gate)        # SwiGLU: up * SiLU(gate)
return self.down_proj(up_states)
```

> "合并 QKV + 合并 gate_up" 是 Phi-3 在工程层面相对裸 Llama 的两处优化：**更少的算子、对 ONNX/量化导出更顺**——这与 Phi 主打"边缘/端侧部署"（见 cookbook 大量 ONNX/Olive/MLX/GGUF 内容）的定位高度一致。

### 5.5 LongRoPE：把 4K 训练的模型扩到 128K

Phi-3 的招牌长上下文技术是 **LongRoPE**（微软自研，[arXiv:2402.13753](https://arxiv.org/abs/2402.13753)）。它的思想：RoPE 的不同频率维度对"外推"的敏感度不同，**与其全局放缩，不如给每个频率维度学一组"缩放因子"**，并区分**短序列因子（short_factor）**与**长序列因子（long_factor）**。

在 `modeling_phi3.py` 里，`Phi3RotaryEmbedding`（`:67`）本身只负责**分发**：根据 `config.rope_parameters["rope_type"]` 从 `ROPE_INIT_FUNCTIONS` 里取对应初始化函数（`:77-81`）：

```python
# modeling_phi3.py:77-81
self.rope_type = self.config.rope_parameters["rope_type"]   # 对 128K 模型即 "longrope"
rope_init_fn = self.compute_default_rope_parameters
if self.rope_type != "default":
    rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]      # → _compute_longrope_parameters
inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
```

真正的 LongRoPE 逻辑在 `transformers/src/transformers/modeling_rope_utils.py` 的 **`_compute_longrope_parameters`（约 `:614`）**，URL：
`https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py`

三处关键：

```python
# modeling_rope_utils.py:~634-635  —— 从 config 读两组因子
long_factor  = rope_parameters_dict["long_factor"]
short_factor = rope_parameters_dict["short_factor"]

# modeling_rope_utils.py:~650-653  —— 按当前序列长度选短/长因子
if seq_len and seq_len > original_max_position_embeddings:
    ext_factors = torch.tensor(long_factor, ...)    # 超过原始训练长度 → 用 long_factor
else:
    ext_factors = torch.tensor(short_factor, ...)   # 短序列 → 用 short_factor（几乎不动）

# modeling_rope_utils.py:~654-655  —— 用逐维因子调制 inv_freq
inv_freq_shape = torch.arange(0, dim, 2, ...).float() / dim
inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

# modeling_rope_utils.py:~642-645  —— attention_factor：长上下文时给 logits 一个幅度补偿
attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))
```

要点解读：
- **逐维缩放向量**（`ext_factors` 是长度 = `dim/2` 的向量，不是标量），这正是 LongRoPE 区别于 linear / NTK / YaRN 全局放缩的核心；
- **短/长双因子切换**：保证"短序列性能不退化、长序列才启用外推"——这是 Phi-3-128k 在 4K 内仍保持原精度的原因；
- **attention_factor**（即 `Phi3RotaryEmbedding.forward` 里乘到 cos/sin 上的 `attention_scaling`，`:128-129`）补偿长上下文下 attention logits 的分布漂移。

> 历史命名：早期 Phi-3 config 里这种 RoPE 叫 `"su"`（short/longrope 的早期名），现 transformers 主线已统一为 `"longrope"`（`ROPE_INIT_FUNCTIONS` 的 key，`modeling_rope_utils.py:~768`）。

### 5.6 Phi-3.5-MoE：稀疏 MoE（基于 `modeling_phimoe.py`）

Phi-3.5-MoE 把每层 MLP 换成 **稀疏专家混合（Sparse MoE）**。源码 URL：
`https://github.com/huggingface/transformers/blob/main/src/transformers/models/phimoe/modeling_phimoe.py`

**(1) 专家网络** `PhimoeExperts`（约 `:592`）：每个专家就是一个 SwiGLU MLP，但权重以**三维 Parameter** 批量存放（`num_experts` 维），同样合并 gate/up：

```python
# modeling_phimoe.py:~598-600
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
self.down_proj    = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
self.act_fn = ACT2FN[config.hidden_act]
# forward(~:609-611): chunk 出 gate/up → act_fn(gate) * up → down_proj
```

**(2) 路由 + 稀疏块** `PhimoeSparseMoeBlock`（约 `:651`）：用一个线性 `router` 把每个 token 打分到 16 个专家，选 **top-2**（`top_k = num_experts_per_tok`），训练时还可加 **jitter noise**（输入抖动，提升路由鲁棒性）：

```python
# modeling_phimoe.py:~617-619
self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)  # 打分到 16 专家
self.top_k  = config.num_experts_per_tok                                            # = 2
# forward(~:676-684): router → softmax → 选 top-2 专家 → 仅这两个专家参与计算
```

**(3) 负载均衡辅助损失** `load_balancing_loss_func`（约 `:759`）：MoE 训练的关键正则，防止"少数专家被挤爆、多数专家闲置"。它对路由概率做 softmax、取 top-k，计算"每个专家被分到的 token 占比 × 平均路由概率"，乘以专家数作为惩罚：

```python
# modeling_phimoe.py:~785-788, ~813
routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
_, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
...
overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))   # × num_experts
```

> 设计取舍：Phi-3.5-MoE 用 **16 专家 / top-2 / ~6.6B 激活** 实现"~42B 的知识容量、~7B 的推理算力"。对比 [[03-deepseek]] 的 DeepSeek-V3（256 路由专家 + 共享专家、细粒度 MoE），Phi 的 MoE 更"经典/保守"（无共享专家、专家数少），符合它"能在受限硬件上跑"的产品取向。

---

## 6. 源码走读

### 6.1 架构走读（transformers）

把第 5 章串成"一次 forward 走一遍"，行号均指 `modeling_phi3.py`：

1. **入口** `Phi3ForCausalLM.forward`（`:449`）→ 调 `Phi3Model.forward`（`:379`）。
2. `Phi3Model`（`:359`）持有：`embed_tokens`、`layers`（`Phi3DecoderLayer × N`）、`norm`（最终 RMSNorm）、`rotary_emb`（`Phi3RotaryEmbedding`，全局算一次 cos/sin 供各层复用）。
3. 每层 `Phi3DecoderLayer`（`:295`）按 5.1 的 Pre-Norm 顺序：`input_layernorm` → `self_attn`（合并 QKV，`:208`）→ 残差 → `post_attention_layernorm` → `mlp`（合并 gate_up SwiGLU，`:49`）→ 残差。
4. 注意力内部：`qkv_proj` 一把出 Q/K/V 并切片（`:237-241`）→ `apply_rotary_pos_emb`（`:178`，把 LongRoPE 的 cos/sin 旋进 Q/K）→ `repeat_kv` 展开 GQA（`:141`）→ `ALL_ATTENTION_FUNCTIONS` 分发的注意力（`:253`）→ `o_proj`。
5. 末层 `norm` 后接 `lm_head`（`nn.Linear(hidden, vocab)`，`:433` 内）产出 logits。

> 学习提示：要把"配置→形状"对上号，重点看三个 `config` 字段——`num_attention_heads` / `num_key_value_heads`（决定是否 GQA）、`intermediate_size`（决定 MLP 宽度）、`rope_parameters`（决定是否 LongRoPE 及 short/long factor）。改这三个就能从 Phi-3-mini 配出 small/medium。

### 6.2 微调配方走读（本地 PhiCookBook）

cookbook 的 `md/03.FineTuning/` 提供了一整套从"参数高效微调（PEFT）"到"端侧优化"的配方。下面走读其中**最完整、可直接照跑**的一条——`olive-lab`（用微软 **Olive** 工具链把 Phi-3.5-mini 微调成"旅游问答"专家并导出 ONNX）。引用：`forks/PhiCookBook/md/03.FineTuning/olive-lab/readme.md`。

这条配方的精髓是顺序：**先量化、再微调、后图优化**（`olive-lab/readme.md:134` 明确解释"量化后再微调，微调能恢复量化损失，精度更好"）：

```bash
# ① AWQ 量化（activation-aware，~8min，7.5GB→2.5GB）   readme.md:122-128
olive quantize \
   --model_name_or_path microsoft/Phi-3.5-mini-instruct \
   --algorithm awq --output_path models/phi/awq

# ② LoRA 微调（在量化模型上，max_steps=100，~6min）     readme.md:137-146
olive finetune \
    --method lora \
    --model_name_or_path models/phi/awq \
    --data_files "data/data_sample_travel.jsonl" --data_name "json" \
    --text_template "<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>" \
    --max_steps 100 --output_path ./models/phi/ft

# ③ auto-opt 抓 ONNX 图 + 融合/压缩，产出可端侧跑的模型   readme.md:153-160
olive auto-opt \
   --model_name_or_path models/phi/ft/model --adapter_path models/phi/ft/adapter \
   --device cpu --provider CPUExecutionProvider --use_ort_genai \
   --output_path models/phi/onnx-ao
```

走读要点：
- **Chat 模板** `<|user|>\n...<|end|>\n<|assistant|>\n...`（`readme.md:142`）是 Phi-3/3.5 的官方对话格式（特殊 token `<|user|>` / `<|assistant|>` / `<|end|>`）——微调数据必须按此拼接，否则推理时模板不匹配。
- **AWQ**（[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)，`readme.md:119`）按激活分布选择性保护重要权重，比朴素权重量化更保精度。
- 产物是 **LoRA adapter**（`adapter_weights.onnx_adapter`），推理时 `set_active_adapter`（`readme.md:189`）热插拔——天然支持 **multi-LoRA 服务**（一份基座 + 多个任务 adapter）。
- 硬件门槛：`readme.md:6` 要求 **A10/A100 + CUDA 12+**；整个流程 ~35min。

**其余配方（同目录，速览其超参/工具取向）**：

| 配方文件 | 方法 | 关键点 |
|---|---|---|
| `FineTuning_Lora.md` | LoRA（`loralib` / `peft`） | rank **r=16**（`:38`），只标记 `lora_` 参数可训（`:47`），`bfloat16` 混合精度（`:82`），基座 `microsoft/Phi-3-mini-4k-instruct`（`:82`） |
| `FineTuning_Qlora.md` | QLoRA（4-bit + LoRA） | `bitsandbytes` 4-bit 加载（`:7`），需从源码装 `accelerate`/`transformers`，显存最省 |
| `FineTuning_Scenarios.md` | 方法对照表 | LoRA/QLoRA/PEFT/DoRA（参数高效）vs DeepSpeed/ZeRO（分布式/省显存）的适用矩阵（`:36-46`）；Phi-3/3.5/4 已支持**Serverless 微调**（`:24`） |
| `FineTuning_MLX.md` / `03.Inference/MLX_Inference.md` | Apple MLX | 在 Apple Silicon 上微调/推理（端侧路线） |
| `FineTuning_Vision.md` / `FineTuning_Phi-3-visionWandB.md` | 视觉微调 | Phi-3-vision + W&B 跟踪 |
| `FineTuning_Kaito.md` / `FineTuning_AIFoundry.md` / `FineTuning_MLSDK.md` | K8s(Kaito) / Azure AI | 云上托管微调 |

> 注：`FineTuning_Lora.md` / `FineTuning_Qlora.md` 正文主要给"步骤说明"，真正的 Python 脚本（`code/03.Finetuning/*.ipynb`、`FineTrainingScript.py`）和大图片在本 fork 中已被 **sparse-checkout 过滤**（cookbook README `:39-46` 解释了 sparse 克隆）；故本文以这些 md 的**步骤与超参文字**为准，并以 `olive-lab` 这条"自带完整脚本"的配方作为主走读对象。LoRA 的官方完整脚本可见 HF model card 的 `sample_finetune.py`（`FineTuning_Lora.md:94` 给了链接）。

---

## 7. 训练机制：数据策划（核心）+ SFT + DPO

Phi 的训练"秘方"几乎全在**数据**，模型架构反而是最不特别的部分。这正是 Phi 和 [[08-olmo]]（开放全过程）、[[02-qwen]]（蒸馏 + 大小模型族）路线的根本分野。

### 7.1 数据策划流水线（"Textbooks Are All You Need"）

这是 Phi 全系的**第一性原理**，三代一脉相承：

**① Phi-1（2306.11644）：假说的起点。** 只做一件事——Python 代码。数据只有约 **7B token**：**6B 从网页里筛出的"教科书质量"代码 + 1B 用 GPT-3.5 合成的"教科书与练习"**。1.3B 的模型就拿到 **HumanEval 50.6% pass@1**，逼平当时大得多的模型。结论：**"数据质量"可以换"参数规模"**。

> 筛选用了一个巧思：训练一个**小分类器**来判断网页代码是否"具有教育价值（educational value）"，只留高价值样本——这比简单去重/规则过滤更接近"人类老师挑教材"。

**② Phi-1.5 / Phi-3：把配方推广到通用语言。** Phi-3 论文把数据策略概括为两阶段、按"教育水平"分级：
- 重度筛选的**公开网页数据**（按"推理/教育价值"打分过滤）；
- 大量 **LLM 合成数据**（合成"教科书式"讲解、习题、对话），覆盖数学/代码/常识/逻辑。
- 论文称之为 **"data optimal regime"**：在固定算力下，不是"喂更多 token"，而是"喂对模型当前能力最有增益的 token"。

**③ Phi-4（2412.08905）：合成数据成为主角。** 这是质变——Phi-4 用了约 **400B token 的合成数据**，跨 **50 类**精心设计的数据集（已核实，技术报告）。合成手段包括：多智能体提示、自我修订（self-revision）、指令反转（从代码反推指令）、把高质量代码/证明改写成问答等。**网页真实数据从"主菜"退为"调味"**，主要用于补充世界知识与多样性。后训练阶段还加了 **40 种语言**的多语言数据。

> 一句话：**Phi-1 证明"好数据 > 大规模"，Phi-4 把这条路线工业化到"主要用合成数据训前沿小模型"。**

### 7.2 后训练：SFT + DPO（含 Phi-4 的 pivotal-token DPO）

Phi-4 的对齐管线（HF card 与技术报告核实）是 **SFT → 迭代 DPO（iterative DPO）**：

- **SFT**：在高质量指令/对话数据上做监督微调，含 40 语言多语料。
- **DPO（Direct Preference Optimization）**：用偏好对 `(chosen, rejected)` 直接优化，无需独立奖励模型。Phi-4 做了**多轮**。
- **Pivotal Token Search（PTS，关键创新）**：技术报告观察到——一条回答的对错，往往由**少数几个"关键 token（pivotal token）"**决定（比如解题时选错一个公式/一步运算）。PTS 用**预训练模型自己的采样**去定位这些关键 token，构造**token 级**的偏好对喂给**第一轮 DPO**，从而把偏好信号精确打到"真正决定成败的那几步"，而不是对整条回答笼统打分。这显著提升了数学/推理的稳定性。

> 对照：[[03-deepseek]] 用大规模 **RL（GRPO）** 逼出长思维链；Phi-4 走的是**更省算力的偏好优化（DPO + PTS）**路线——小团队/小模型也能做"对齐到推理"。后续 **Phi-4-reasoning** 才进一步引入"长思维链 + 蒸馏/RL"专攻推理（与 DeepSeek-R1 路线靠拢）。

### 7.3 安全与"负责任 AI"

cookbook 单列了 `01.Introduction/01/01.AISafety.md` 与 `05/ResponsibleAI.md`：Phi 的后训练含**安全对齐**（red-teaming、有害内容拒答），PTS 的偏好数据里也专门有"未知话题 / 安全"类别（技术报告）——这对"要落到边缘设备、脱离云端审核"的 SLM 尤为重要。

---

## 8. 学习要点与横向对比

### 8.1 Phi 教会我们什么

1. **数据为王（data-centric）是 Phi 的灵魂。** 在固定/受限算力下，**提升数据质量的边际收益 > 增加参数**。"教科书级合成数据 + 重度筛选网页"是一条已被三代验证的可复制路线。这是本文最该带走的一点。
2. **小模型策略 = 数据 + 后训练 + 工程合并。** Phi-4 才 14B 却全面超过自家 14B 的 Phi-3-medium——差距来自数据与 DPO，不是规模。架构上的"合并 QKV / 合并 gate_up / LongRoPE / 可选 GQA"都服务于"在小硬件上又快又省"。
3. **端侧落地是产品锚点。** cookbook 的重心（ONNX / Olive / MLX / GGUF / int4 量化 / multi-LoRA）说明 Phi 的目标场景是**边缘/端侧**，这反过来塑造了它的架构取舍（合并投影利于导出、MIT 许可利于嵌入产品）。
4. **token 级对齐（PTS）** 是个轻量但有效的 trick：把偏好信号打到"决定成败的关键 token"，比整段打分更高效——值得在自己的 DPO 流程里借鉴。

### 8.2 横向对比

| 维度 | **Phi**（本文） | [[08-olmo]] OLMo | [[02-qwen]] Qwen | [[03-deepseek]] DeepSeek |
|---|---|---|---|---|
| 核心理念 | **数据为王**：合成"教科书"+ 重筛网页，让小模型对标大模型 | **完全开放**：数据/代码/checkpoint/日志全公开，复现"过程" | **大小模型族 + 蒸馏**：大模型蒸馏小模型，全尺寸覆盖 | **推理 + RL**：用大规模 RL(GRPO) 逼出长思维链 |
| 数据透明度 | 配方公开、**数据集本身不开**（合成 pipeline 描述 ≠ 放数据） | **数据完全开放**（Dolma/Dolmino 可下载） | 部分公开 | 部分公开 |
| 主力尺寸 | **SLM 1.3B–14B**（端侧） | 1B–32B | 0.5B–235B（含 MoE） | 671B(MoE) 等大模型 |
| 后训练 | SFT + **DPO + PTS（token 级）** | SFT + DPO/RLVR（Tülu 3 配方开源） | SFT + DPO/RL + 蒸馏 | 大规模 **RL（GRPO）** + 蒸馏 |
| 架构特别处 | 合并 QKV/gate_up、**LongRoPE**、Phi-3.5-MoE | 稳定性改造（QK-norm/reordered-norm/Z-loss） | GQA、部分 MoE、YaRN | 细粒度 MoE + MLA + 共享专家 |
| 许可证 | **MIT**（最宽松） | Apache-2.0 | Apache-2.0（多数） | MIT / 自定义 |

**和"数据全开"的 OLMo 比**：两者都把"数据"当核心，但方向相反——**OLMo 开放数据让你复现过程；Phi 不开放数据，而是公开"如何造好数据"的方法论**。学 OLMo 看的是 `train.py` 怎么写（过程），学 Phi 看的是"数据怎么策划"（配方）。

**和"蒸馏"的 Qwen 比**：Qwen 用**大模型蒸馏**喂小模型（教师→学生）；Phi 用**合成数据**（让 GPT-3.5/4 造"教科书"）——本质都是"用强模型的知识去教小模型"，但 Phi 更强调"造教材"而非"对齐 logits/输出"。两条路线可互补。

---

## 9. 参考

**一手来源（已核实）**
- Phi-1《Textbooks Are All You Need》：https://arxiv.org/abs/2306.11644 （1.3B / ~7B token / HumanEval 50.6% pass@1）
- Phi-1.5：https://arxiv.org/abs/2309.05463
- Phi-3 Technical Report：https://arxiv.org/abs/2404.14219
- Phi-4 Technical Report：https://arxiv.org/abs/2412.08905 （14B / 9.8T token / ~400B 合成 / SFT+iterative DPO / Pivotal Token Search）
- LongRoPE：https://arxiv.org/abs/2402.13753
- AWQ 量化：https://arxiv.org/abs/2306.00978
- HF model cards：`microsoft/phi-4`、`Phi-3-mini-128k-instruct`、`Phi-3.5-mini-instruct`、`Phi-3.5-MoE-instruct`（参数 / 上下文 / 训练 token / MIT 许可 / benchmark 均逐字取自对应卡片）

**架构源码（本文 5、6.1 章引用，行号对 `main` 分支）**
- `https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py` —— `Phi3MLP:49`、`Phi3RotaryEmbedding:67`、`Phi3Attention:208`、`Phi3RMSNorm:275`、`Phi3DecoderLayer:295`、`Phi3Model:359`、`Phi3ForCausalLM:433`
- `https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py` —— `_compute_longrope_parameters:~614`、`ROPE_INIT_FUNCTIONS:~768`
- `https://github.com/huggingface/transformers/blob/main/src/transformers/models/phimoe/modeling_phimoe.py` —— `PhimoeExperts:~592`、`PhimoeSparseMoeBlock:~651`、`load_balancing_loss_func:~759`

**本地教程 fork（本文 6.2 章引用）**
- `forks/PhiCookBook/md/01.Introduction/01/01.PhiFamily.md` —— 家族总览表
- `forks/PhiCookBook/md/03.FineTuning/olive-lab/readme.md` —— AWQ 量化 + LoRA 微调 + ONNX 优化完整配方
- `forks/PhiCookBook/md/03.FineTuning/FineTuning_Lora.md` / `FineTuning_Qlora.md` / `FineTuning_Scenarios.md` —— LoRA/QLoRA 步骤与超参、方法对照
- fork 元数据：commit `0e5d477`（2026-05-26），`md/` 82 个 Markdown / ~1.35 万行

> **诚实声明**：标"约/未核实"处表示本次未在一手来源逐字确认（多为论文配置表中的层数/隐藏维度/头数，model card 未逐字给出）。架构源码引用的是 transformers `main` 分支当前版本，行号可能随上游重构漂移，引用时以 commit 锚定为宜。
