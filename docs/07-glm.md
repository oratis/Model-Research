# GLM / ChatGLM 深度分析（autoregressive blank infilling → MoE 355B-A32B / partial RoPE / QK-Norm / MTP / slime RL）

> 本文档为 Model-Research 学习仓库的 **GLM / ChatGLM（智谱 Zhipu AI / Z.ai，清华 KEG & THUDM 谱系）** 专题。
> 本地走读基于 fork submodule `forks/GLM-4.5`（官方仓库 `zai-org/GLM-4.5`，主要是 README + 推理脚本 + 工具推理指南；权重不在仓库内）。
> **架构源码**章节因本仓库不含建模代码，统一引用 HuggingFace `transformers` 主分支的 `glm4_moe`（GLM-4.5/4.6/4.7）与 `glm4`（GLM-4-9B 类 dense）建模文件，给出 **URL + 行号**。
> 规格 / benchmark 数字尽量给官方来源（HF model card、arXiv 技术报告、官方 README）；无法逐一核实者标注「**约 / 未核实**」。
> 撰写日期：2026-06-06。fork 内 README 已演进到 GLM-4.7 时代（同仓库覆盖 4.5/4.6/4.7），本文以 **GLM-4.5（技术报告锚点）/ 4.6（200K 长上下文）** 为主线，4.7/Flash 作为时间线补充。

---

## 1. 概览（Overview）

**出品方**：智谱 AI（Zhipu AI / 北京智谱华章），海外品牌 **Z.ai**；技术根脉来自 **清华大学 KEG 实验室 / THUDM**（Tsinghua Knowledge Engineering Group）。HuggingFace 组织已从 `THUDM` 迁移/并行到 **`zai-org`**。

**定位**：GLM 一词最初指 **General Language Model**——一种用 *autoregressive blank infilling*（自回归空白填充）统一「双向理解（NLU）+ 单向生成（NLG）」的预训练范式（区别于纯 encoder 的 BERT、纯 decoder 的 GPT、encoder-decoder 的 T5）。发展到 **GLM-4.5 系列**，定位转为「**面向智能体（agent-native）的基础模型**」：用 MoE 把总参数与单 token 计算解耦（355B 总参 / 32B 激活），把 **推理（Reasoning）+ 编码（Coding）+ 智能体（Agentic）** 三类能力统一进一份混合推理（hybrid reasoning）权重。技术报告标题即 **「GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models」**（[arXiv 2508.06471](https://arxiv.org/abs/2508.06471)）。

**许可证**：GLM-4.5 / 4.5-Air（含 Base、FP8）以 **MIT** 许可证发布，可商用、可二次开发、可蒸馏（fork `README.md:69`）。这与「开放权重 + 宽松许可」的开源前沿阵营（DeepSeek、Qwen）一致。

**时间线（谱系，sourced）**：

| 时间 | 里程碑 | 关键点 | 来源 |
|---|---|---|---|
| 2021-03 | **原始 GLM** | *Autoregressive Blank Infilling*；2D 位置编码；统一双向+单向 | [arXiv 2103.10360](https://arxiv.org/abs/2103.10360)（ACL 2022） |
| 2022-10 | **GLM-130B** | 130B 中英双语；训练稳定性（embedding gradient shrink、DeepNorm）；INT4 单机推理 | [arXiv 2210.02414](https://arxiv.org/abs/2210.02414)（ICLR 2023） |
| 2023-03 | **ChatGLM-6B** | 首个广泛流行的中文对话开源模型；消费级显卡可跑（INT4） | [arXiv 2406.12793](https://arxiv.org/abs/2406.12793) |
| 2023-06 | **ChatGLM2-6B** | 上下文 8K→32K；FlashAttention；推理提速 | 同上 survey |
| 2023-10 | **ChatGLM3-6B** | 新 prompt 格式 + 工具调用 / 代码解释器（Agent 雏形） | 同上 survey |
| 2024-06 | **GLM-4 / GLM-4-9B** | ~10T tokens 预训练；对齐 SFT+RLHF；长上下文 128K/1M；**GLM-4 All Tools**（自主选工具：浏览器/Python/文生图） | [arXiv 2406.12793](https://arxiv.org/abs/2406.12793) |
| 2025-04 | **GLM-4-0414 系列** | GLM-4-9B-0414 / GLM-Z1（推理）等，dense 架构刷新 | [HF zai-org](https://huggingface.co/zai-org) |
| 2025-07 | **GLM-4.5 / 4.5-Air** | **MoE 355B-A32B / 106B-A12B**；混合推理；MIT；slime RL；23T tokens | [arXiv 2508.06471](https://arxiv.org/abs/2508.06471) |
| 2025-09 | **GLM-4.6** | 上下文 128K→**200K**；编码/推理/agent 增强；推理中工具调用 | fork `README.md:46-56` |
| 2025-12+ | **GLM-4.7 / 4.7-Flash** | 交错思考强化 + 保留思考 + 轮次级思考；Flash 为 30B-A3B | fork `README.md:20-44` |

> 说明：fork 仓库 `forks/GLM-4.5` 的 `README.md` 已更新到「GLM-4.7 & GLM-4.6 & GLM-4.5」三代合一的状态（见 fork `README.md:1`），因此本文档时间线把 4.7/Flash 也纳入；但 **唯一对应公开技术报告的锚点版本是 GLM-4.5**（arXiv 2508.06471），4.6/4.7 以官方 README/博客为准。GLM-5（2026 初）属本文档撰写时点之后的迭代，仅在横向对比中略提，不展开。

---

## 2. 关键链接（Key Links）

| 资源 | 链接 |
|---|---|
| 官方 GitHub（GLM-4.5，含推理脚本/TIR 指南） | https://github.com/zai-org/GLM-4.5 |
| 本仓库 fork | `forks/GLM-4.5`（`README.md` / `README_zh.md` / `inference/` / `resources/glm_4.6_tir_guide.md`） |
| HuggingFace 组织（原 THUDM） | https://huggingface.co/zai-org |
| HF: GLM-4.5 / GLM-4.5-Air | https://huggingface.co/zai-org/GLM-4.5 ・ https://huggingface.co/zai-org/GLM-4.5-Air |
| HF: GLM-4.6 / GLM-4-9B-0414 | https://huggingface.co/zai-org/GLM-4.6 ・ https://huggingface.co/zai-org/GLM-4-9B-0414 |
| RL 框架 **slime** | https://github.com/THUDM/slime ・ 文档 https://thudm.github.io/slime/ |
| transformers 建模（MoE，4.5/4.6/4.7） | https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm4_moe |
| transformers 建模（dense，GLM-4-9B 类） | https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm4 |
| vLLM MTP 实现 | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/glm4_moe_mtp.py |
| SGLang 实现 | https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/glm4_moe.py |
| 论文：GLM-4.5（ARC Foundation Models） | [arXiv 2508.06471](https://arxiv.org/abs/2508.06471) |
| 论文：原始 GLM（blank infilling） | [arXiv 2103.10360](https://arxiv.org/abs/2103.10360) |
| 论文：GLM-130B | [arXiv 2210.02414](https://arxiv.org/abs/2210.02414) |
| 论文：ChatGLM（130B→GLM-4 All Tools survey） | [arXiv 2406.12793](https://arxiv.org/abs/2406.12793) |

> 本仓库主体（fork）只含 **文档 + 推理脚本 + 工具推理模板**，不含建模源码与权重。因此第 5 / 6.1 章的「逐行架构走读」一律以 transformers 主分支建模文件为准（URL + 行号），第 6.2 章走读本地 `forks/GLM-4.5/inference/` 与 TIR 指南。

---

## 3. 模型规格（Specs）

下表以 **HuggingFace model card 的 `config.json`** 为准（即生产权重的实际超参），与 README / 技术报告交叉验证。GLM-4.5 / 4.6 共用 `model_type = glm4_moe`。

### 3.1 GLM-4.5 / 4.5-Air / 4.6（MoE 主线）

| 维度 | GLM-4.5 | GLM-4.5-Air | GLM-4.6 | 来源 |
|---|---|---|---|---|
| 总参数 / 激活 | **355B / 32B** | **106B / 12B** | 355B / 32B | fork `README.md:60-62`；HF config |
| `num_hidden_layers`（层数） | **92** | 46 | 92 | HF `config.json` |
| `hidden_size`（隐藏维） | 5120 | 4096 | 5120 | HF `config.json` |
| `num_attention_heads` | **96** | 96 | 96 | HF `config.json` |
| `num_key_value_heads`（GQA KV 头） | **8** | 8 | 8 | HF `config.json` |
| `head_dim` | 128 | 128 | 128 | HF `config.json` |
| `partial_rotary_factor`（部分 RoPE 比例） | **0.5** | 0.5 | 0.5 | HF `config.json` |
| `use_qk_norm`（QK-Norm） | **true** | **false** | **true** | HF `config.json` |
| `n_routed_experts`（路由专家数） | **160** | 128 | 160 | HF `config.json` |
| `n_shared_experts`（共享专家数） | 1 | 1 | 1 | HF `config.json` |
| `num_experts_per_tok`（每 token 激活专家） | 8 | 8 | 8 | HF `config.json` |
| `moe_intermediate_size`（专家中间维） | 1536 | 1408 | 1536 | HF `config.json` |
| `intermediate_size`（dense 层中间维） | 12288 | — | 12288 | HF `config.json` |
| `first_k_dense_replace`（前 N 层为 dense） | **3** | 1 | 3 | HF `config.json` |
| `n_group` / `topk_group`（专家分组路由） | 1 / 1 | 1 / 1 | 1 / 1 | HF `config.json` |
| `routed_scaling_factor` | 2.5 | — | — | HF `config.json` |
| `norm_topk_prob` | true | — | — | HF `config.json` |
| `num_nextn_predict_layers`（**MTP** 层数） | **1** | 1 | 1 | HF `config.json` |
| `max_position_embeddings`（上下文） | **131072 ≈ 128K** | 131072 | **202752 ≈ 200K** | HF `config.json`；fork `README.md:50` |
| `vocab_size` | 151552 | 151552 | 151552 | HF `config.json` |
| `rope_theta` | 1,000,000 | — | — | HF `config.json` |
| 精度版本 | BF16 / FP8 | BF16 / FP8 | BF16 / FP8 | fork `README.md:80-92` |
| 许可证 | MIT | MIT | MIT | fork `README.md:69` |

**读数要点**：
- **deep & thin（深而窄）**：92 层、`hidden_size` 仅 5120——相比 DeepSeek-V3（61 层 / 7168 维）明显「更深更窄」。这是技术报告强调的刻意取舍（见 §5.5）。
- **`n_group=1, topk_group=1`**：GLM-4.5 的 MoE 路由代码沿用了 DeepSeek 风格的「分组受限路由」结构，但配置上 **退化为单组**（即不做跨组限制），等价于在全部 160 个路由专家上做 sigmoid 打分 + top-8。
- **Air 不开 QK-Norm**（`use_qk_norm=false`），主线 355B 与 4.6 开启——说明 QK-Norm 是按模型规模选择性启用的稳定性手段。
- **MTP**：`num_nextn_predict_layers=1`，即配 1 个多 token 预测层用于投机解码（见 §5.4）。

### 3.2 GLM-4-9B-0414（dense 对照）

| 维度 | GLM-4-9B-0414 | 来源 |
|---|---|---|
| 架构 | `Glm4ForCausalLM`（dense） | HF `config.json` |
| `num_hidden_layers` | 40 | HF `config.json` |
| `hidden_size` | 4096 | HF `config.json` |
| `num_attention_heads` / `num_key_value_heads` | 32 / 2（GQA） | HF `config.json` |
| `head_dim` | 128 | HF `config.json` |
| `intermediate_size` | 13696 | HF `config.json` |
| `partial_rotary_factor` | 0.5 | HF `config.json` |
| `max_position_embeddings` | 32768（更早的 GLM-4-9B 支持 128K/1M 变体） | HF `config.json` / [arXiv 2406.12793](https://arxiv.org/abs/2406.12793) |
| `vocab_size` | 151552 | HF `config.json` |

> GLM-4-9B-0414 走 `glm4`（dense）建模路径，其 decoder 层带 **post-self-attn / post-mlp 额外 LayerNorm（sandwich norm）**——见 §5.6 与 §6.1。

### 3.3 推理资源（fork README 实测口径）

按 fork `README.md:97-130`（条件：开 MTP、batch≤8、原生 FP8、>1T 内存）：

| 模型 | 精度 | GPU（全功能） | GPU（满 128K 上下文） |
|---|---|---|---|
| GLM-4.5 | BF16 / FP8 | H100×16 / H100×8 | H100×32 / H100×16 |
| GLM-4.5-Air | BF16 / FP8 | H100×4 / H100×2 | H100×8 / H100×4 |
| GLM-4.7-Flash | BF16 | H100×1 | H100×2 |

---

## 4. Benchmark

> 数字来源：GLM-4.5 技术报告（[arXiv 2508.06471](https://arxiv.org/abs/2508.06471)）/ 官方博客 / HF model card。GLM-4.5 在 12 项行业基准的综合得分 **63.2**（在当时所有开/闭源模型中**排名第 3**，agentic 子项排名第 2）；GLM-4.5-Air 综合 **59.8**（fork `README.md:72-74`、arXiv 摘要）。下列单项分数取自技术报告/官方榜单，**部分为单点核实，标注「约/未核实」者请以技术报告原表为准**。

### 4.1 GLM-4.5 / 4.5-Air 单项（官方）

| Benchmark（类别） | GLM-4.5 | GLM-4.5-Air | 来源/备注 |
|---|---|---|---|
| MMLU-Pro（知识/推理） | 84.6 | 81.4 | 技术报告 / HF card |
| GPQA（研究生级科学） | 79.1 | 约 75（未核实精确值） | 技术报告 |
| AIME 2024（竞赛数学） | 91.0 | 89.4 | arXiv 摘要 / 技术报告 |
| MATH-500 | 98.2 | 98.1 | 技术报告 |
| LiveCodeBench（2407–2501，代码） | 72.9 | 约 70（未核实） | 技术报告 |
| SWE-bench Verified（真实仓库修 bug） | 64.2 | 57.6 | arXiv 摘要 / 技术报告 |
| Terminal-Bench（终端 agent） | 37.5 | 30.0 | 技术报告 |
| TAU-bench（工具 agent，retail） | 79.7 / 综合 70.1 | 77.9 | 技术报告（TAU-bench 综合 70.1 见 arXiv 摘要） |
| BFCL-v3（函数调用） | 77.8（榜首区间，未核实精确值） | 76.4 | 技术报告 |
| BrowseComp（网页浏览 agent） | 26.4 | 约（未核实） | 技术报告 |
| HLE（Humanity's Last Exam，带工具） | 约 8–14（口径不一，未核实） | — | HF card 文本给 8.32（疑为某子设定） |

### 4.2 后续版本相对增量（官方 README，相对值）

GLM-4.6 相对 4.5：8 项公开基准全面提升，并在对比 **DeepSeek-V3.1-Terminus、Claude Sonnet 4** 时有竞争优势（fork `README.md:56`）。
GLM-4.7 相对 4.6（fork `README.md:24-27`）：

| Benchmark | GLM-4.7 | Δ vs 4.6 |
|---|---|---|
| SWE-bench | 73.8% | +5.8 |
| SWE-bench Multilingual | 66.7% | +12.9 |
| Terminal-Bench 2.0 | 41% | +16.5 |
| HLE（人类终极考试） | 42.8% | +12.4 |

> 解读：GLM 系列的 benchmark 叙事高度偏向 **agentic / coding / tool-use**（SWE-bench、Terminal-Bench、TAU-bench、BFCL、BrowseComp），这与「agent-native 基础模型」的产品定位一致；纯知识/数学（MMLU-Pro、AIME、MATH）作为基础能力背书。具体数字会随评测设定（thinking/non-thinking、是否带工具）变化，引用时务必带版本与设定。

---

## 5. 架构剖析（Architecture）

GLM-4.5/4.6/4.7 在 transformers 中实现为 **`glm4_moe`**；GLM-4-9B 类 dense 模型实现为 **`glm4`**。以下逐点引建模源码 URL + 行号（transformers `main` 分支，撰写日核对）。

源码主文件：
- MoE：`https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py`
- dense：`https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4/modeling_glm4.py`

### 5.1 Attention 总览（GQA + partial RoPE + 可选 QK-Norm）

`Glm4MoeAttention`（`modeling_glm4_moe.py:193-269`）的结构要点：
- **GQA（grouped-query attention）**：`num_key_value_groups = num_attention_heads // num_key_value_heads`（`:201`）。GLM-4.5 即 96 个 Q 头共享 8 个 KV 头（12:1），大幅压缩 KV cache。
- **head_dim 解耦**：`head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)`（`:200`）。GLM-4.5 显式设 `head_dim=128`，而 `hidden_size/num_heads = 5120/96 ≈ 53.3`——即 **head_dim 与 hidden_size 解耦**，单头维度更大（128），这是「多头 + 大头维」的 deep-thin 配套设计。
- **attention bias**：`q/k/v_proj` 带 `bias=config.attention_bias`（`:207-215`，GLM-4.5 config 中 `attention_bias=true`），`o_proj` 无 bias（`:216`）。

### 5.2 Partial RoPE（部分旋转位置编码）

GLM 系列的标志性细节之一：**只对每个 head 的一部分维度施加 RoPE，其余维度不旋转**。

- 旋转维度计算：`Glm4MoeRotaryEmbedding` 内 `partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)`，`dim = int(head_dim * partial_rotary_factor)`（`modeling_glm4_moe.py:85-87`）。GLM-4.5 `partial_rotary_factor=0.5` ⇒ 128 维头中仅 **前 64 维** 参与 RoPE。
- 应用切分：`apply_rotary_pos_emb`（`:157-190`）按 `rotary_dim = cos.shape[-1]` 把 Q/K 切成 `q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]`（`:179-181`），仅对 `*_rot` 部分做 `(q_rot*cos)+(rotate_half(q_rot)*sin)`（`:184-185`），再 `torch.cat([q_embed, q_pass], dim=-1)` 拼回（`:188-189`）。
- 直觉：保留一部分「非位置相关」的表征通道，常被认为有利于长上下文外推与稳定性；GLM 从早期就采用（原始 GLM 用 2D 位置编码，GLM-4 系列改为 partial RoPE）。

**一组直观的数对照**（GLM-4.5，便于建立量感）：
- 每头 128 维，其中 64 维（前半）随相对位置旋转、64 维（后半）保持「绝对/内容」语义。`rope_theta=1e6` 配合 partial（仅半数维度参与）是常见的长上下文配方——大 `theta` 拉长波长、partial 减少高频维度对外推的干扰。
- **KV cache 量级**：GLM-4.5 用 GQA 把 KV 头从 96 压到 8。单 token 单层 KV 缓存 = `2 (K&V) × num_kv_heads(8) × head_dim(128) = 2048` 个元素；92 层即约 `2048 × 92 ≈ 18.8 万` 元素/token（BF16 约 377 KB/token）。若不用 GQA（96 头），同量将放大 12 倍。这解释了为何 deep-thin + 多头的设计仍能把长上下文（128K→200K）的显存压住——**深度换能力、GQA 换显存** 两手并用。

### 5.3 QK-Norm（Query/Key 归一化）

- 定义（条件化）：`use_qk_norm = config.use_qk_norm`；若开启，`self.q_norm = Glm4MoeRMSNorm(head_dim, ...)`、`self.k_norm = Glm4MoeRMSNorm(head_dim, ...)`（`modeling_glm4_moe.py:217-220`）。
- 应用位置：在 reshape 成多头之后、transpose 与 RoPE 之前，对 Q/K 各做一次 RMSNorm：`query_states = self.q_norm(query_states)` / `key_states = self.k_norm(key_states)`（`:237-239`，源码注释直言 *"main diff from Llama"*）。
- 作用：对每个 head 的 Q/K 向量做 RMS 归一化，约束注意力 logits 的尺度，提升大模型/长训练的数值稳定性。**GLM-4.5/4.6 开启，GLM-4.5-Air 关闭**（见 §3.1 config 差异）。

### 5.4 MoE 路由（sigmoid 门控 + 分组 + 共享专家 + loss-free 偏置）

GLM-4.5 的 MoE 在结构上明显借鉴 **DeepSeekMoE**（细粒度专家 + 共享专家 + auxiliary-loss-free 负载均衡），但用 **sigmoid** 而非 softmax 打分。核心在 `Glm4MoeMoE`（`modeling_glm4_moe.py:368-421`）与 `Glm4MoeTopkRouter`（`:287-305`）：

- **路由打分用 sigmoid**：`route_tokens_to_experts` 中 `router_logits = router_logits.sigmoid()`（`:389`）——每个专家独立打分而非归一化竞争。
- **loss-free 负载均衡偏置**：`router_logits_for_choice = router_logits + self.gate.e_score_correction_bias`（`:390`）。`e_score_correction_bias` 是注册的 buffer（`:299`），且被 `_keep_in_fp32_modules_strict`/初始化为零（`:488`/`:496`），训练时只用于「选择」而不进梯度——这正是 DeepSeek-V3 式的 **auxiliary-loss-free** 均衡。
- **分组受限路由（group-limited routing）**：先把专家 reshape 成 `(-1, n_group, n_routed_experts//n_group)`，取每组 top-2 之和当组分，再 `topk(group_scores, k=topk_group)` 选组、`masked_fill(-inf)` 屏蔽未选组，最后在被选组内 `topk(..., k=top_k)`（`:391-405`）。GLM-4.5 配置 `n_group=1, topk_group=1` ⇒ 退化为「全专家单组」上的 top-8。
- **top-k 权重归一化 + 缩放**：`if norm_topk_prob: topk_weights /= (sum + 1e-20)`，再 `topk_weights *= routed_scaling_factor`（`:407-410`，GLM-4.5 `routed_scaling_factor=2.5`）。
- **共享专家始终参与**：`Glm4MoeMoE.__init__` 建 `shared_experts = Glm4MoeMLP(intermediate_size = moe_intermediate_size * n_shared_experts)`（`:378-380`）；`forward` 中先算路由专家输出，再 **加上共享专家对原始输入的输出**：`hidden_states = hidden_states + self.shared_experts(residuals)`（`:413-421`）。
- **专家本体**：`Glm4MoeNaiveMoe`（`:329-366`）是参考实现（按 token 分发到 `Glm4MoeMLP` 专家）；`Glm4MoeMLP`（`:271-285`）是标准 SwiGLU（`down(act(gate)*up)`，`hidden_act=silu`）。

**参数预算的直觉**（为何 355B 总参却只激活 32B）：每层 160 个路由专家、每个专家 `moe_intermediate_size=1536`，但每 token 只激活 8 个 + 1 共享 ≈ 9 个专家的 MLP；外加 3 层 dense（`first_k_dense_replace=3`）与注意力参数。于是 **绝大多数参数「躺在」未被激活的专家里**——总参数堆容量（知识/技能多样性），激活参数定算力（推理/训练成本）。这正是 MoE 的核心杠杆，与 `[[03-deepseek]]` 的 671B-A37B 同理，只是 GLM 把「深度」也作为一个独立杠杆叠加上去。

### 5.5 deep & thin（深而窄）设计取舍

GLM-4.5 技术报告明确采用 **depth-over-width（深度优先于宽度）**：在同等参数预算下增加层数、收窄每层宽度，并提高注意力头数（96 头）。表现为：
- 层数 92（vs DeepSeek-V3 的 61），`hidden_size` 5120（vs 7168）。
- `num_attention_heads=96` 配 `head_dim=128`（解耦于 hidden_size）。
- 直觉与权衡：更深 → 更强的串行/组合推理能力（利于 reasoning、agentic 多步），但 **更深更难做张量/流水并行**（关键路径更长、激活通信更多），且对训练稳定性更敏感——这也解释了为何同时上 **QK-Norm**（稳 logits）与 **partial RoPE**（稳长程）。该取舍与「agent-native、重推理」的定位自洽。

### 5.6 dense 分支（GLM-4-9B）的差异

`glm4`（dense）相对 `glm4_moe` 的两点关键差异：
- **Sandwich norm（三明治归一化）**：`Glm4DecoderLayer`（`modeling_glm4.py:67-110`）除常规 `input_layernorm`/`post_attention_layernorm`（`:75`）外，额外有 `post_self_attn_layernorm`（`:76`）与 `post_mlp_layernorm`（`:77`），并在子层 **输出端再归一化**：`hidden_states = self.post_self_attn_layernorm(hidden_states)`（`:102`）、`... = self.post_mlp_layernorm(hidden_states)`（`:108`）。即「pre-norm + post-norm」夹心，进一步稳住残差流。
- **fused gate_up MLP**：`Glm4MLP`（`:49-64`）把 gate/up 合一为 `gate_up_proj: hidden→2*intermediate`（`:54`），`forward` 里 `gate, up_states = up_states.chunk(2, dim=-1)`，再 `up_states = up_states * activation_fn(gate)`（`:61-62`）——SwiGLU 的融合实现，省一次 matmul launch。
- dense 分支 **不含 QK-Norm**（`Glm4Attention`，`:197-260`，无 `q_norm`/`k_norm`）。

> MoE 分支的 `Glm4MoeDecoderLayer`（`modeling_glm4_moe.py:424-471`）只用常规两个 norm（`input_layernorm` `:436` + `post_attention_layernorm` `:437`），不带 sandwich 的两个 post-norm——这是 dense 与 MoE 两条实现路径的可见差异之一。

### 5.7 混合推理（Hybrid Reasoning）与 agentic / tool-use

- **双模式**：GLM-4.5/Air 是 **混合推理模型**——*thinking mode*（复杂推理 + 工具调用，输出 `<think>...</think>` 思维块）与 *non-thinking mode*（即时响应）共享同一份权重（fork `README.md:65-66`）。推理引擎默认开 thinking，可用 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭（fork `README.md:218-219`）。
- **交错思考 / 保留思考（4.7 起强化）**：交错思考 = 每次响应/工具调用前都思考；**保留思考** = 编码 agent 多轮对话中自动保留所有思维块、复用既有推理；**轮次级思考** = 按轮开关思考以平衡延迟/准确（fork `README.md:33-40`）。
- **agent-native**：模型把「工具使用」当一等公民，内置 `python` / `browser.search` / `browser.open` / `browser.find` 等工具协议（见 §6.2 与 fork `glm_4.6_tir_guide.md:32-36`），并为搜索 agent 设计了 thinking 模式下的专用 toolcall 格式（`resources/trajectory_search.json`）。

---

## 6. 源码走读（Code Walkthrough）

### 6.1 架构源码：transformers `modeling_glm4_moe.py`（URL:行）

> 全文件 650 行（撰写日 `main` 分支）。下表为关键符号定位，便于对照阅读。

| 符号 / 机制 | 行号 | 说明 |
|---|---|---|
| `Glm4MoeRotaryEmbedding` | `:46-111` | RoPE；`:85-87` 用 `partial_rotary_factor` 算 `dim` |
| `repeat_kv` | `:113-122` | GQA 的 KV 头复制 |
| `eager_attention_forward` | `:125-148` | 朴素注意力（含 `repeat_kv` 调用 `:135-136`） |
| `rotate_half` | `:150-155` | RoPE 半旋转 |
| `apply_rotary_pos_emb` | `:157-190` | **partial RoPE 切分**：`:179-181` 切 rot/pass，`:188-189` 拼回 |
| `Glm4MoeAttention` | `:193-269` | GQA `:201`；QK-Norm 定义 `:217-220`、应用 `:237-239` |
| `Glm4MoeMLP` | `:271-285` | SwiGLU 专家/dense MLP（silu） |
| `Glm4MoeTopkRouter` | `:287-305` | 路由权重 `:298`、`e_score_correction_bias` buffer `:299` |
| `Glm4MoeRMSNorm` | `:308-326` | RMSNorm |
| `Glm4MoeNaiveMoe` | `:329-366` | 专家分发参考实现 |
| `Glm4MoeMoE` | `:368-421` | sigmoid 路由 `:389`、loss-free 偏置 `:390`、分组 `:391-405`、归一+缩放 `:407-410`、共享专家相加 `:420` |
| `Glm4MoeDecoderLayer` | `:424-471` | `first_k_dense_replace` 选择 dense/MoE `:431-434` |
| `Glm4MoeModel` | `:503-575` | 主干 |
| `Glm4MoeForCausalLM` | `:577-650` | LM 头 |

dense 对照（`modeling_glm4.py`，536 行）关键定位：

| 符号 | 行号 | 说明 |
|---|---|---|
| `Glm4MLP`（fused gate_up） | `:49-64` | `gate_up_proj` `:54`、`chunk(2)` `:61` |
| `Glm4DecoderLayer`（sandwich norm） | `:67-110` | `post_self_attn_layernorm` `:76`、`post_mlp_layernorm` `:77`、应用 `:102`/`:108` |
| `apply_rotary_pos_emb`（partial） | `:157-190` | `:184` 切分 |
| `Glm4Attention`（无 QK-Norm） | `:197-260` | GQA `:205` |
| `Glm4RotaryEmbedding` | `:262-328` | `partial_rotary_factor` `:301-303` |

> **关于 MTP**：transformers 的 `glm4_moe` 建模文件本身 **不含独立的 MTP 模块**（`num_nextn_predict_layers` 只是 config 字段，由推理引擎读取）。多 token 预测层的实际推理实现位于 **vLLM 的 `glm4_moe_mtp.py`** 与 **SGLang 的 `glm4_moe.py`**（fork `README.md:94`）。fork README 的部署命令也通过 `--speculative-config.method mtp`（vLLM）或 `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（SGLang）启用 MTP 投机解码（fork `README.md:104-106, 168-193`）。

### 6.2 本地推理脚本与工具推理（`forks/GLM-4.5/`）

**(a) 最小 transformers 推理**——`forks/GLM-4.5/inference/trans_infer_cli.py`：
- `MODEL_PATH = "zai-org/GLM-4.7"`（`:4`，fork 已更新到 4.7 默认）。
- 用 `tokenizer.apply_chat_template(..., add_generation_prompt=True, return_dict=True, return_tensors="pt")` 构造输入（`:7-13`）——chat template 内置了 thinking/工具协议。
- `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="auto")` 加载（`:14-18`），`model.generate(max_new_tokens=128, do_sample=False)`（`:20`）。
- 教学价值：展示「权重 + chat template」即可跑，建模逻辑全在 transformers 的 `glm4_moe`/`glm4_moe_lite`（4.7-Flash）。

**(b) OpenAI 兼容工具调用**——`forks/GLM-4.5/inference/api_request.py`：
- 用 `openai.OpenAI` 指向本地 vLLM/SGLang 服务（`:1-4, 66-69`），传 OpenAI 风格 `tools`（`get_weather`、`return_delivered_order_items`，`:7-53`）。
- 演示「模型发 `tool_calls` → 程序执行 → 把 tool 结果回灌（`role:"tool"`）→ 二次请求拿最终答」的完整回合（`:70-92`）。
- 关闭思考用 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`（`:75, 89`）。

**(c) 工具集成推理 TIR**——`forks/GLM-4.5/resources/glm_4.6_tir_guide.md`：
- 四步循环：给工具定义 → 模型自主决定是否调用并给参数 → 用户执行并回灌结果 → 模型继续，直到信息足够（`:3-8`）。
- **内置工具协议**：`python`（解释器）、`browser.search`、`browser.open`、`browser.find`（`:32-94`）——这是 GLM agent 能力的「标准接口」。
- 启服务参数：`--tool-call-parser glm45 --reasoning-parser glm45 --speculative-algorithm EAGLE ...`（`:14-28`），即 reasoning 与 tool-call 的自动解析 + MTP 投机解码。
- **手动解析协议（不依赖 parser 时）**：思维块用 `<think>...</think>` 包裹；每个工具调用用 `<tool_call>...</tool_call>` 包裹，函数名紧跟 `<tool_call>`，参数用 `<arg_key>...</arg_key><arg_value>...</arg_value>`（`:178-183`）。指南给出 `parse_model_response` 正则解析示例（`:190-266`），并演示直连 completions API + 手动解析的循环（`:271-365`）。
- 教学价值：把「混合推理 + agentic tool-use」落到 **可见的 token 级协议**，是理解 GLM agent 行为的最佳一手材料。

---

## 7. 训练机制（Training）

> 来源：GLM-4.5 技术报告（[arXiv 2508.06471](https://arxiv.org/abs/2508.06471)）、slime 官方（[GitHub](https://github.com/THUDM/slime) / [docs](https://thudm.github.io/slime/)）、fork README。具体数字以技术报告为准；下列为综述级提炼，细粒度数字标注「约/未核实」。

### 7.1 预训练 + mid-training（多阶段）

- **规模**：多阶段训练，合计 **约 23T tokens**（arXiv 摘要）；技术报告口径中包含大比例 **代码 + 推理** 数据（约 7T tokens 量级，未核实精确切分）。
- **mid-training（中段训练）**：在通用预训练之后、对齐之前，插入面向 **repo 级代码、长上下文、推理、agent 轨迹** 的中段训练，把基础模型「掰向」ARC（agentic/reasoning/coding）能力；长上下文能力即在此阶段扩展（GLM-4.6 进一步扩到 200K）。
- 这一「general pretrain → mid-training → post-training」的三段式，与 DeepSeek/Qwen 的「大规模预训练 + 专项续训」思路一致，但 GLM 把 **agent 轨迹/工具使用** 提前到 mid-training 注入，是其 agent-native 定位的训练侧体现。

### 7.2 后训练：专家迭代 + 强化学习 + RLHF

- **expert model iteration（专家模型迭代）**：先训出若干 **领域专家模型**（如推理专家、编码专家、agent 专家），再把它们的能力 **蒸馏/合并** 回统一的混合推理模型——用「分而治之再统一」绕开单一 RL 难以同时优化多目标的问题。
- **多目标 RL**：
  - **Reasoning RL**：在数学/科学/代码等可验证任务上做 RLVR（可验证奖励 RL）。
  - **Agentic RL**：在多轮工具使用 / 浏览 / 软件工程任务上做 agent RL（长轨迹、环境反馈、verifier 引导分支）。
  - **RLHF**：对话偏好对齐（风格、可读性、安全）。
- **混合推理对齐**：thinking / non-thinking 双模式需要在后训练中同时对齐，保证一份权重可切换。

**「专家迭代 → 统一」配方的直觉**（为何不直接对统一模型做多目标 RL）：
1. 多目标 RL 互相打架——把数学、代码、agent、对话偏好放进同一个 RL 目标，奖励信号会相互稀释、彼此遗忘。
2. 因此先在 **各自最优的数据/奖励/超参** 下分别 RL 出领域专家（reasoning / coding / agentic），让每个专家把自己那条赛道推到极致。
3. 再用 **拒绝采样 / 蒸馏 / 数据回灌**，把专家产出的高质量轨迹合成到统一模型上（监督式吸收 + 轻量 RL 收敛），得到「样样都强」的单一混合推理权重。
- 这套「先分化专家、再蒸馏统一」的范式，与 `[[03-deepseek]]` 用 R1 产数据回灌 V3、以及多家前沿模型的「专家蒸馏」后训练同源；GLM 的差异点是把 **agentic 轨迹** 当作一等专家方向，并配 slime 把长轨迹 RL 工程化（见 §7.3）。

### 7.3 自研 RL 框架 **slime**

slime（[github.com/THUDM/slime](https://github.com/THUDM/slime)）是智谱自研的 **LLM 后训练 / RL Scaling 框架**，是 GLM-4.5/4.6/4.7 等的 RL 基础设施。要点：
- **三件套模块化架构**：**训练后端 = Megatron-LM**（高性能并行训练）+ **rollout 后端 = SGLang（+ 自定义 router）**（高吞吐采样生成）+ **中央 Data Buffer**（管理 prompt 初始化与 rollout 存储）。
- **异步 / 解耦**：训练与生成解耦，轨迹可独立异步生成；引入 **Active Partial Rollouts (APRIL)** 等系统级优化，缓解「RL 中 90%+ 时间耗在生成」的瓶颈。
- **灵活 rollout 接口**：可在不改底层基础设施的前提下实现任务特定的 rollout 逻辑——多轮交互循环、工具调用、环境反馈处理、verifier 引导分支——从而在 **统一训练栈** 内支持 reasoning RL / general RL / **agentic RL** / on-policy 蒸馏。
- 教学价值：slime 是「**为 agentic RL 而生**」的框架范例——把推理引擎（SGLang）当作一等的「环境/采样器」嵌进 RL 循环，是当前长轨迹 agent RL 的主流工程形态。

### 7.4 微调与部署生态（fork README）

- 微调：官方给出 LLaMA-Factory（LoRA：GLM-4.5 H100×16 / Air H100×4）与 ms-swift（LoRA/SFT/RL，H20 集群）配置（fork `README.md:137-155`）。
- 部署：vLLM / SGLang 一等支持，含 **PD 分离（prefill/decode disaggregation）** 与 **MTP 投机解码** 模板（fork `README.md:165-204`）；Ascend NPU（xLLM）、AMD GPU 亦有部署文档（`example/`）。

---

## 8. 学习要点与横向对比（Takeaways & Comparison）

### 8.1 一句话提炼

GLM 从「用 blank-infilling 统一理解与生成」的学术范式，演化为「**深而窄的大 MoE + 混合推理 + agent-native**」的前沿开源基座；架构上把 DeepSeekMoE 的「细粒度专家 + 共享专家 + loss-free 均衡」与自家的 **partial RoPE / 选择性 QK-Norm / sandwich-norm（dense）/ MTP** 组合，训练上靠 **mid-training 注入 agent 能力 + slime 异步 RL** 把 ARC 三能力拧成一股绳。

### 8.2 与大 MoE 同行的横向对比

对照 `[[03-deepseek]]`（DeepSeek-V3 671B-A37B）与 `[[09-minimax]]`（MiniMax-01 456B-A45.9B，lightning/linear attention）：

| 维度 | GLM-4.5 | DeepSeek-V3 `[[03-deepseek]]` | MiniMax-01 `[[09-minimax]]` |
|---|---|---|---|
| 规模 | 355B-A32B | 671B-A37B | 456B-A45.9B |
| 注意力 | **GQA + partial RoPE + QK-Norm** | **MLA（潜空间 KV 压缩）** | **Lightning/Linear Attn + 周期性 softmax** |
| MoE 均衡 | sigmoid + **loss-free 偏置**（n_group=1 退化） | sigmoid + **loss-free 偏置** + 分组受限路由 | softmax top-k（aux loss） |
| 共享专家 | 1 | 1 | 有 |
| 形状取舍 | **deep & thin（92 层 / 5120）** | shallow-wide（61 层 / 7168） | 宽 + 线性注意力换长上下文 |
| 投机解码 | **MTP（1 层）** | **MTP（多 token 预测）** | — |
| 上下文 | 128K（4.6→200K） | 128K | **~1M（4M 实验）** |
| 训练侧亮点 | mid-training 注入 agent + **slime 异步 RL** | **FP8 训练 + auxiliary-loss-free** | 线性注意力训练效率 |

**关键差异**：GLM 与 DeepSeek 的 MoE「家族相似度」很高（loss-free 偏置、共享专家、MTP），主要分歧在 **注意力（GLM 走 GQA+partial RoPE+QK-Norm，DeepSeek 走 MLA）** 与 **形状（GLM 深窄 vs DeepSeek 浅宽）**；MiniMax 则在注意力机制上走得更远（线性注意力 + 超长上下文）。三者都用「共享专家 + 细粒度专家」的 DeepSeekMoE 范式做骨架。

### 8.3 混合推理对照 `[[02-qwen]]`

GLM-4.5 与 Qwen3 都是 **一份权重双模式（thinking / non-thinking）** 的混合推理路线（对照 DeepSeek 早期 R1 用独立推理模型、后由 V3.1 才合一）。差异在落地侧：
- **GLM**：thinking 用 `<think>` 块 + `enable_thinking` 开关，并把「**保留思考 / 轮次级思考**」做成 agent 多轮的一等特性（4.7），强调 **agentic** 场景下的思考复用与一致性。
- **Qwen3** `[[02-qwen]]`：同样支持思考开关，定位更偏通用；GLM 的差异化在于 **更重 agent / tool-use 的产品-训练-协议一体化**（内置 browser/python 工具协议 + TIR 指南 + 搜索 agent 模板 + slime agentic RL）。

### 8.4 给学习者的 checklist

- 想理解 **partial RoPE / QK-Norm**：读 `modeling_glm4_moe.py:157-190`（切分）与 `:217-239`（QK-Norm 定义/应用），对照 config `partial_rotary_factor=0.5`、`use_qk_norm`。
- 想理解 **loss-free MoE 均衡**：读 `:389-410`（sigmoid + `e_score_correction_bias` + 分组 + 归一缩放），与 `[[03-deepseek]]` 的 DeepSeekMoE 对照。
- 想理解 **MTP 投机解码**：config `num_nextn_predict_layers=1` + vLLM `glm4_moe_mtp.py` + fork README 的 `--speculative-*` 参数。
- 想理解 **agentic 协议**：读 fork `resources/glm_4.6_tir_guide.md`（`<tool_call>/<arg_key>/<arg_value>` 协议 + browser/python 工具）。
- 想理解 **agentic RL 工程**：读 slime（Megatron 训练 + SGLang rollout + Data Buffer + APRIL）。

---

## 9. 参考（References）

**官方仓库 / 模型 / 框架**
- GLM-4.5 GitHub（含推理脚本、TIR 指南）：https://github.com/zai-org/GLM-4.5
- 本仓库 fork：`forks/GLM-4.5`（`README.md` / `README_zh.md` / `inference/trans_infer_cli.py` / `inference/api_request.py` / `resources/glm_4.6_tir_guide.md` / `resources/trajectory_search.json`）
- HuggingFace 组织（原 THUDM）：https://huggingface.co/zai-org
- HF model cards：GLM-4.5 / GLM-4.5-Air / GLM-4.6 / GLM-4-9B-0414（`config.json` 为规格来源）
- slime（RL 框架）：https://github.com/THUDM/slime ・ 文档 https://thudm.github.io/slime/

**建模源码（transformers `main`）**
- MoE：https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py
- dense：https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4/modeling_glm4.py
- vLLM MTP：https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/glm4_moe_mtp.py
- SGLang：https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/glm4_moe.py

**论文**
- GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — [arXiv 2508.06471](https://arxiv.org/abs/2508.06471)
- 原始 GLM: General Language Model Pretraining with Autoregressive Blank Infilling — [arXiv 2103.10360](https://arxiv.org/abs/2103.10360)
- GLM-130B: An Open Bilingual Pre-trained Model — [arXiv 2210.02414](https://arxiv.org/abs/2210.02414)
- ChatGLM: A Family of LLMs from GLM-130B to GLM-4 (All Tools) — [arXiv 2406.12793](https://arxiv.org/abs/2406.12793)

**相关本仓库文档**：`[[02-qwen]]`（混合推理对照）、`[[03-deepseek]]`（大 MoE / MLA / MTP / loss-free 对照）、`[[09-minimax]]`（大 MoE / 线性注意力 / 超长上下文对照）。

> 数字纪律：本文 §3 规格取自 HF `config.json`（精确）；§4 benchmark 取自技术报告/官方榜单，凡单点核实或口径存疑者已标「约/未核实」，引用请回溯 [arXiv 2508.06471](https://arxiv.org/abs/2508.06471) 原表并注明 thinking/工具设定与评测版本。§5/§6.1 行号对照 transformers `main` 分支撰写日快照，后续上游重构可能产生行号漂移。

---

## 附录：术语速查（Glossary）

| 术语 | 含义（GLM 语境） |
|---|---|
| **GLM**（General Language Model） | 用 *autoregressive blank infilling* 统一「双向理解 + 单向生成」的预训练范式（[2103.10360](https://arxiv.org/abs/2103.10360)）；后泛指智谱该系列模型 |
| **autoregressive blank infilling** | 遮盖若干文本 span，以任意顺序自回归地填回；通过 span 的数量/长度配置覆盖 NLU / 无条件生成 / 条件生成 |
| **MoE / A32B** | Mixture-of-Experts；「355B-A32B」= 总参 355B、每 token 激活 32B（active）|
| **partial RoPE** | 仅对 head 的一部分维度（GLM-4.5 为 50%）施加旋转位置编码，其余维度不旋转 |
| **QK-Norm** | 对每头 Q/K 向量各做一次 RMSNorm（GLM-4.5/4.6 开、Air 关），稳住注意力 logits 尺度 |
| **GQA** | grouped-query attention；多 Q 头共享少量 KV 头（GLM-4.5 为 96:8），压缩 KV cache |
| **shared expert（共享专家）** | 对每个 token 恒激活的专家，承载通用能力，与路由专家相加 |
| **auxiliary-loss-free 均衡** | 用可学习的 `e_score_correction_bias` 仅影响「选专家」而不进损失/梯度的负载均衡（DeepSeek-V3 式）|
| **MTP**（multi-token prediction） | 额外的「下 N 个 token 预测」层（GLM-4.5 `num_nextn_predict_layers=1`），推理期做投机解码加速 |
| **first_k_dense_replace** | 前 k 层用普通 dense MLP、其余层用 MoE（GLM-4.5=3、Air=1）|
| **deep & thin** | 深度优先于宽度的形状取舍（92 层 / 5120 维 / 96 头）|
| **hybrid reasoning（混合推理）** | 一份权重支持 thinking（`<think>` 思维块）/ non-thinking 双模式 |
| **interleaved / preserved / turn-level thinking** | 4.7 引入：动作间思考 / 多轮保留思维块复用 / 按轮开关思考 |
| **TIR**（tool-integrated reasoning） | 推理过程中调用工具（python/browser）的循环（见 fork `glm_4.6_tir_guide.md`）|
| **slime** | 智谱自研 RL 后训练框架：Megatron 训练 + SGLang rollout + Data Buffer + APRIL；支撑 agentic RL |
| **expert model iteration** | 先分别 RL 出领域专家，再蒸馏/合并回统一混合推理模型 |
