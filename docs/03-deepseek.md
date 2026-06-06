# DeepSeek 深度分析（MLA / DeepSeekMoE / MTP / FP8 / R1 强化学习）

> 本文档为 Model-Research 学习仓库的 DeepSeek 专题。源码走读基于本地 fork submodule
> `forks/DeepSeek-V3`（commit 内容为官方 inference demo）与 `forks/DeepSeek-R1`（含 `DeepSeek_R1.pdf` 论文）。
> 所有 `文件:行号` 引用均已对照当前文件内容核对。规格/benchmark 数字尽量给官方来源 URL，无法核实者标注「未核实/约」。
> 撰写日期：2026-06-06。

---

## 1. 概览

**出品方**：DeepSeek-AI（深度求索，杭州幻方量化背景）。主打「开源 + 极致训练效率 + 强推理」，模型权重以宽松许可证（V3 系列代码 MIT、模型 Model Agreement；R1 系列代码与权重均 MIT）发布，支持商用与蒸馏。

**定位**：用 MoE（Mixture-of-Experts）把「总参数规模」与「单 token 计算成本」解耦——671B 总参但每 token 仅激活 37B，从而以接近中型 dense 模型的推理/训练成本，达到前沿闭源模型档位的能力。V3 是通用基座，R1 是在 V3-Base 上用大规模 RL 训出的推理（reasoning）模型。

**版本演进表**：

| 版本 | 时间 | 关键变化 | 来源 |
|---|---|---|---|
| DeepSeek LLM (V1) | 2023-11 | 7B/67B dense，对标 LLaMA | [arXiv 2401.02954](https://arxiv.org/abs/2401.02954) |
| DeepSeek-V2 | 2024-05 | 首发 **MLA + DeepSeekMoE**；236B-A21B；KV cache 降 93.3% | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) |
| DeepSeek-V2.5 | 2024-09 | V2-Chat 与 Coder 合并；通用+代码能力统一 | [HF deepseek-ai](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) |
| **DeepSeek-V3** | 2024-12 | 671B-A37B；**auxiliary-loss-free 负载均衡 + MTP + FP8 训练**；14.8T tokens | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) |
| **DeepSeek-R1 / R1-Zero** | 2025-01 | 在 V3-Base 上做 **GRPO 强化学习**；R1-Zero 纯 RL，R1 加 cold-start；蒸馏到 Qwen/Llama | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) |
| DeepSeek-R1-0528 | 2025-05 | R1 小版本升级；AIME 2025 70.0→87.5；新增 R1-0528-Qwen3-8B 蒸馏 | [HF R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| DeepSeek-V3.1 | 2025-08 | **hybrid thinking**（一份权重双模式：思考/非思考）；+840B tokens 续训；UE8M0 FP8 | [HF V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) |
| DeepSeek-V3.1-Terminus | 2025-09 | V3.1 的稳定/工具调用增强版（作为 V3.2 的对照基线） | [GitHub V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |
| DeepSeek-V3.2-Exp | 2025-09（实验版）| 首发 **DeepSeek Sparse Attention (DSA)**，长上下文近线性复杂度，质量持平 Terminus | [HF V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) |
| DeepSeek-V3.2 | 2025-12 | DSA 正式版；160K 上下文；IMO/CMO/ICPC 金牌级 | [HF V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |

> 说明：本仓库 fork 的是 **V3 / R1** 的官方 inference 仓库。V3.1/V3.2/R1-0528 为联网核实到的后续版本（截至 2026-06）；其中 fork 内 `configs/config_v3.1.json` 已带 `"scale_fmt": "ue8m0"` 字段，与 V3.1 公布的 UE8M0 FP8 缩放格式吻合（见 §6）。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| 官方 GitHub（V3，含 inference 源码） | https://github.com/deepseek-ai/DeepSeek-V3 |
| 官方 GitHub（R1） | https://github.com/deepseek-ai/DeepSeek-R1 |
| 官方 GitHub（V3.2-Exp，DSA） | https://github.com/deepseek-ai/DeepSeek-V3.2-Exp |
| 本仓库 fork（V3） | `forks/DeepSeek-V3`（`inference/model.py` 等） |
| 本仓库 fork（R1） | `forks/DeepSeek-R1`（`DeepSeek_R1.pdf`、`README.md`） |
| HuggingFace 组织 | https://huggingface.co/deepseek-ai |
| HF: DeepSeek-V3 / V3-Base | https://huggingface.co/deepseek-ai/DeepSeek-V3 |
| HF: DeepSeek-R1 / R1-Zero | https://huggingface.co/deepseek-ai/DeepSeek-R1 |
| 论文：DeepSeek-V3 Technical Report | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) |
| 论文：DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) |
| 论文：DeepSeek-V2（MLA/MoE 起源） | [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) |
| 论文：DeepSeekMath（GRPO 起源） | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) |

---

## 3. 模型规格

DeepSeek-V3（671B）配置以本地 `forks/DeepSeek-V3/inference/configs/config_671B.json` 为准（即生产权重的实际超参），与官方 README 交叉验证。

| 维度 | DeepSeek-V3 (671B) | 来源 |
|---|---|---|
| 总参数 | 671B | README.md:47 / [arXiv 2412.19437 摘要](https://arxiv.org/abs/2412.19437) |
| 激活参数/token | 37B（精确 36.7B，含 0.9B Embedding + 0.9B Head） | README.md:47 / README_WEIGHTS.md:20-22 |
| 层数 `n_layers` | 61 | config_671B.json |
| 前 N 层为 dense `n_dense_layers` | 3（前 3 层用普通 MLP，其余用 MoE） | config_671B.json |
| 隐藏维 `dim` | 7168 | config_671B.json |
| dense MLP 中间维 `inter_dim` | 18432 | config_671B.json |
| MoE 专家中间维 `moe_inter_dim` | 2048 | config_671B.json |
| 注意力头数 `n_heads` | 128 | config_671B.json |
| MLA: `kv_lora_rank`（KV 压缩潜维） | 512 | config_671B.json / model.py:76 |
| MLA: `q_lora_rank`（Q 压缩潜维） | 1536 | config_671B.json |
| MLA: `qk_nope_head_dim` / `qk_rope_head_dim` | 128 / 64 | config_671B.json / model.py:77-78 |
| MLA: `v_head_dim` | 128 | config_671B.json |
| MoE: 路由专家数 `n_routed_experts` | 256 | config_671B.json |
| MoE: 共享专家数 `n_shared_experts` | 1 | config_671B.json |
| MoE: 每 token 激活专家 `n_activated_experts` | 8（+1 共享，常激活） | config_671B.json |
| MoE: 专家分组 `n_expert_groups` / `n_limited_groups` | 8 / 4（受限组路由） | config_671B.json |
| 路由打分 `score_func` | sigmoid（V3 用 sigmoid，V2/16B 用 softmax） | config_671B.json / model.py:564 |
| 词表 `vocab_size` | 129280 | config_671B.json |
| 上下文长度 | 128K（YaRN 外推，原生 4K → ×40） | README.md:93 / model.py:81-83 |
| 训练精度 | FP8（`dtype: fp8`，e4m3，128×128 block scaling） | config_671B.json / README_WEIGHTS.md:62 |
| MTP 模块 | 1 个（`num_nextn_predict_layers=1`，11.5B 独立参数） | README_WEIGHTS.md:6,38 |
| HF 权重总大小 | 685B（671B 主模型 + 14B MTP 模块） | README.md:99 |

R1 / R1-Zero **复用 V3-Base 的全部架构与规格**（671B-A37B、128K），只是后训练不同（见 §7）。来源：R1 README.md:78「DeepSeek-R1-Zero & DeepSeek-R1 are trained based on DeepSeek-V3-Base」。

仓库内另有两个对照配置（来自 DeepSeek-V2 家族，方便学习/小规模跑通）：

| 配置 | 总参 | dim | 层 | 路由专家 | 激活专家 | 打分 | 来源 |
|---|---|---|---|---|---|---|---|
| `config_16B.json` | ~16B | 2048 | 27 | 64 | 6 | softmax | configs/config_16B.json |
| `config_236B.json` | ~236B (V2) | 5120 | 60 | 160 | 6 | softmax | configs/config_236B.json |
| `config_671B.json` | 671B (V3) | 7168 | 61 | 256 | 8 | sigmoid | configs/config_671B.json |

---

## 4. Benchmark

> 标注「自报」= DeepSeek 官方 README/论文 报告值；评测设置见各来源脚注（如 R1 系列输出上限 32768 tokens、temperature 0.6、top-p 0.95、pass@1）。第三方独立复现可能有出入。

### 4.1 DeepSeek-V3-Chat（自报，来源 V3 README.md:167-194）

| Benchmark (Metric) | DeepSeek-V3 | 对照（最优他者） | 来源 |
|---|---|---|---|
| MMLU (EM) | 88.5 | LLaMA-3.1-405B 88.6 | README.md:172 |
| MMLU-Pro (EM) | 75.9 | Claude-3.5-Sonnet 78.0 | README.md:174 |
| GPQA-Diamond (Pass@1) | 59.1 | Claude-3.5-Sonnet 65.0 | README.md:177 |
| MATH-500 (EM) | **90.2** | 全场最高 | README.md:189 |
| AIME 2024 (Pass@1) | **39.2** | 全场最高 | README.md:188 |
| Codeforces (Percentile) | **51.6** | 全场最高 | README.md:184 |
| SWE Verified (Resolved) | 42.0 | Claude-3.5-Sonnet 50.8 | README.md:185 |
| LiveCodeBench (Pass@1-COT) | **40.5** | 全场最高 | README.md:182 |

### 4.2 DeepSeek-R1（自报，来源 R1 README.md:106-131 / 论文 Table 4）

| Benchmark (Metric) | DeepSeek-R1 | DeepSeek-V3 | OpenAI o1-1217 | 来源 |
|---|---|---|---|---|
| MMLU (Pass@1) | 90.8 | 88.5 | **91.8** | README.md:111 |
| MMLU-Pro (EM) | **84.0** | 75.9 | – | README.md:113 |
| GPQA-Diamond (Pass@1) | 71.5 | 59.1 | **75.7** | README.md:116 |
| MATH-500 (Pass@1) | **97.3** | 90.2 | 96.4 | README.md:127 |
| AIME 2024 (Pass@1) | **79.8** | 39.2 | 79.2 | README.md:126 |
| Codeforces (Rating) | 2029 | 1134 | **2061** | README.md:123 |
| LiveCodeBench (Pass@1-COT) | **65.9** | – | 63.4 | README.md:121 |
| SWE Verified (Resolved) | 49.2 | 42.0 | 48.9 | README.md:124 |

要点：R1 相对 V3 在数学/代码/知识推理上大幅跃升（AIME 39.2→79.8、MATH-500 90.2→97.3），整体与 o1-1217 同档；工程类（SWE/Aider）提升有限（论文 §3.1 自承 RL 训练数据少）。

### 4.3 R1 蒸馏模型（自报，来源 R1 README.md:141-152 / 论文 Table 5）

| 蒸馏模型 | 基座 | AIME24 pass@1 | MATH-500 | GPQA-D | 来源 |
|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28.9 | 83.9 | 33.8 | README.md:147 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55.5 | 92.8 | 49.1 | README.md:148 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | **72.6** | 94.3 | 62.1 | README.md:150 |
| R1-Distill-Llama-70B | Llama-3.3-70B-Inst | 70.0 | **94.5** | **65.2** | README.md:152 |

### 4.4 后续版本（联网核实）

| 版本 | 关键 benchmark | 来源 |
|---|---|---|
| R1-0528 | AIME 2025 70.0→87.5；GPQA-D 71.5→81.0；LiveCodeBench 63.5→73.3（自报） | [HF R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| V3.2-Exp vs V3.1-Terminus | AIME25 88.4→89.3；Codeforces 2046→2121；MMLU-Pro 85.0=85.0；GPQA-D 80.7→79.9；SWE-bench Multilingual 57.8→57.9 | [GitHub V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) |

---

## 5. 架构剖析

DeepSeek-V3 的 Transformer block 结构很标准（pre-norm 残差），关键创新全在「注意力」与「FFN」两处：注意力用 **MLA**，FFN（除前 3 层）用 **DeepSeekMoE**。Block 装配见 `model.py:706-735`：

```python
# model.py:715-718  每个 block：MLA 注意力 + (前 n_dense_layers 层 MLP，否则 MoE)
self.attn = MLA(args)
self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
self.attn_norm = RMSNorm(args.dim)
self.ffn_norm = RMSNorm(args.dim)
# model.py:733-734  pre-norm 残差：x = x + attn(norm(x)); x = x + ffn(norm(x))
```

### 5.1 MLA：Multi-head Latent Attention（KV 低秩压缩 + decoupled RoPE）

**动机**：标准 MHA 推理时要缓存每个 token、每个 head 的完整 K 和 V，KV cache 随上下文线性膨胀，是长上下文显存墙。MLA 的思路：不缓存高维 K/V，而是把它们**压缩成一个低维潜向量 `c_KV`（维度 `kv_lora_rank=512`）来缓存**，用时再用上投影矩阵 `wkv_b` 还原。相对 DeepSeek-V2 报告 KV cache 降低 93.3%（来源 [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) 摘要）。

**结构定义**（`model.py:412-444`）：

```python
# model.py:430-433  KV 下投影到 (kv_lora_rank + qk_rope_head_dim)，再上投影回 nope+v
self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)  # 下投影：产出 c_KV 和解耦的 k_pe
self.kv_norm = RMSNorm(self.kv_lora_rank)
self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))  # 上投影
self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
```

**decoupled RoPE（分离的旋转位置编码维度）**：MLA 把每个 head 的 query/key 拆成两段——`qk_nope_head_dim=128`（不加位置编码，可被低秩吸收）+ `qk_rope_head_dim=64`（单独走 RoPE）。为什么要解耦？因为低秩压缩后的 K 无法再正确地施加 position-dependent 的 RoPE（RoPE 与上投影矩阵不可交换），所以 RoPE 维度必须从压缩路径里**单独拎出来**，对所有 head 共享一份 `k_pe`。拆分与施加 RoPE 见 `model.py:466-470`：

```python
# model.py:466-470  query 拆 nope/rope 两段，仅 rope 段过 RoPE；k 的 rope 段全 head 共享
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
q_pe = apply_rotary_emb(q_pe, freqs_cis)
kv = self.wkv_a(x)                                                   # 下投影
kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)               # 解耦 RoPE，仅 1 份（unsqueeze 后广播到所有 head）
```

**推理期省显存的关键——"absorb" 实现**（`attn_impl="absorb"`，默认，见 `model.py:17`）：缓存的不是还原后的高维 K/V，而是 `kv_cache`（压缩潜向量，维度 512）+ `pe_cache`（解耦 RoPE，维度 64）。注意力打分时，把上投影矩阵 `wkv_b` **吸收进 query** 一侧（`q_nope @ wkv_b`），从而无需把每个缓存 token 还原成高维 K：

```python
# model.py:443-444  缓存的是压缩潜向量(512)+解耦RoPE(64)，而非高维K/V
self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), ...)
self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), ...)
# model.py:481-487  把上投影吸收进 q_nope；score = q_nope·c_KV + q_pe·k_pe
wkv_b = self.wkv_b.weight ...
wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])   # 吸收上投影
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)                            # 只存压缩潜向量
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
          torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
```

对比 `attn_impl="naive"`（`model.py:471-479`）会真的把 K/V 还原并缓存高维张量——那条分支只为教学/对拍，生产用 absorb。这正是 MLA「推理期 KV cache 大幅缩小」的工程落点。

> Q 侧也做了低秩压缩（`q_lora_rank=1536`，见 `model.py:426-429`），但 Q 不进 KV cache，压缩 Q 主要是省参数/激活，不省 cache。当 `q_lora_rank==0`（16B 配置）时 Q 走普通线性（`model.py:424-425`）。

### 5.2 DeepSeekMoE：细粒度专家 + 共享专家 + auxiliary-loss-free 负载均衡

**两个设计**（`MoE` 类 `model.py:636-693`）：
1. **细粒度路由专家**（256 个，每 token 选 8 个）：把专家切得更细，提高组合多样性。
2. **共享专家**（`shared_experts`，V3 为 1 个）：对**所有 token 恒定激活**，承接通用知识，让路由专家专注「专」的部分。见 `model.py:667` 与 `model.py:690`：

```python
# model.py:667  共享专家：一个更宽的 MLP（n_shared_experts * moe_inter_dim），对所有 token 都算
self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
# model.py:681-693  forward：门控选专家 → 累加被选路由专家输出 → 再加上共享专家 z
weights, indices = self.gate(x)
...
for i in range(self.experts_start_idx, self.experts_end_idx):
    ...
    y[idx] += expert(x[idx]) * weights[idx, top, None]   # 加权累加路由专家
z = self.shared_experts(x)                               # 共享专家（恒激活）
return (y + z).view(shape)
```

**auxiliary-loss-free 负载均衡（V3 关键创新）**：传统 MoE 用一个 auxiliary load-balancing loss 逼专家均衡，但这个辅助 loss 与语言建模目标冲突、会损害性能。V3 改为**给每个专家的路由打分加一个可学习的 bias**，只用 bias 调整「选哪些专家」，而**最终的加权系数仍用不含 bias 的原始打分**——这样均衡完全不污染梯度/损失。落点在 `Gate` 类（`model.py:535-598`）：

```python
# model.py:564  仅 671B(dim==7168) 配置带可学习 bias（auxiliary-loss-free 的载体）
self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None
# model.py:576-583  原始打分 original_scores 留作权重；bias 只用于"选择"
scores = linear(x, self.weight)
scores = scores.softmax(...) if self.score_func == "softmax" else scores.sigmoid()  # V3 用 sigmoid
original_scores = scores
if self.bias is not None:
    scores = scores + self.bias            # bias 仅影响 topk 选择
# model.py:584-593  分组受限路由（n_groups=8, topk_groups=4）：先选组再组内选专家
...
indices = torch.topk(scores, self.topk, dim=-1)[1]        # 用"加了bias的分数"选专家
# model.py:594-597  但权重取自 original_scores（不含 bias）
weights = original_scores.gather(1, indices)
if self.score_func == "sigmoid":
    weights /= weights.sum(dim=-1, keepdim=True)          # sigmoid 路径归一化
weights *= self.route_scale
```

> 注意：`model.py` 是**推理**代码，bias 在这里是已训练好的固定参数；论文里它在训练时按各专家负载动态增减（过载专家调低 bias、欠载调高）。这一点训练逻辑不在 fork 的 inference 源码内（来源 [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)）。分组受限路由（group-limited routing）的作用是限制每个 token 的专家跨越的「组数」，减少跨节点 all-to-all 通信。

### 5.3 MTP：Multi-Token Prediction（多 token 预测训练目标）

V3 在训练时**额外预测未来第 2 个 token**（不仅是 next-token），用一个轻量 MTP 模块串接在主干之后，提供更密集的训练信号；推理时该模块可丢弃，或用于 speculative decoding 加速。要点：

- 模块组成：共享主干的 Embedding 与输出 Head，外加 `enorm`/`hnorm`（RMSNorm）、`eh_proj`（降维投影）、以及一层与主干同构的 Transformer（即 `model.layers.61`）。来源 README_WEIGHTS.md:42-48。
- 参数量：11.5B 独立参数（不含共享的 Embedding/Head），激活 2.4B。开源 V3 权重含 **1 个** MTP 模块（`num_nextn_predict_layers=1`）。来源 README_WEIGHTS.md:6,38。
- **在 inference fork 中 MTP 被显式忽略**：`forks/DeepSeek-V3/inference/model.py` 全文无 MTP 类（`grep` 确认）；权重转换脚本直接跳过第 61 层：

```python
# convert.py:53-54  转换权重时跳过 MTP 模块（layers.61），inference demo 不加载它
if "model.layers.61" in name:
    continue
```

即：MTP 是「训练期目标 + 可选推理加速模块」，不属于必需的前向路径，所以官方 demo 把它剥离。来源 README.md:66-67、convert.py:53。

### 5.4 FP8 混合精度

V3 原生以 FP8（`torch.float8_e4m3fn`）存权重并做训练，是首个在超大规模上验证 FP8 训练可行性的工作（来源 [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) 摘要 / README.md:73）。落点：

- dtype 切换：`model.py:760`，`Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16`。
- 量化粒度：**128×128 block scaling**（权重）+ per-token-per-128-channel 动态量化（激活）。来源 README_WEIGHTS.md:62-92。
- 在线反量化/GEMM 见 §6.3 的 `kernel.py`。

### 5.5 RoPE（YaRN 外推）与 RMSNorm

- **RoPE + YaRN**：原生 4K（`original_seq_len`）经 `rope_factor=40` 外推到 128K。`precompute_freqs_cis`（`model.py:297-375`）实现 YaRN 风格的分频修正：低频维不缩放、高频维按 factor 缩放，用 `find_correction_range` + `linear_ramp_factor` 平滑过渡（`model.py:367-370`）。长上下文时还按 `mscale` 调整 softmax 温度（`model.py:435-437`）。
- **RMSNorm**：标准实现，直接调 `F.rms_norm`（`model.py:270-294`），全模型 pre-norm。

---

## 6. 源码走读

### 6.1 关键文件清单（`forks/DeepSeek-V3/inference/`）

| 文件 | 行数 | 作用 |
|---|---|---|
| `model.py` | 808 | 全部模型定义：MLA / Gate / MoE / MLP / Block / Transformer |
| `generate.py` | 185 | 推理入口：采样、自回归生成循环、交互/批处理、分布式加载 |
| `kernel.py` | 196 | Triton 写的 FP8 算子：`act_quant` / `weight_dequant` / `fp8_gemm` |
| `convert.py` | 96 | HF 权重 → 本 demo 的分片格式（含专家分片、跳过 MTP） |
| `fp8_cast_bf16.py` | 112 | FP8 权重 → BF16（给不支持 FP8 的环境用） |
| `configs/*.json` | – | 16B / 236B / 671B / v3.1 四套超参 |

### 6.2 走读 model.py：forward 主链路

`Transformer.forward`（`model.py:772-798`）：embed → 逐 block → 取最后位置的 norm → Head 出 logits。注意它**只取序列最后一个位置**算 logits（`model.py:792`，`self.norm(h)[:, -1]`），因为这是配合 `generate.py` 的逐步解码：

```python
# model.py:784-793
h = self.embed(tokens)
freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
mask = None
if seqlen > 1:
    mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)  # 因果mask
for layer in self.layers:
    h = layer(h, start_pos, freqs_cis, mask)
h = self.norm(h)[:, -1]      # 只要最后一个 token 的表示
logits = self.head(h)
```

**MLA.forward**（`model.py:446-497`）：已在 §5.1 逐行解析（拆 nope/rope → 解耦 RoPE → absorb 吸收上投影 → 缓存压缩潜向量 → 两项 einsum 求 score → softmax → 取值 → `wo` 输出）。

**Gate.forward**（`model.py:566-598`）：已在 §5.2 解析（打分 → bias 仅用于选择 → 分组受限 topk → 权重取原始打分）。

**MoE.forward**（`model.py:669-693`）：稀疏分发。先 `bincount` 统计各专家命中数（`model.py:683`），只对本 rank 负责且被命中的专家做计算（`model.py:684-689`），最后叠加共享专家。专家在分布式下按 rank 切分（`model.py:660-666`，`n_local_experts = n_routed_experts // world_size`），未分到本 rank 的专家槽位为 `None`。

**MLP.forward / Expert.forward**：都是 SwiGLU（`w2(silu(w1(x)) * w3(x))`，`model.py:532` / `model.py:633`）。区别仅在并行策略：dense MLP 用 Column/Row Parallel 切分（`model.py:518-520`），而 MoE 专家是整块放在某个 rank 上（`Expert` 用普通 `Linear`，`model.py:619-621`），靠「专家并行」而非「张量并行」分担。

### 6.3 走读 generate.py 与 kernel.py

**生成循环**（`generate.py:60-71`）：典型 KV-cache 增量解码——每步只把新 token 喂进去（`tokens[:, prev_pos:cur_pos]`），靠 MLA 内部 cache 续接：

```python
# generate.py:60-67
for cur_pos in range(min(prompt_lens), total_len):
    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)   # 增量前向
    next_token = sample(logits, temperature) if temperature > 0 else logits.argmax(dim=-1)
    next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)  # prompt 部分不覆盖
    tokens[:, cur_pos] = next_token
```

采样函数 `sample`（`generate.py:14-27`）用 Gumbel-max 技巧（指数噪声 + argmax）实现温度采样。分布式加载权重 `model{rank}-mp{world_size}.safetensors`（`generate.py:119`），与 `convert.py` 的分片命名对应。

**FP8 GEMM**（`kernel.py`）：
- `act_quant`（`kernel.py:38-57`）：对激活做 block-wise 量化到 e4m3，scale = `amax/448`（448 是 e4m3 最大值）。V3.1 的 **UE8M0** 格式在此处现身——把 scale 量化成 2 的整数幂（`kernel.py:29-31`，`exp=ceil(log2(s)); s=exp2(exp)`），对应 `config_v3.1.json` 的 `"scale_fmt": "ue8m0"`。
- `weight_dequant`（`kernel.py:60-110`）：把 128×128 block 的 FP8 权重 × scale 还原。
- `fp8_gemm`（`kernel.py:118-196`）：autotune 的 Triton kernel，累加时按 block 乘上 a、b 两侧 scale（`kernel.py:162`，`accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]`）。

`linear()`（`model.py:131-163`）按权重 dtype 分流：BF16 权重直接 `F.linear`；FP8 权重在 `gemm_impl=="bf16"` 时先反量化再算，否则量化激活后走 `fp8_gemm`。

---

## 7. 训练机制

### 7.1 DeepSeek-V3 预训练（来源 [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) / V3 README）

- **数据**：14.8T tokens（多样高质量）。来源 README.md:50,76。
- **算力**：2.788M H800 GPU 小时（其中预训练 2.664M，后训练 0.1M），训练全程无不可恢复的 loss spike、无回滚。来源 README.md:52,76。
- **FP8 混合精度训练**：首次在超大规模验证 FP8 训练可行（细粒度 128×128 block scaling）。来源 README.md:73。
- **DualPipe 并行**：论文提出的双向流水线并行算法，让计算与跨节点 MoE all-to-all 通信**几乎完全重叠**，突破通信瓶颈。来源 README.md:74、[arXiv 2412.19437](https://arxiv.org/abs/2412.19437)。（注：DualPipe 调度逻辑不在 fork 的 inference 源码中。）
- **MTP 目标**：多 token 预测提供更密集监督信号（见 §5.3）。

### 7.2 DeepSeek-V3 后训练

- **SFT**：在多领域指令数据上监督微调。
- **RL**：进一步对齐；并**从 R1 蒸馏长链推理**能力进 V3——把 R1 的 verification/reflection 模式注入 V3，同时控制输出风格与长度。来源 README.md:80-82。

### 7.3 DeepSeek-R1：GRPO / R1-Zero / cold-start / 蒸馏（来源 `forks/DeepSeek-R1/DeepSeek_R1.pdf`）

#### GRPO 目标函数（论文 §2.2.1，公式 1-3）

GRPO（Group Relative Policy Optimization，源自 DeepSeekMath，[arXiv 2402.03300](https://arxiv.org/abs/2402.03300)）相对 PPO 的核心改动：**砍掉与策略同尺寸的 critic/value 模型，改用「组内相对得分」估计 baseline**，大幅省显存。对每个问题 $q$，从旧策略采样一组 $G$ 个输出 $\{o_1,...,o_G\}$，优化目标（论文公式 1）：

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\Big(\min\big(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i,\ \mathrm{clip}(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)},1-\varepsilon,1+\varepsilon)A_i\big) - \beta\,\mathbb{D}_{KL}(\pi_\theta\|\pi_{ref})\Big)\Big]
$$

其中优势 $A_i$ 用**组内奖励标准化**得到（论文公式 3）——这是 GRPO 的灵魂，不需要 value 网络：

$$
A_i = \frac{r_i - \mathrm{mean}(\{r_1,...,r_G\})}{\mathrm{std}(\{r_1,...,r_G\})}
$$

KL 项用无偏估计（论文公式 2，k3 estimator），始终非负。$\varepsilon$（裁剪）与 $\beta$（KL 系数）为超参。

#### R1-Zero = 纯 RL（仅规则奖励，涌现长链推理）

- 直接在 **DeepSeek-V3-Base** 上做 GRPO，**完全不做 SFT**。论文 §2.2。
- **规则奖励（rule-based reward）**，两类，**不用神经奖励模型**（避免 reward hacking）：
  - Accuracy reward：数学答案放进 `\boxed{}` 可规则校验；代码用编译器/测试用例判对错。
  - Format reward：强制把思考过程包在 `<think>...</think>` 里。论文 §2.2.2。
- **涌现**：随 RL 步数增加，response 长度自然变长（论文 Fig.3），自发出现 self-verification、reflection、探索多解；AIME 2024 pass@1 从 15.6% 升到 71.0%，majority-vote 再到 86.7%（论文 §2.2.4、Table 2）。论文记录了著名的 **"aha moment"**——模型在中间 checkpoint 学会用拟人口吻「Wait, wait... that's an aha moment」回头重审（论文 Table 3，page 9）。
- 缺陷：可读性差、中英混杂（language mixing）。论文 §2.2 / page 9。

#### R1 = cold-start SFT → RL → 拒绝采样 → 再 SFT/RL（四阶段，论文 §2.3）

1. **Cold start**：先收集**数千条**长 CoT 高质量数据，SFT 微调 V3-Base 作为 RL 初始 actor，解决 R1-Zero 冷启动不稳与可读性问题；定义 `|special_token|<reasoning>|special_token|<summary>` 输出格式。论文 §2.3.1。
2. **Reasoning-oriented RL**：同 R1-Zero 的 GRPO，聚焦数学/代码/科学/逻辑；额外加 **language consistency reward**（目标语言词占比）缓解中英混杂——虽略降准确率但更符合人类偏好，最终与 accuracy reward 直接相加。论文 §2.3.2。
3. **拒绝采样（rejection sampling）+ SFT**：RL 收敛后，用该 checkpoint 对 reasoning prompt **采样多条只留对的**（约 600k reasoning 样本，部分用 DeepSeek-V3 作 generative reward model 判对错、过滤混语言/长段/代码块）；再叠加约 200k 非推理样本（写作/事实 QA/自我认知/翻译，复用 V3 的 SFT pipeline）；合计约 **800k** 样本，对 V3-Base 训 2 个 epoch。论文 §2.3.3。
4. **全场景 RL**：第二阶段 RL 兼顾 helpfulness（只评最终 summary）与 harmlessness（评整段含 reasoning），reasoning 用规则奖励、通用用偏好奖励模型。论文 §2.3.4。

#### 蒸馏（distillation，论文 §2.4）

- 用 R1 生成的 **800k** 样本，**直接 SFT** 六个开源 dense 模型（不做 RL）：基座为 Qwen2.5-Math-1.5B/7B、Qwen2.5-14B/32B、Llama-3.1-8B、Llama-3.3-70B-Instruct。论文 §2.4 / R1 README.md:87-92。
- **关键结论（论文 §4.1）**：在 Qwen-32B-Base 上直接做大规模 RL（DeepSeek-R1-Zero-Qwen-32B，>10K 步）只能追平 QwQ-32B-Preview，而**从 R1 蒸馏的 32B 全面更强**（论文 Table 6）。说明「大模型发现的推理模式」蒸给小模型，比让小模型自己 RL 更划算——这是该论文给业界最重要的可复现经验之一。

#### 失败尝试（论文 §4.2，反面经验很有学习价值）

- **PRM（Process Reward Model）**：难定义「步」、难判中间步对错、易 reward hacking，得不偿失。
- **MCTS（蒙特卡洛树搜索）**：token 生成空间指数级大、value model 难训，自搜索迭代提升困难。

---

## 8. 学习要点与横向对比

**DeepSeek 最值得学的 5 点**：
1. **MLA**：用「低秩潜向量缓存 + 上投影吸收进 query + 解耦 RoPE」把 KV cache 砍到极小，是长上下文推理成本的范式级优化（落点 `model.py:443-487`）。
2. **auxiliary-loss-free 负载均衡**：用可学习 bias「只调选择、不调权重」实现 MoE 均衡而不伤建模目标（落点 `model.py:564,582-594`）——比传统 aux-loss 干净。
3. **GRPO**：去掉 critic、用组内相对优势做 RL，省一半显存还能跑出 o1 级推理。
4. **R1-Zero 的纯 RL 涌现**：仅靠规则奖励，长链推理与 reflection 可以「自己长出来」，刷新了「推理必须靠海量 SFT」的认知。
5. **蒸馏 > 小模型自 RL**：把强模型的推理轨迹 SFT 给小 dense 模型，性价比远超在小模型上 RL。

**与本仓库其它模型的交叉对比**：
- **[[09-minimax]]**：同为超大 MoE 路线，可对比二者的注意力（MiniMax 的 lightning/linear attention vs DeepSeek 的 MLA）与负载均衡策略。
- **[[02-qwen]]**：R1 的蒸馏小模型直接以 Qwen2.5 为基座；R1-0528 还出了 Qwen3-8B 蒸馏版——可对照 Qwen 自家的 dense/MoE 与推理后训练。
- **[[01-llama]]**：R1-Distill-Llama-8B/70B 以 Llama-3.1/3.3 为基座；可对比 LLaMA 的标准 GQA 注意力与 dense FFN，反衬 MLA + MoE 的取舍。
- **[[08-olmo]]**：OLMo 主打「全开训练流程透明」，与 DeepSeek「重权重+技术报告、但训练框架（DualPipe/FP8 调度）未完全开源」形成对照。
- **[[07-glm]]**：同为中国团队的大模型/MoE 路线，可对比工程与对齐取舍。

---

## 9. 参考

**本地源码（fork submodule）**
- `forks/DeepSeek-V3/inference/model.py`（MLA `:396-497`、Gate `:535-598`、MoE `:636-693`、Transformer `:738-798`）
- `forks/DeepSeek-V3/inference/kernel.py`（FP8：`act_quant`/`weight_dequant`/`fp8_gemm`）
- `forks/DeepSeek-V3/inference/generate.py`、`convert.py`（`:53` 跳过 MTP）、`configs/config_671B.json`、`config_v3.1.json`
- `forks/DeepSeek-V3/README.md`、`README_WEIGHTS.md`
- `forks/DeepSeek-R1/DeepSeek_R1.pdf`（GRPO 公式 §2.2.1；四阶段 pipeline §2.3；蒸馏 §2.4；失败尝试 §4.2）、`forks/DeepSeek-R1/README.md`

**官方 / 论文**
- DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL — https://arxiv.org/abs/2501.12948
- DeepSeek-V2（MLA/MoE，KV cache −93.3%）— https://arxiv.org/abs/2405.04434
- DeepSeekMath（GRPO 起源）— https://arxiv.org/abs/2402.03300
- GitHub: DeepSeek-V3 — https://github.com/deepseek-ai/DeepSeek-V3
- GitHub: DeepSeek-R1 — https://github.com/deepseek-ai/DeepSeek-R1
- GitHub: DeepSeek-V3.2-Exp（DSA）— https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- HuggingFace: deepseek-ai — https://huggingface.co/deepseek-ai
- HF: DeepSeek-R1-0528 — https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
- HF: DeepSeek-V3.1 — https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- HF: DeepSeek-V3.2-Exp — https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp

**核实备注**：V3/R1 核心规格与 benchmark 已对官方 arXiv 摘要、README、论文 PDF 核实；后续版本（R1-0528、V3.1、V3.2-Exp/V3.2）经 HuggingFace model card 与官方 GitHub 核实。各后续版本 benchmark 为 DeepSeek **自报**值，第三方复现可能有差异（已在 §4.4 标注）。V3.2-Exp 的精确公开发布日（HF 模型卡显示 2025-09 上线、arXiv 技术报告 2025-11）两处口径略有差异，已在版本表标「2025-09（实验版）」。
