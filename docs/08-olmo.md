# OLMo (AllenAI)：把"预训练全流程"完全开放的模型

> 本文聚焦 **OLMo / OLMo 2**（Allen Institute for AI，简称 AI2 / AllenAI）。OLMo 的最大价值不是某一项 benchmark，而是它把**整条训练流水线**——数据（Dolma / Dolmino）、训练代码、所有中间 checkpoint、wandb 训练日志——**全部开源**，是学习"**预训练 loop 到底怎么写**"的最佳样本。因此本文把**训练循环（train loop）**作为核心章节，逐段走读 `forks/OLMo/olmo/train.py`，并用真实 `文件:行号` 佐证。
>
> 源码 fork 位置：`forks/OLMo`（git submodule，对应 `github.com/allenai/OLMo`，本地 commit `090253d`，版本 `ai2-olmo 0.6.0`，最后提交 2025-11-24）。
>
> 横向阅读：[[01-llama]]、[[03-deepseek]]、[[06-phi]]（数据为王）、[[04-mistral]]、[[05-gemma]]。

---

## 1. 概览

**OLMo（Open Language Model）** 是 AI2 推出的"**完全开放（fully open）**"语言模型系列。与多数"开放权重（open-weights）"模型只放出权重不同，OLMo 公开了复现所需的**一切**：

- **训练数据**：Dolma（OLMo 1）、OLMo-Mix-1124 + Dolmino-Mix-1124（OLMo 2），均在 HuggingFace 上完整开放；
- **训练代码**：本仓库 `forks/OLMo`（PyTorch + FSDP），包括数据预处理、训练 loop、评测；
- **所有中间 checkpoint**：每 ~1000 步一个，可从任意点续训或做"训练动力学"研究；
- **训练日志**：每个尺寸的 wandb 报告全程公开（README 内有链接）；
- **评测与后训练**：OLMES / olmes 评测套件、Tülu 3 后训练配方全部开源。

> 定位一句话：**别人开放"结果"，OLMo 开放"过程"**。如果你想看一个工业级 LLM 预训练 loop 长什么样、超参怎么填、稳定性怎么调，OLMo 是最值得逐行读的代码库。

### 时间线与版本

| 时间 | 版本 | 要点 |
|---|---|---|
| 2024-02 | **OLMo 1**（1B / 7B），论文 [arXiv:2402.00838](https://arxiv.org/abs/2402.00838) | 首发"完全开放"，Dolma 语料，non-parametric LayerNorm、SwiGLU、RoPE | 
| 2024-04 / 07 | OLMo 1.7 / OLMo-7B-0424 / 0724 | 数据与配方迭代（MMLU 大幅提升） |
| 2024-11 | **OLMo 2**（7B / 13B），论文 [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)《2 OLMo 2 Furious》 | 架构稳定性改造（RMSNorm + reordered norm + QK-norm + Z-loss），两阶段课程 |
| 2025-03 | **OLMo 2 32B**（0325） | 首个"完全开放"且在多项学术 benchmark 上超过 GPT-3.5 Turbo / GPT-4o mini 的模型；改用新训练器 [OLMo-core](https://github.com/allenai/OLMo-core) |
| 2025-04 | **OLMo 2 1B**（0425） | 补齐小尺寸 |

> 注意（来自本仓库 README 顶部声明）：本 `OLMo` 仓库**已不再活跃维护**，最新发布迁移到 [OLMo-core](https://github.com/allenai/OLMo-core)；32B 即由 OLMo-core 训练。但 1B/7B/13B 的训练代码仍完整保留在本仓库，是本文走读的对象。

**许可证**：代码与模型均为 **Apache-2.0**（已核实，HF model cards / blog 均注明）。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| GitHub（本仓库 fork 对应） | https://github.com/allenai/OLMo |
| 新训练器（32B / 后续） | https://github.com/allenai/OLMo-core |
| 本地 fork | `forks/OLMo` |
| HuggingFace 组织 | https://huggingface.co/allenai |
| OLMo 2 7B / 13B / 32B / 1B | `allenai/OLMo-2-1124-7B`、`OLMo-2-1124-13B`、`OLMo-2-0325-32B`、`OLMo-2-0425-1B` |
| OLMo 1 论文 | https://arxiv.org/abs/2402.00838 |
| OLMo 2 论文《2 OLMo 2 Furious》 | https://arxiv.org/abs/2501.00656 |
| Dolma 语料论文 | https://arxiv.org/abs/2402.00159 |
| Tülu 3 后训练论文 | https://arxiv.org/abs/2411.15124 |
| 官方博客（OLMo 2 / 32B / 发布说明） | https://allenai.org/blog/olmo2 · https://allenai.org/blog/olmo2-32B · https://allenai.org/olmo/release-notes |
| 评测套件 | https://github.com/allenai/olmes |

---

## 3. 模型规格

下表的"架构数字"来自**本地真实训练配置**（`forks/OLMo/configs/official-*`）+ HF model cards 交叉核对；token 规模来自官方 blog/README（已核实）。

| 规格 | OLMo 1 7B | OLMo 2 1B | OLMo 2 7B | OLMo 2 13B | OLMo 2 32B |
|---|---|---|---|---|---|
| 参数量 | 7B | 1B | 7B | 13B | 32B |
| 层数 n_layers | 32 | 16 | 32 | 40 | 64 |
| 隐藏维度 d_model | 4096 | 2048 | 4096 | 5120 | 5120 |
| 注意力头数 | 32 | 16 | 32 | 40 | 40 |
| MLP 隐藏维度 | 22016 | 16384（mlp_ratio=8）| 22016 | 13824（约）| — |
| 上下文长度 | 2048 | 4096 | 4096 | 4096 | 4096 |
| 位置编码 | RoPE | RoPE θ=500000 | RoPE θ=500000 | RoPE θ=500000 | RoPE |
| 归一化 | 非参数化 LayerNorm | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| norm 位置 | 前置（pre-norm）| 后置（reordered）| 后置（reordered）| 后置（reordered）| 后置（reordered）|
| QK-norm | 否 | 是 | 是 | 是 | 是 |
| 激活 | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| bias | 无 | 无 | 无 | 无 | 无 |
| 词表 / tokenizer | 50280（GPT-NeoX 改）| 100278（Dolma2）| 100278（Dolma2）| 100278（Dolma2）| Dolma2 |
| 预训练 tokens | ~2.46T | Stage1 4T + Stage2 ~50B | Stage1 ~3.9T + Stage2 50B | Stage1 5T + Stage2 ~400B | ~6T + mid-train |

来源：`configs/official-1124/OLMo2-7B-stage1.yaml`、`configs/official-0425/OLMo2-1B-stage1.yaml`、`configs/official-0724/OLMo-7B.yaml`；HF `allenai/OLMo-2-1124-7B`、`allenai/OLMo-2-0325-32B`；OLMo 1 论文 2402.00838。

> 13B 的 MLP 隐藏维度（13824）来自 HF card 推断，标"约"。OLMo 2 7B/13B 均为 **MHA（无 GQA）**：config 中 `n_kv_heads` 默认等于 `n_heads`（`olmo/config.py:252`），7B stage1 yaml 也未设置 `n_kv_heads`。

---

## 4. Benchmark

以下为 **OLMo 2 base（预训练后、未指令微调）** 官方报告分数（来自 `allenai/OLMo-2-1124-7B` model card 与官方 blog，已核实）：

| Benchmark | OLMo 2 7B | OLMo 2 13B | OLMo 2 32B |
|---|---|---|---|
| MMLU | 63.7 | 67.5 | 74.9 |
| GSM8K | 67.5 | 75.1 | 61.0* |
| HellaSwag | 83.8 | 86.4 | 89.7 |
| ARC-Challenge | 79.8 | 83.5 | 90.4 |
| WinoGrande | 77.2 | 81.5 | — |
| DROP | 60.8 | 70.7 | 74.3 |
| NaturalQuestions | 36.9 | 46.7 | — |
| AGIEval | 50.4 | 54.2 | — |
| TriviaQA | — | — | 88.0 |

\* 32B 的 GSM8K 值在 HF 评测表中为 61.0（与 7B/13B 的统计口径/few-shot 设置可能不同，标注备查）。32B 平均分 72.9（HF card）。

来源：https://huggingface.co/allenai/OLMo-2-1124-7B · https://huggingface.co/allenai/OLMo-2-0325-32B · https://allenai.org/blog/olmo2

官方定位（已核实）：
- OLMo 2 7B/13B 是"**当时最好的完全开放模型**"，常优于同尺寸 open-weight 模型；
- OLMo 2 32B 是"**首个在一套多技能学术 benchmark 上超过 GPT-3.5 Turbo 与 GPT-4o mini 的完全开放模型**"，并逼近 Qwen 2.5 32B、Mistral 24B（来源：https://allenai.org/blog/olmo2-32B）。

> 指令微调（Instruct）变体走 Tülu 3 配方，见 §7.2；其 GSM8K / IFEval / MATH 等会显著高于 base。

---

## 5. 架构剖析：OLMo 1 → OLMo 2 逐点对比

OLMo 的架构主体是一个标准 decoder-only Transformer，但**所有结构选项都做成了 config 开关**（`olmo/config.py: ModelConfig`），同一份 `model.py` 既能搭出 OLMo 1 也能搭出 OLMo 2。这正是它适合学习的地方——把"架构演进"变成几个布尔/枚举开关的对比。

下表把 OLMo 2 相对 OLMo 1 的关键改动，对应到**配置项**与**代码行**：

| 维度 | OLMo 1 | OLMo 2 | 配置项 | 代码佐证 |
|---|---|---|---|---|
| 归一化类型 | 非参数化 LayerNorm | **RMSNorm**（带 affine）| `layer_norm_type` | `model.py:228` `RMSLayerNorm` |
| norm 位置 | pre-norm（attn/ffn 之前）| **reordered**：attn/ffn 之**后** | `norm_after` | `model.py:746` / `786` / `814` |
| QK-norm | 无 | **对 Q、K 做 norm** | `attention_layer_norm` | `model.py:432-441`、应用于 `:603-605` |
| 训练损失 | 仅交叉熵 | **+ Z-loss** | `softmax_auxiliary_loss` | `train.py:137-143` |
| 位置编码 | RoPE（θ=10000）| RoPE（**θ=500000**）| `rope_theta` | `model.py:287` |
| tokenizer | GPT-NeoX 改（50280）| **Dolma2**（100278）| `vocab_size` | yaml `vocab_size` |
| 初始化 | mitchell | normal（std=0.02）| `init_fn` | `model.py:489-503` |

下面逐项看代码。

### 5.1 归一化：RMSNorm vs 非参数化 LayerNorm

OLMo 1 7B 用的是**非参数化 LayerNorm**（`layer_norm_type: default` + `layer_norm_with_affine: false`，见 `configs/official-0724/OLMo-7B.yaml:24-25`）——即不带可学习的 gain/bias。OLMo 2 改用 **RMSNorm**（`layer_norm_type: rms`，`OLMo2-7B-stage1.yaml:18`）。RMSNorm 实现：

```python
# olmo/model.py:241  RMSLayerNorm.forward —— 只用均方根，不减均值，比 LayerNorm 更省、更稳
def forward(self, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(enabled=False, device_type=x.device.type):
        og_dtype = x.dtype
        x = x.to(torch.float32)                       # 关键：在 fp32 下算归一化，避免 bf16 数值问题
        variance = x.pow(2).mean(-1, keepdim=True)    # 均方（不含 mean 中心化）
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(og_dtype)
    if self.weight is not None:                        # OLMo2 带 affine（weight），OLMo1 default 不带
        return self.weight * x
    return x
```

### 5.2 reordered norm（OLMo 2 的"后置归一化"）

这是 OLMo 2 稳定性提升的核心之一。标准 pre-norm 块是 `x + Attn(LN(x))`；OLMo 2 改成"**先做子层、再 norm、再加残差**"，即 `x + LN(Attn(x))`。注意 OLMo 2 的 `block_type` 仍是 `sequential`（不是 `llama`），靠 `norm_after: true` 切换。代码用同一个 `OLMoSequentialBlock.forward` 通过 `if not self.config.norm_after` / `if self.config.norm_after` 两套分支实现：

```python
# olmo/model.py:744-794  （OLMoSequentialBlock.forward 片段）
# apply norm before —— OLMo1 路径：先 norm 再投影
if not self.config.norm_after:
    h = self.attn_norm(x)
else:
    h = x                                    # OLMo2：先不 norm，原样投影
qkv = self.att_proj(h)
...
att, cache = self.attention(q, k, v, attention_bias, ...)
if self.config.norm_after:                   # OLMo2：在 attention 输出之后才 norm
    att = self.attn_norm(att)
x = x + self.dropout(att)                     # 残差相加
```

FFN 侧同理（`model.py:800-821`）：OLMo 1 是 `ff_norm` 在 `ff_proj` 之前；OLMo 2 是 `ff_out` 之后才 `ff_norm`，再加回 `og_x`。这种"把 norm 移到残差分支末端"的做法（论文称 reordered norm，引 Liu et al. 2022），让残差流保持未归一化的尺度，配合 RMSNorm + QK-norm 显著降低了大模型训练中的 loss 尖峰。

### 5.3 QK-norm（对 Query / Key 归一化）

OLMo 2 开启 `attention_layer_norm: true`，在算注意力分数前，对 Q、K 各做一次 LayerNorm/RMSNorm。这能抑制注意力 logits 爆炸，是大模型稳定训练的常见技巧（论文引 Dehghani et al. 2023）。构造：

```python
# olmo/model.py:432-441  —— attention_layer_norm 开关下建 k_norm / q_norm
self.k_norm: Optional[LayerNormBase] = None
self.q_norm: Optional[LayerNormBase] = None
if config.attention_layer_norm:
    self.k_norm = LayerNormBase.build(
        config, size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
        elementwise_affine=config.attention_layer_norm_with_affine)
    self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)
```

应用处（注意是在 reshape 成多头**之前**，对整段投影做 norm）：

```python
# olmo/model.py:603-605  （OLMoBlock.attention 片段）
if self.q_norm is not None and self.k_norm is not None:
    q = self.q_norm(q).to(dtype=dtype)
    k = self.k_norm(k).to(dtype=dtype)
```

### 5.4 RoPE、SwiGLU、无 bias

- **RoPE**：`RotaryEmbedding`（`model.py:258`）。`inv_freq = 1/θ^(...)`，OLMo 2 把 `rope_theta` 从默认 10000 提到 **500000**（`model.py:287`；yaml `rope_theta: 500000`），以更好支持 4096 长上下文。默认 `rope_full_precision=True`，旋转在 fp32 下计算（`model.py:308-309`）。
- **SwiGLU**：`ff_proj` 输出维度是 `mlp_hidden_size`，经 `SwiGLU` 后 chunk 成两半、`silu(gate)*x`，所以实际 FFN 宽度需要乘 `output_multiplier`（=2/3，见 `model.py:365-372`）。注意 OLMo 把 SwiGLU 的"门"和"值"放在**一个** `ff_proj` 里再 chunk，而非两个独立矩阵。
- **无 bias**：所有 `nn.Linear` 的 `bias=config.include_bias`，OLMo 全系列 `include_bias: false`（`model.py:452-462`、`695-701`）。
- **注意力实现**：优先 FlashAttention（`flash_attn_func`），否则回退 `F.scaled_dot_product_attention`；由于 torch SDPA 不支持 GQA，代码手动 `repeat_interleave` 把 KV 头扩展到 Q 头数（`model.py:570-577`）。
- **权重不绑定**：OLMo 全系列 `weight_tying: false`，输出投影是独立的 `transformer.ff_out`（`model.py:1129-1139`、`1454-1457`）。

### 5.5 最终输出与 logits

```python
# olmo/model.py:1447-1459
x = self.transformer.ln_f(x)                          # 最终 LayerNorm/RMSNorm
if self.config.weight_tying:
    logits = F.linear(x, self.transformer.wte.weight, None)
else:
    logits = self.transformer.ff_out(x)               # OLMo：独立输出头
if self.config.scale_logits:
    logits.mul_(1 / math.sqrt(self.config.d_model))   # 可选 logit 缩放（OLMo2 默认关）
```

> HF 推理封装：`hf_olmo/modeling_olmo.py:41` 的 `OLMoForCausalLM` 只是把原生 `OLMo`（`from olmo.model import OLMo`，`modeling_olmo.py:12`）包进 `PreTrainedModel`，forward 里做 shift-by-one 的交叉熵（`modeling_olmo.py:119-141`）。也就是说 **HF 版与训练版共用同一套 `model.py`**，没有重写一遍，复现一致性强。（注：OLMo 2 在最新版 `transformers` 里也有官方 `Olmo2` 实现，本仓库 `hf_olmo` 是早期通用封装。）

---

## 6. 源码走读：训练循环（核心）

这一节是本文重点。OLMo 的训练入口是 `scripts/train.py`，真正的训练逻辑在 `olmo/train.py` 的 `Trainer` 类（`train.py:209`）。下面按"**从入口到一个 train step 再到 fit 主循环**"的顺序逐段走读。

### 6.0 入口：`scripts/train.py`

启动方式（README）：`torchrun --nproc_per_node=8 scripts/train.py {config.yaml}`。入口干三件大事：

1. **建数据加载器**：`train_loader = build_train_dataloader(cfg)`（`scripts/train.py:132`）；
2. **建模型并包 FSDP**：`olmo_model = OLMo(cfg.model)`（`:140`）→ 设激活检查点（`:152`）→ 按 `distributed_strategy` 包成 DDP / **FSDP** / 单卡（`:154-226`）；
3. **建优化器 + 调度器 + Trainer**：`build_optimizer` / `build_scheduler`（`:237-238`），再塞进 `Trainer`（`:250`）调用 `trainer.fit()`。

FSDP 包装（`scripts/train.py:210-220`）：

```python
# scripts/train.py:210  —— FSDP 分片包装
dist_model = FSDP(
    olmo_model,
    sharding_strategy=cfg.fsdp.sharding_strategy,    # FULL_SHARD / HYBRID_SHARD ...
    mixed_precision=cfg.fsdp_precision,              # 混合精度（bf16）
    auto_wrap_policy=wrap_policy,                    # 按 block 包，见 get_fsdp_wrap_policy
    use_orig_params=cfg.fsdp.use_orig_params,        # 为 torch.compile 与逐参数 metric 所需
    limit_all_gathers=True,
    device_id=get_local_rank(),
    param_init_fn=param_init_fn,                     # init_device=meta 时 FSDP 负责真正初始化
)
```

包装粒度由 `OLMo.get_fsdp_wrap_policy`（`model.py:1467`）决定，默认 `by_block_and_size`（yaml `wrapping_strategy`）——**每个 Transformer block 单独作为一个 FSDP unit** 分片，配合 `init_device: meta` 实现"参数不在主机上物化两次"。

### 6.1 损失函数：交叉熵 + Z-loss

训练损失定义在模块级 `cross_entropy_loss`（`train.py:124`）。Z-loss 是 OLMo 2 稳定性三件套的最后一件：

```python
# olmo/train.py:124-145  cross_entropy_loss
def cross_entropy_loss(logits, labels, ignore_index=-100, reduction="mean",
                       compute_z_loss=False, z_loss_multiplier=1e-4):
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    if not compute_z_loss:
        return loss, None
    z_squared = logits.logsumexp(-1).pow(2)            # Z-loss = (logsumexp of logits)^2
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    z_loss = z_loss_multiplier * z_squared             # OLMo2 用 multiplier=1e-5
    return loss, z_loss
```

**Z-loss 的作用**：交叉熵只约束"正确类的相对 logit"，softmax 分母（配分函数 Z）可以漂移到很大/很小。Z-loss 惩罚 `logsumexp(logits)^2`，把 Z 拉回 1 附近，防止 logits 整体爆炸——这是 PaLM 引入、OLMo 2 采用的关键稳定技巧。配置入口：`softmax_auxiliary_loss`（`config.py:1244`）+ `auxiliary_loss_multiplier`（`config.py:1250`，OLMo2 yaml 设 `1e-5`）。若装了 flash-attn，还会走 fused 的 Triton 版（`train.py:156` `fused_loss_fn`，把 z-loss 作为 `lse_square_scale`）。

### 6.2 一次前向：`model_forward`

```python
# olmo/train.py:731-757  model_forward（计算 logits + 损失）
logits = self.dist_model(input_ids=batch["input_ids"], attention_mask=..., ...).logits
logits_for_loss = logits[..., :-1, :].contiguous()       # 去掉最后一个位置（无 label）
logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
labels = self.get_labels(batch)                          # 见下：input 左移一位
labels = labels.view(-1)
ce_loss, z_loss = self.loss_fn(
    logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction, compute_z_loss=compute_z_loss)
```

label 构造（`train.py:715-729`）：把 `input_ids` 克隆后左移一位（`labels[..., 1:]`），并把 `label_mask` / `attention_mask`（padding）/ `instance_mask`（被过滤的样本）对应位置填 `-100` 让交叉熵忽略。这就是标准的 next-token 监督。

### 6.3 micro-batch 与梯度累积：`train_micro_batch` / `train_batch`

OLMo 用 **micro-batch 累积** 实现大 batch（`device_train_microbatch_size=2`，但 `global_train_batch_size=1024`）。`train_batch` 把一个 batch 切成多个 micro-batch（`split_batch`，`train.py:934`），逐个前向+反向、累积梯度：

```python
# olmo/train.py:759-780  train_micro_batch —— 注意损失按"整个 batch 的 token 数"归一
ce_loss, z_loss, logits = self.model_forward(
    micro_batch, compute_z_loss=self.cfg.softmax_auxiliary_loss, loss_reduction="sum")
ce_loss = ce_loss / batch_size_in_tokens               # sum 后除以总 token 数 = 正确的 token 级均值
if self.cfg.softmax_auxiliary_loss:
    z_loss = z_loss / batch_size_in_tokens
    loss = ce_loss + z_loss                            # 最终被优化的 loss = CE + Z
else:
    loss = ce_loss
```

```python
# olmo/train.py:809-824  train_batch 的反向（每个 micro-batch 各自 backward 累积梯度）
with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):  # bf16
    loss, ce_loss, z_loss = self.train_micro_batch(micro_batch, batch_size_in_tokens)
    ce_batch_loss += ce_loss.detach()
    if z_loss is not None:
        z_batch_loss += z_loss.detach()
loss.backward()                                        # 梯度累积：不在这里 step
```

**要点**：因为损失用 `reduction="sum"` 再除以 `batch_size_in_tokens`（整个 global batch 的 token 数），多个 micro-batch 的梯度直接相加就等价于在大 batch 上算 token 级平均，不需要再额外缩放——这是写梯度累积时最容易错的地方，OLMo 处理得很干净。autocast 用 bf16（`precision: amp_bf16`）。

### 6.4 一个完整 step：`train_step`（梯度裁剪 → 调 LR → optimizer.step）

`train_step`（`train.py:832`）是整条 loop 的浓缩。按顺序：

```python
# olmo/train.py:844-887  train_step 主干（精简）
self.optim.zero_grad(set_to_none=True)                 # 1. 清梯度
batch = move_to_device(batch, self.device)
ce_batch_loss, z_batch_loss = self.train_batch(batch)  # 2. 前向+反向（含累积）

if reduce_global_loss:                                  # 3. 跨 rank 汇报 loss（仅日志步）
    dist.reduce(ce_batch_loss, 0); ce_batch_loss.div_(get_world_size())

# 4. 梯度裁剪 + 收集 optim 指标（global grad-norm clipping）
optim_metrics = self.optim.clip_grads_and_collect_metrics(
    self.global_step, collect_param_metrics=..., process_group=self.dist_model.process_group)

# 5. 按调度器更新本步 LR 与 max_grad_norm
for group in self.optim.param_groups:
    group["lr"] = self.scheduler.get_lr(
        self.cfg.optimizer.learning_rate, self.scheduler_current, self.scheduler_max)
    group["max_grad_norm"] = self.scheduler.get_max_grad_norm(self.cfg.max_grad_norm, ...)

self.optim.step()                                      # 6. 参数更新（AdamW）
if torch.isnan(ce_batch_loss):                         # 7. NaN 守卫：炸了就抛错停训
    raise ValueError("nan loss encountered")
```

几个值得学的细节：
- **LR 是每步现算的**：调度器不是 PyTorch 的 `LRScheduler`，而是 OLMo 自己的 `Scheduler.get_lr(initial_lr, step, max_steps)`，每步把结果写回 `group["lr"]`（`train.py:872-878`）。好处是续训时不依赖优化器内部状态，按 `global_step` 完全可复现。
- **梯度裁剪是全局范数裁剪**，且**裁剪阈值本身也能 warmup**（OLMo 1 用 `grad_clip_warmup_factor: 10.0` 前 1000 步放宽阈值，`OLMo-7B.yaml:55-56`）。实现见 `optim.py:300-351`：算全局 `total_grad_norm`，`clip_coef = max_norm/(norm+1e-6)`，再 `clamp(max=1.0)` 乘回梯度——用 clamp 而非 `if<1` 是为了避免 host-device 同步。
- **`reduce_global_loss` 只在需要日志的步做**（`train.py:854`、`fit` 里 `should_log_this_step`），平时不做跨卡 all-reduce，省通信。
- **NaN 守卫**：loss 出现 NaN 直接 `raise`（`train.py:891-894`），让坏 run 立刻失败而不是悄悄跑废。

### 6.5 主循环：`fit()`

`fit()`（`train.py:1109`）把上面的 `train_step` 套进 epoch/batch 双层循环，并在固定间隔做 checkpoint / 评测 / 日志。骨架：

```python
# olmo/train.py:1196-1333  fit 主循环（高度精简）
for epoch in range(self.epoch or 0, self.max_epochs):
    for batch in self.train_loader:
        self.global_step += 1
        self.global_train_tokens_seen += global_batch_size * seq_len      # 以 token 计进度
        metrics = self.train_step(batch, reduce_global_loss=should_log_this_step)  # ← 一步训练
        ...                                                                # 日志 / wandb.log
        # 周期性存 sharded checkpoint（FSDP）
        if self.global_step % self.cfg.save_interval == 0:
            self.save_checkpoint(CheckpointType.sharded)
        # 周期性存 unsharded（用于发布 / HF 转换）
        if self.global_step % self.cfg.save_interval_unsharded == 0:
            self.save_checkpoint(CheckpointType.unsharded)
        # in-loop 下游评测
        if self.global_step % self.cfg.eval_interval == 0 or self.global_step >= stop_at:
            eval_metrics = self.eval(); wandb.log(eval_metrics, step=self.global_step)
            self.dist_model.train()                                       # 评完切回 train 模式
        if self.global_step >= stop_at:
            break
    else:                                                                  # 一个 epoch 跑完
        self.dataset.reshuffle(self.epoch)                                # 换 seed 重新洗牌
        continue
    break
```

**学习点**：
- **进度以 token 计**：`global_train_tokens_seen`，调度器 `units: tokens`（yaml）、`t_warmup`/`t_max` 都是 token 数（`OLMo2-7B-stage1.yaml:59-60` 的 `t_warmup: 8388608000`≈8.4B、`t_max: 5e12`），比"以 step 计"更稳健（换 batch size 不变曲线）。
- **三种 checkpoint**：sharded（FSDP 分片，最适合断点续训）、ephemeral（短期、会被清理）、unsharded（合并成完整权重，用于发布/转 HF）。这就是"每 1000 步一个可用 checkpoint"的来源（`save_interval: 1000`）。
- **in-loop 评测**：`eval()`（`train.py:1011`）在训练途中跑 `Evaluator`（perplexity + 一大批下游任务，见 7B yaml 第 135-214 行的 piqa/hellaswag/arc/mmlu... 列表），分数实时写 wandb——这就是 OLMo"训练日志全开"里那些曲线的来源。
- **可取消 / 可续训**：`check_if_cancelled`（`train.py:1057`）支持优雅停机并存最终 checkpoint；epoch 末 `reshuffle(epoch)` 用新 `seed+epoch` 重洗，保证多 epoch 数据顺序确定且可复现。
- **FSDP 不喜欢自动 GC**：`fit` 开头 `gc.disable()`，改为按 `gen1_gc_interval` 手动 `gc.collect(1)`（`train.py:1122-1123`、`1336-1337`）——这是大规模 FSDP 训练的实战经验。

### 6.6 数据加载：memory-mapped + 全局索引 + 可复现 shuffle

数据是 OLMo "可复现" 的基石。预训练数据被预处理成**连续 token 的 `.npy` 文件**，用 `MemMapDataset`（`olmo/data/memmap_dataset.py:20`）以 **numpy memmap** 方式按 `chunk_size`（=max_seq_len）切片读取——不把整个语料读进内存，按需 mmap：

```python
# olmo/data/memmap_dataset.py:157-161  按偏移直接从 memmap 切一个训练样本
def _read_chunk_from_memmap(self, path, index, dtype=None):
    dtype = dtype or self.dtype                          # 默认 uint16；OLMo2 用 uint32（词表>65536）
    item_size = dtype(0).itemsize
    bytes_start = index * item_size * self._chunk_size   # 第 index 个 chunk 的字节起点
    num_bytes = item_size * self._chunk_size
```

`build_train_dataloader`（`olmo/data/__init__.py:127`）把多个 memmap 文件拼成一个逻辑大数据集，再包一层 `IterableDataset` 做分布式采样：

```python
# olmo/data/__init__.py:156-167
dataset = IterableDataset(
    dataset, train_config.global_train_batch_size,
    seed=seed, epoch=train_config.epoch or 0, shuffle=True,
    drop_last=train_config.data.drop_last,
    world_size=world_size, rank=rank, work_dir=work_dir)   # work_dir 用来落盘"全局索引"
```

**可复现 shuffle** 的关键（`olmo/data/iterable_dataset.py:91-114`）：用 numpy PCG64、`seed = self.seed + self.epoch` 对 `[0..N)` 全局打乱，并把这个**全局索引数组落盘成 `global_indices.npy`**（`iterable_dataset.py:75-88`），各 rank 再从中按 `rank::world_size` 切自己那份、按 `start_index` 跳过已训部分：

```python
# olmo/data/iterable_dataset.py:91-98
indices = np.arange(len(self.dataset), dtype=np.uint32)
if self.shuffle:
    rng = np.random.Generator(np.random.PCG64(seed=self.seed + self.epoch))  # 确定性洗牌
    rng.shuffle(indices)
```

这套设计带来三个性质：(1) **数据顺序完全确定**（给定 seed+epoch）；(2) **可从任意 step 续训**（`start_index` 跳过即可，`iterable_dataset.py:23-24` 的 docstring 明说支持任意点重启）；(3) **可审计**——`train_step` 还能把每步用到的样本 index 写进 `data-indices/*.tsv.gz`（`train.py:836-838` + `scripts/train.py:242-247`）。这就是"OLMo 能精确说出第 N 步喂了哪些数据"的实现。

### 6.7 配置系统：`olmo/config.py`

所有超参都是 dataclass（`config.py`），yaml 直接反序列化，命令行可 `--a.b.c=value` 覆盖（README 示例）。关键类：`ModelConfig`（`config.py:235`，架构）、`OptimizerConfig`（`:505`）、`SchedulerConfig`（`:570`）、`TrainConfig`（`:948`，总配置）。前面架构对比表里的每个开关都在 `ModelConfig` 里有对应字段（`d_model:242`、`norm_after:472`、`attention_layer_norm:339`、`rope_theta:319`、`include_bias:385`…）。

### 6.8 优化器与 LR 调度：`olmo/optim.py`

- **优化器**：`build_optimizer`（`optim.py:942`）支持 `adamw`（`AdamW`，`optim.py:506`，继承 `torch.optim.AdamW`）与 `lionw`（`LionW`，`optim.py:365`）。OLMo 全系列实际用 **AdamW**（β=(0.9, 0.95)，wd=0.1，eps=1e-8；见 7B yaml）。`get_param_groups`（`optim.py:829`）按 `decay_embeddings`/`decay_norm_and_bias` 把参数分成"加权重衰减"和"不加"两组——OLMo2 对 embedding 不做 wd（`decay_embeddings: false`）。
- **LR 调度**：`Scheduler`（`optim.py:651`）是自定义抽象类。OLMo 1 用 `LinearWithWarmup`（`optim.py:713`），OLMo 2 用 `CosWithWarmup`（`optim.py:694`）：

```python
# olmo/optim.py:699-709  CosWithWarmup.get_lr —— warmup 后余弦退火到 alpha_f*lr
def get_lr(self, initial_lr, step, max_steps):
    max_steps = max_steps if self.t_max is None else self.t_max
    eta_min = initial_lr * self.alpha_f                 # 最低 LR = 0.1*lr（alpha_f=0.1）
    if step < self.warmup_steps:
        return self._linear_warmup(initial_lr, step, self.warmup_steps)  # 线性 warmup
    elif step >= max_steps:
        return eta_min
    else:
        step = step - self.warmup_steps
        max_steps = max_steps - self.warmup_steps
        return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2
```

> **Stage 2（退火/microanneal）就靠调度器实现**：stage2 配置把调度器换成 `linear_with_warmup` 且 `t_warmup: 0`、`alpha_f: 0`（`OLMo2-7B-stage2-seed42.yaml:56-59`），即从 stage1 末尾的 LR **线性衰减到 0**，配合高质量数据"打磨"模型。这正是 §7 课程的代码落点。

---

## 7. 训练机制

### 7.1 预训练数据与两阶段课程

OLMo 的预训练数据本身就是开放成果：

- **OLMo 1**：**Dolma**（[arXiv:2402.00159](https://arxiv.org/abs/2402.00159)，3T-token 开放语料）的 ~2T-token 采样；7B 最终训到 **~2.46T tokens**，线性 LR 衰减（已核实，OLMo 1 论文 / HF card）。tokenizer 为 GPT-NeoX-20B 改造版（加 PII mask token），词表 50280→50304（凑 128 倍数提吞吐）。
- **OLMo 2 两阶段课程**（已核实，blog/README）：
  - **Stage 1（占算力 ~90%）**：在 **OLMo-Mix-1124**（~3.9T tokens，来自 DCLM、Dolma、Starcoder、Proof Pile II）上训练。7B 约 1 个 epoch（~3.9–4T），13B 训 1.2 epoch 到 **5T**，32B 训 ~1.5 epoch 到 **~6T**。
  - **Stage 2（退火 / microanneal）**：在 **Dolmino-Mix-1124**（高质量定向数据，总 843B tokens，采样成 50B/100B/300B 三种 mix）上，**从 stage1 checkpoint 把 LR 线性退火到 0**。7B 用 50B mix 训 3 份；13B 用 100B×3 + 300B×1；并把多份结果**模型平均（"souping"，权重直接平均）** 得到最终模型（README §"Stage 2"）。

> 这一"大规模网页打底 + 后期高质量退火"的两段式，是 OLMo 2 相对 OLMo 1 最重要的**数据课程**创新，与 [[06-phi]] 的"数据质量为王"思路异曲同工，但 OLMo 把每一阶段的数据、配置、checkpoint 都开放了，可逐段复现。

**tokenizer**：OLMo 2 用 **Dolma2 tokenizer**（`tokenizers/allenai_dolma2.json`，yaml `tokenizer.identifier`），词表 100278（embedding padding 到 100352）。

### 7.2 关键超参（来自真实 yaml）

| 超参 | OLMo 1 7B | OLMo 2 7B（stage1）| 来源 |
|---|---|---|---|
| optimizer | AdamW | AdamW | yaml `optimizer.name` |
| LR（峰值）| 3.0e-4 | 3.0e-4（1B 用 4.0e-4）| `learning_rate` |
| betas | (0.9, 0.95) | (0.9, 0.95) | `betas` |
| weight decay | 0.1 | 0.1 | `weight_decay` |
| LR schedule | linear + warmup | cosine + warmup（→0.1×）| `scheduler.name` |
| warmup | 5000 steps | ~8.4B tokens | `t_warmup` |
| grad clip | 1.0（前 1000 步 ×10）| 1.0 | `max_grad_norm` |
| global batch | 2048 | 1024 | `global_train_batch_size` |
| micro-batch / dev | — | 2 | `device_train_microbatch_size` |
| 精度 | amp | **amp_bf16** | `precision` |
| Z-loss | 无 | 有，mult=1e-5 | `softmax_auxiliary_loss` |
| 并行 | FSDP | FSDP（by_block_and_size）| `fsdp.wrapping_strategy` |

### 7.3 并行与稳定性技巧小结

- **并行**：FSDP（默认 FULL_SHARD，可选 HYBRID_SHARD 多副本，`scripts/train.py:188-208`），按 block 包装；bf16 混合精度；`init_device: meta` 让 FSDP 分配时不双重物化参数；可选 `torch.compile`（逐 block 编译，`scripts/train.py:149-150`）。激活检查点：`set_activation_checkpointing`（`model.py:1154`），策略见 `should_checkpoint_block`（`model.py:91`），可按 block 粒度重算以省显存。
- **稳定性五件套**（OLMo 2 论文明列，已核实）：① RMSNorm；② reordered norm（后置）；③ QK-norm；④ Z-loss；⑤ 更好的初始化（保持激活/梯度尺度）。再加上 **全局梯度范数裁剪 + 裁剪 warmup**、**NaN 守卫**、**RoPE/归一化在 fp32 计算**，共同把大模型 loss 尖峰压下去。

### 7.4 后训练：Tülu 3 配方（SFT → DPO → RLVR）

OLMo 2 的 Instruct 变体直接套用 AI2 自家的 **Tülu 3** 后训练流水线（[arXiv:2411.15124](https://arxiv.org/abs/2411.15124)，已核实）：

1. **SFT**（监督微调）：在指令-回答数据上微调；
2. **DPO**（Direct Preference Optimization）：用偏好对做对齐；
3. **RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励的强化学习）**：复用 RLHF 的目标，但**把奖励模型换成"确定性验证函数"**——对数学、可验证的指令遵循等任务，答案对则奖励、错则 0，再用 **PPO** 优化（Tülu 3 原始配方）。OLMo 2 32B 则用了 Tülu 3.1 的 **GRPO**（Group Relative Policy Optimization）变体。

> 本仓库提供 `scripts/prepare_tulu_data.py` 做 Tülu 数据准备；RLVR 与 GRPO 的完整训练代码在 Tülu / open-instruct 生态（与本预训练仓库分离）。RLVR 与 DPO 的本质区别：DPO 靠"成对偏好"，RLVR 靠"客观可验证的对错信号"，因此能在 GSM8K 这类有 ground-truth 的任务上做定向提升。

---

## 8. 学习要点与横向对比

**对"想看真实训练代码"的人，OLMo 价值最大。** 多数开放模型只给你权重和一篇论文；OLMo 给你**一份能直接 `torchrun` 跑起来的工业级预训练 loop**，且数据/日志/checkpoint 全可对照。要点：

1. **训练 loop 的标准结构**（`train.py`）：`fit` 双层循环 → `train_step`（zero_grad → 前向反向累积 → reduce loss → 全局 grad-norm 裁剪 → 现算 LR → AdamW step → NaN 守卫）→ 周期性 checkpoint / in-loop eval。这套结构几乎可直接迁移到自己的训练脚本。
2. **梯度累积怎么写对**：用 `reduction="sum"` 再除以 global-batch token 数，多 micro-batch 梯度直接相加即等价 token 级均值（`train.py:765`）。
3. **可复现性怎么落地**：确定性 memmap 数据 + 落盘全局索引 + `start_index` 续训 + 每步 data-indices 审计（§6.6）。
4. **大模型稳定性怎么调**：RMSNorm + reordered norm + QK-norm + Z-loss + grad-clip warmup + NaN 守卫 + fp32 归一化（§5、§7.3）——OLMo 2 把这些做成可单独开关的实验（CHANGELOG line 60-61：专门加了 qk_norm / norm reordering / zloss 的实验脚本）。
5. **架构即配置**：同一份 `model.py` 靠 `norm_after`/`attention_layer_norm`/`layer_norm_type` 等开关在 OLMo 1 与 OLMo 2 间切换，是读"架构演进"的活教材。

**横向对比**：

| 维度 | OLMo 2 | 对比 |
|---|---|---|
| 开放程度 | 数据+代码+checkpoint+日志全开 | 高于 [[01-llama]]（开权重+代码，数据不全开）、[[03-deepseek]]（开权重+技术报告）、[[06-phi]]（开权重，数据闭源） |
| 架构稳定性 | RMSNorm+reorder+QK-norm+Z-loss | 与 [[03-deepseek]]、[[01-llama]] 的 pre-norm RMSNorm 同属"稳定训练"路线，但 reordered norm 较少见 |
| 数据课程 | 网页打底 + 高质量退火两阶段 | 与 [[06-phi]]"数据质量为王"理念相通，但 OLMo 把数据完全开放、可复现 |
| 注意力 | MHA（7B/13B）| 对比 [[01-llama]]/[[03-deepseek]] 的 GQA/MLA，OLMo 1B-13B 走更朴素的 MHA |
| 后训练 | Tülu 3：SFT+DPO+RLVR(PPO/GRPO) | RLVR 是其特色，与 [[03-deepseek]] R1 的 RL-for-reasoning 思路相邻但奖励来源不同（验证函数 vs 规则/RM）|

> 一句话总结：**要学预训练全流程，先读 OLMo 的 `train.py`；要学"数据决定上限"，对照着读 [[06-phi]]；要学规模化架构与 RL，对照 [[01-llama]] 与 [[03-deepseek]]。**

---

## 9. 参考

**源码（本地 fork，`forks/OLMo`）**
- 架构：`olmo/model.py`（RMSNorm `:228`、RoPE `:258`、SwiGLU `:365`、OLMoBlock `:411`、QK-norm `:432`/`:603`、Sequential/reordered-norm `:678`/`:746`、forward `:1253`、FSDP wrap `:1467`）
- 训练循环：`olmo/train.py`（Z-loss `:124`、model_forward `:731`、train_micro_batch `:759`、train_batch `:782`、train_step `:832`、eval `:1011`、fit `:1109`）
- 配置：`olmo/config.py`（ModelConfig `:235`、OptimizerConfig `:505`、softmax_auxiliary_loss `:1244`）
- 优化器/调度：`olmo/optim.py`（AdamW `:506`、Scheduler `:651`、CosWithWarmup `:694`、grad clip `:300-351`、build_optimizer `:942`）
- 数据：`olmo/data/__init__.py`（build_train_dataloader `:127`）、`olmo/data/iterable_dataset.py`（可复现 shuffle `:91`）、`olmo/data/memmap_dataset.py`（memmap 读取 `:157`）
- 入口与封装：`scripts/train.py`（FSDP `:210`）、`hf_olmo/modeling_olmo.py`（`OLMoForCausalLM` `:41`）
- 真实配置：`configs/official-1124/OLMo2-7B-stage1.yaml`、`configs/official-0425/OLMo2-1B-stage1.yaml`、`configs/official-0724/OLMo-7B.yaml`、`configs/official-1124/OLMo2-7B-stage2-seed42.yaml`

**论文 / 官方**
- OLMo（2402.00838）：https://arxiv.org/abs/2402.00838
- 2 OLMo 2 Furious（2501.00656）：https://arxiv.org/abs/2501.00656
- Dolma（2402.00159）：https://arxiv.org/abs/2402.00159
- Tülu 3（2411.15124）：https://arxiv.org/abs/2411.15124
- 博客：https://allenai.org/blog/olmo2 · https://allenai.org/blog/olmo2-32B
- HF：https://huggingface.co/allenai/OLMo-2-1124-7B · https://huggingface.co/allenai/OLMo-2-1124-13B · https://huggingface.co/allenai/OLMo-2-0325-32B · https://huggingface.co/allenai/OLMo-2-0425-1B
- GitHub：https://github.com/allenai/OLMo · https://github.com/allenai/OLMo-core · https://github.com/allenai/olmes

> **核实状态**：架构/训练走读均引用本地真实代码行（已逐一核对）；token 规模、benchmark、后训练配方均来自官方 blog / HF card / arXiv（已核实）。少量标"约/备查"项：13B MLP 隐藏维度（HF 推断）、32B GSM8K 口径（HF 表中 61.0，统计设置可能与 7B/13B 不同）。
