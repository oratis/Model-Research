# MiniMax 深度分析（Lightning Attention 线性注意力 / hybrid 7:1 / 大 MoE / 超长上下文 / M1 + CISPO）

> 本文聚焦 **MiniMax-01 系列**（MiniMax-Text-01、MiniMax-VL-01）与其推理升级版 **MiniMax-M1**。MiniMax 的核心创新不在某个 benchmark，而在 **把"线性注意力（linear attention）"第一次成功放大到商用规模**：用 **Lightning Attention**（I/O 感知的线性注意力，O(n) 复杂度）与传统 **Softmax 注意力**按 **7:1** 混合（hybrid），叠加 **456B 总参 / 45.9B 激活**的大 MoE，把训练上下文拉到 **1M tokens**、推理外推到 **4M tokens**。M1 进一步证明：这套混合架构让"超长 CoT / 长上下文 RL"在算力上变得可负担，并提出 **CISPO** 这一新 RL 算法。
>
> 源码 fork 位置：`forks/MiniMax-01`（git submodule，对应 `github.com/MiniMax-AI/MiniMax-01`，本地 commit `57cf223`，最后提交 2025-07-07）。
>
> ⚠️ **重要说明（关于源码范围）**：本 fork 的 `inference/minimax-text-01.py`、`inference/minimax-vl-01.py` 是 **部署/推理启动脚本（launcher）**，并 **不包含模型本体实现**（Lightning Attention、MoE 的 `forward` 写在 HF 模型仓库的 `modeling_minimax_text_01.py` 中，通过 `trust_remote_code=True` 在加载时下载执行——本地 fork 内不存在该文件）。因此本文的"架构剖析/源码走读"采用**双轨佐证**：① 凡本地能引到的（量化白名单暴露的模块名、device_map 切分、EOS id 等）一律给 **`forks/MiniMax-01/...:行号`**；② 算法原理（线性注意力推导、Lightning tiling、MoE 公式、DeepNorm 系数）引 **MiniMax-01 论文 `MiniMax-01.pdf`**（本地，68 页，已用 pypdf 提取核对）+ HF model card。每个数字都标注来源，凡本地无法 100% 确认的标"约/待核实"。
>
> 横向阅读：[[03-deepseek]]（大 MoE / MLA / RL）、[[04-mistral]]（滑窗注意力）、[[05-gemma]]（局部-全局交替）、[[07-glm]]（大 MoE）、[[01-llama]]、[[02-qwen]]。

---

## 1. 概览

**MiniMax**（稀宇科技，MiniMax AI）是一家中国 AI 公司，旗下有 Hailuo（海螺）等产品。MiniMax-01 系列于 **2025-01-15** 开源（见本地 `LICENSE-MODEL:4` "Model Release Date: 15 January 2025"），是当时少有的**把线性注意力做到 456B 规模并公开权重**的工作。

**一句话定位**：别人用滑窗（[[04-mistral]]）、局部-全局交替（[[05-gemma]]）来"省"softmax 注意力的二次开销，MiniMax 走得更激进——**主体直接换成 O(n) 的线性注意力（Lightning Attention），只保留 1/8 的 softmax 全注意力来兜住"精确检索"能力**。这让它在 100K+ 超长序列上的 FLOPs 远低于纯 softmax 模型，是"长上下文 + 大模型"的代表方案。

### 时间线与版本

| 时间 | 版本 | 要点 | 来源 |
|---|---|---|---|
| 2025-01-15 | **MiniMax-Text-01**（456B-A45.9B） | 首发；Lightning+Softmax hybrid (7:1) + MoE(32 专家 top-2)；训练 1M / 推理 4M | `LICENSE-MODEL:4`、[arXiv:2501.08313](https://arxiv.org/abs/2501.08313) |
| 2025-01-15 | **MiniMax-VL-01** | 在 Text-01 上加 303M ViT + 2 层 MLP projector，动态分辨率；512B 视觉-语言 token 续训 | `MiniMax-VL-01-Model-Card.md:44-47` |
| 2025-06-16 | **MiniMax-M1**（40K / 80K 思考预算） | 推理模型（reasoning）；沿用同一 hybrid+MoE 底座；大规模 RL + 新算法 **CISPO**；原生 1M 上下文 | [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) |

**许可证（注意两条不同）**：
- **代码** = **MIT**（`LICENSE-CODE:3` "MIT License / Copyright 2025 MiniMax AI"）。
- **MiniMax-Text-01 / VL-01 权重** = 自定义 **"MiniMax Model License / Model Agreement"**（`LICENSE-MODEL`，**非** 标准 MIT/Apache，含使用条款）。
- **MiniMax-M1 权重** = **Apache-2.0**（HF model card `MiniMaxAI/MiniMax-M1-80k` 标注，已核实）。

> 所以严谨说法是：**01 系列权重是"自定义协议"、M1 权重是 Apache-2.0、所有代码是 MIT**。任务模板里"MIT/Apache 核实"——核实结论见此。

---

## 2. 关键链接

| 资源 | 链接 |
|---|---|
| GitHub（本仓库 fork 对应） | https://github.com/MiniMax-AI/MiniMax-01 |
| 本地 fork | `forks/MiniMax-01`（commit `57cf223`，2025-07-07） |
| HuggingFace 组织 | https://huggingface.co/MiniMaxAI |
| MiniMax-Text-01（HF） | https://huggingface.co/MiniMaxAI/MiniMax-Text-01 |
| MiniMax-VL-01（HF） | https://huggingface.co/MiniMaxAI/MiniMax-VL-01 |
| MiniMax-M1-80k / 40k（HF） | https://huggingface.co/MiniMaxAI/MiniMax-M1-80k · https://huggingface.co/MiniMaxAI/MiniMax-M1-40k |
| 论文 MiniMax-01（本地 `MiniMax-01.pdf`） | https://arxiv.org/abs/2501.08313 |
| 论文 MiniMax-M1 | https://arxiv.org/abs/2506.13585 |
| 官网 / Chat / Platform | https://www.minimax.io · https://chat.minimax.io · https://www.minimax.io/platform |
| 长上下文评测套件（本地） | `forks/MiniMax-01/evaluation/MR-NIAH/`（Multi-Round NIAH，2K→1M） |

---

## 3. 模型规格

下表"架构数字"来自 **本地 model card**（`MiniMax-Text-01-Model-Card.md:54-67`）+ **论文** + **HF card 交叉核对**；凡有冲突项已显式标注。

| 规格 | MiniMax-Text-01 | MiniMax-M1（80k/40k） | 来源 |
|---|---|---|---|
| 总参数（total） | **456B** | 456B（同底座） | model-card:55 / HF M1 card |
| 激活参数（activated/token） | **45.9B** | 45.9B | model-card:56 / HF M1 card |
| 层数 n_layers | **80** | 80 | model-card:57 |
| 隐藏维度 hidden_size | **6144** | 6144 | model-card:66 |
| 注意力混合 | **每 7 层 Lightning + 第 8 层 Softmax**（7:1） | 同 | model-card:58 |
| 注意力头数 | **64**（head_dim **128**） | 64 / 128 | model-card:59-60 |
| Softmax 层的 GQA group | **8**（GQA, group size 8） | 8 | 论文 §2（架构段） |
| MoE 专家数 | **32**，**top-2** 路由 | 32 / top-2 | model-card:62-64 |
| 专家 FFN 隐藏维度 | **9216** | 9216 | model-card:63 |
| 位置编码 | **RoPE**，作用于 head_dim 的一半 | 同 | model-card:65 |
| RoPE base 频率 | **10,000,000**（1e7，**已发布模型**） | 1e7 | model-card:65 / HF card |
| 词表 vocab | **200,064** | 200,064 | model-card:67 |
| EOS token id | **200020** | — | `inference/minimax-text-01.py:87` |
| 训练上下文 | **1M tokens** | RL 阶段长 CoT；原生 1M | 论文摘要 / HF M1 card |
| 推理上下文（外推） | **4M tokens** | 1M（输入）+ 40K/80K（思考预算） | 论文摘要 / HF M1 card |

> ⚠️ **RoPE base 冲突点（值得记住）**：论文正文写 base **10,000**（"a base frequency set to 10,000"），但 **发布的模型 + model card + HF card 均为 10,000,000（1e7）**。以**发布配置 1e7 为准**（更大的 base 利于长上下文外推），论文里的 1e4 很可能是早期/笔误。

> ⚠️ **Text-01 vs M1 的关系**：M1 不是新底座，而是在 01 同一 hybrid+MoE 架构上做"推理化"后训练（大规模 RL）。所以规格列几乎一致，差异在 **后训练 + 思考预算（thinking budget）**。

**VL-01 额外（ViT 部分）**（`MiniMax-Text-01` README:76-83 / VL card:44）：ViT **303M**，24 层，patch 14，hidden 1024，FFN 4096，16 头，head_dim 64；"ViT-MLP-LLM"三段式，动态分辨率 336×336 → 2016×2016 + 一张 336×336 缩略图。

---

## 4. Benchmark

> 口径说明：① **Text-01 核心学术 + 长上下文**来自本地 `MiniMax-Text-01-Model-Card.md`（自报，标 `*` 者为 0-shot CoT）；② **M1 推理/编码**来自 HF `MiniMax-M1-80k` model card（已联网核实）。对照模型版本以原表为准。

### 4.1 MiniMax-Text-01：核心学术（来源 model-card:73-90）

| 任务 | MiniMax-Text-01 | GPT-4o(11-20) | Claude-3.5-Sonnet(10-22) | DeepSeek-V3 | Qwen2.5-72B |
|---|---|---|---|---|---|
| MMLU* | **88.5** | 85.7 | 88.3 | 88.5 | 86.1 |
| MMLU-Pro* | 75.7 | 74.4 | 78.0 | 75.9 | 71.1 |
| C-SimpleQA | **67.4** | 64.6 | 56.8 | 64.8 | 52.2 |
| IFEval (avg) | 89.1 | 84.1 | 90.1 | 87.3 | 87.2 |
| Arena-Hard | 89.1 | 92.4 | 87.6 | 91.4 | 81.2 |
| GPQA* (diamond) | 54.4 | 46.0 | 65.0 | 59.1 | 49.0 |
| MATH* | 77.4 | 76.6 | 74.1 | 84.6 | 81.8 |
| HumanEval | 86.9 | 90.2 | 93.7 | 92.1 | 86.6 |

> 读法：Text-01（**非推理模型**）在 **通用知识（MMLU 88.5、中文 C-SimpleQA 67.4 第一）** 上与第一梯队齐平；**数学/代码**略弱于 DeepSeek-V3（符合"通用 base"定位）。它的真正杀手锏在长上下文（下表）。

### 4.2 MiniMax-Text-01：长上下文（来源 model-card:100-124）

**RULER（不同长度平均准确率）**——这是 MiniMax 最亮的一张表：

| 模型 | 4k | 64k | 128k | 256k | 512k | 1M |
|---|---|---|---|---|---|---|
| **MiniMax-Text-01** | 0.963 | 0.943 | **0.947** | **0.945** | **0.928** | **0.910** |
| Gemini-1.5-Pro(002) | 0.962 | 0.938 | 0.917 | 0.916 | 0.861 | 0.850 |
| Gemini-2.0-Flash(exp) | 0.960 | 0.937 | 0.860 | 0.797 | 0.709 | - |
| GPT-4o(11-20) | 0.970 | 0.884 | - | - | - | - |

> **在 ≥128k 长度上 MiniMax-Text-01 全面领先**，且一直撑到 1M（0.910）。这正是 hybrid 线性注意力的价值：长序列既不爆显存也不掉检索。
> **LongBench v2**（w/ CoT）overall **56.5**（超过 GPT-4o 51.4、Claude-3.5 46.7、人类 53.7）；**4M NIAH** 大海捞针测试在论文 Figure 14 全绿（128K→4M，token 间隔 ≥1M 时为 0.5M）。

### 4.3 MiniMax-M1：推理 / 长 CoT / 编码（来源 HF `MiniMax-M1-80k` card，已核实）

| 任务 | M1-80k | M1-40k | 说明 |
|---|---|---|---|
| AIME 2024 | **86.0** | 83.3 | 数学竞赛 |
| AIME 2025 | 76.9 | 74.6 | |
| LiveCodeBench | 65.0 | 62.3 | 代码 |
| **SWE-bench Verified** | **56.0** | 55.6 | 真实软件工程（agentic） |
| MMLU-Pro | 81.1 | 80.6 | |
| GPQA Diamond | 70.0 | 69.2 | |
| LongBench-v2 | 61.5 | 61.0 | 长上下文 |
| TAU-bench (airline/retail) | 62.0 / 63.5 | 60.0 / 67.8 | 工具使用 |
| OpenAI-MRCR (128k / 1M) | 73.4 / 56.2 | 76.1 / 58.6 | 长上下文检索 |

> M1 的定位：**开源权重里，长上下文 + agentic 工具/软件工程的第一梯队**，对标 DeepSeek-R1 / Qwen3-235B（HF card / [InfoQ 报道](https://www.infoq.com/news/2025/06/minimax-m1/)）。论文称在 100K 长度上 M1 只需 **DeepSeek-R1 约 25% 的 FLOPs**——这正是 Lightning Attention 省出来的。

### 4.4 MiniMax-VL-01：多模态（来源 VL card:57-75）

| 任务 | MiniMax-VL-01 | GPT-4o(11-20) | Claude-3.5-Sonnet | Qwen2-VL-72B |
|---|---|---|---|---|
| MMMU* | 68.5 | 63.5 | 72.0 | 64.5 |
| ChartQA*(relaxed) | **91.7** | 88.1 | 90.8 | 91.2 |
| DocVQA* | 96.4 | 91.1 | 94.2 | 97.1 |
| OCRBench | **865** | 806 | 790 | 856 |
| MathVista* | 68.6 | 62.1 | 65.4 | 69.6 |
| M-LongDoc(acc) | 32.5 | 41.4 | 31.4 | 11.6 |

> VL-01 在 **OCR / 图表（ChartQA 91.7、OCRBench 865 双第一）** 上突出，长文档 M-LongDoc 也得益于底座的长上下文能力。它把 Text-01 的"超长上下文"优势带到了多模态长文档场景。

---

## 5. 架构剖析

### 5.0 一层 block 长什么样（论文 Figure 3）

论文 Figure 3 给出的单层结构（提取自 `MiniMax-01.pdf` 架构图文字）是标准的 **PostNorm 残差 + DeepNorm**，每个 block = "注意力 + RMSNorm" 后接 "MoE + RMSNorm"，残差用一个**逐层系数 α** 缩放：

```text
# 论文 Figure 3（文字版）：两种 block 交替
LightningAttention + RMSNorm  →  MoE + RMSNorm     （×7）
SoftmaxAttention   + RMSNorm  →  MoE + RMSNorm     （×1，每 8 层一次）
# 残差缩放系数 α（图中标 α），即 DeepNorm 的 PostNorm 系数
```

这个 **α 系数**在本地 launcher 里露了脸——量化时被列入"不量化白名单"，逐层存在：

```python
# inference/minimax-text-01.py:19-20  （int8 量化白名单，按层枚举）
] + [f"model.layers.{i}.coefficient"        for i in range(hf_config.num_hidden_layers)]  # ← 每层一个 α 系数(coefficient)，FP 保留不量化
+ [f"model.layers.{i}.block_sparse_moe.gate" for i in range(hf_config.num_hidden_layers)] # ← 每层一个 MoE 路由门 gate，FP 保留
```

> 为什么这两个模块"不量化"很关键：`coefficient`（DeepNorm 残差系数）和 `block_sparse_moe.gate`（MoE 路由打分）是**数值最敏感**的小张量——量化它们会直接破坏训练稳定性/路由决策。这条白名单**反向证明了**模型结构：**逐层都有 `block_sparse_moe`（即每层 FFN 都是 MoE）+ 逐层 `coefficient`（DeepNorm）**。

### 5.1 Lightning Attention：线性注意力为何能 O(n)

**(1) 线性注意力的数学核心 —— "右乘核技巧"（right product kernel trick）**（论文 §2.2，公式 2-3）：

```text
# 标准（softmax 化简的）NormAttention：先算 n×n 注意力矩阵 → O(n²)
O = Norm( (Q Kᵀ) V )            (式2)    Q,K,V ∈ ℝ^{n×d}

# 线性变体：改变乘法结合顺序，先算 d×d 的 KᵀV → O(n·d²)，对 n 线性
O = Norm( Q (Kᵀ V) )            (式3)
```

直觉：`(QKᵀ)V` 要先生成 `n×n` 的大矩阵（序列每两 token 都要算一次，长度翻倍开销×4）；而 `Q(KᵀV)` 先把 K、V 压成一个 **`d×d` 的"状态矩阵" KᵀV**，再让 Q 去乘它——**这个状态矩阵的大小只和特征维 d 有关，与序列长 n 无关**。于是：
- **训练复杂度** O(n·d²)，对 n **线性**（论文："training complexity of O(nd²)"）；
- **解码（推理）每步** O(d²) **常数**，靠"**递推更新 KᵀV**"实现（论文："recurrently updating the term KᵀV ... constant computational complexity of O(d²), irrespective of the sequence length"）。

> 这一点对照 [[03-deepseek]] 的 MLA 很有意思：**MLA 是把 KV cache *压小*（低秩潜向量），但缓存仍随 n 线性增长**；**线性注意力是把 KV 换成一个 *固定大小* 的状态矩阵 KᵀV，根本不随 n 增长**——这是两条不同的"省长上下文"路线。

把"递推更新 KᵀV"写成解码伪代码，最能体会"为什么没有 KV cache 也能自回归"：

```text
# 线性注意力的自回归解码（去掉 norm/gating，便于理解）
S = 0                         # 状态矩阵 S = ΣKᵀV，形状 d×d，大小恒定（与已生成长度无关）
for t in 1..n:               # 逐 token 解码
    S = S + k_t ᵀ · v_t      # O(d²)：把当前 token 的 K/V 累加进状态
    o_t = q_t · S            # O(d²)：当前查询直接乘状态矩阵，得到输出
# 全程显存只存一个 d×d 的 S —— 对比 softmax 要存 n 个 (k,v) 的 KV cache（随 n 线性膨胀）
```

> **这就是 4M 外推的物理基础**：softmax 解码到第 400 万 token 时，KV cache 已是天文数字；线性注意力的"显存账"始终是一个固定大小的 `S`。代价是 `S` 把历史**有损压缩**了（无法逐 token 精确回看）——所以才需要下面 5.2 的 softmax 层来兜检索。

**(2) 为什么需要 "Lightning"（I/O 感知实现）**（论文 §2.2.1）：

纯线性注意力在 **causal（因果）语言建模**下有个坑——为了保证"只看过去"，朴素实现要做 **cumsum（前缀和）**，这是个**串行、慢、I/O 受限**的操作，吃不满 GPU。Lightning Attention 的做法是 **分块 tiling**：

```text
# 论文 §2.2.1：把注意力拆成 块内(intra-block) + 块间(inter-block)
intra-block：用「左乘」(QKᵀ)V —— 块很小，O(块²) 可忽略，保留因果 mask
inter-block：用「右乘」 Q(KᵀV) —— 跨块用递推的状态矩阵，避免全局 cumsum
```

通过"块内左乘 + 块间右乘"，**绕开了全局 cumsum**，配合定制 CUDA kernel 把整体复杂度压回线性，论文报告 Lightning Attention 推理可达 **>75% MFU**（Model Flops Utilization）。"I/O 感知"一词与 FlashAttention 同源——核心都是减少 HBM↔SRAM 的搬运。

### 5.2 Hybrid 7:1：为什么不全用线性注意力

线性注意力虽快，但**牺牲了精确检索（retrieval）能力**——状态矩阵 KᵀV 把历史"压缩"了，对"在第 80 万 token 处精确找回某个数字"这类任务不如全注意力。MiniMax 的解法是**混合**：

> "a softmax attention is positioned after every 7 lightning attention"（`model-card:58`）——**每 7 层 Lightning，插 1 层 Softmax 全注意力，80 层共 10 个 softmax 层**。

- **7 层 Lightning** 负责 O(n) 高效处理长序列主体；
- **第 8 层 Softmax**（带 RoPE + GQA group=8）周期性地"做一次精确全局检索"，把长程依赖兜回来。

论文消融（§架构, Table 5）也佐证：在 60B/9.3B-A、48 层、500B token 的对照里，**Hybrid-lightning 在 BBH/DROP/MATH 等多数指标上优于纯 Softmax**（如 BBH 32.2 vs 28.2，MATH 6.8 vs 4.6）——既快又不掉点。

> 横向对照（"如何省 softmax 的二次开销"四条路线）：
> - **[[04-mistral]]**：滑动窗口注意力（sliding window）——每个 token 只看最近 W 个，仍是 softmax，靠层数堆叠间接覆盖全局。
> - **[[05-gemma]]**：局部-全局交替（如 5:1 local:global）——大部分层是滑窗 softmax，少数层全局 softmax。
> - **MiniMax**：**主体换成线性注意力**（不是 softmax），只留 1/8 softmax——**更彻底**，复杂度从 O(n²) 真正降到 O(n)。
> - **[[03-deepseek]]**：保留全 softmax，但用 MLA 压 KV cache——省的是**显存**不是**计算量级**。

### 5.3 MoE：32 专家 top-2 + 全局路由 + 辅助负载均衡

**(1) MoE 公式**（论文 §2.1, 式1）：

```text
# 每个 token x_t 的输出 = top-k 专家 FFN 的加权和
h_t = Σ_{i=1..E}  Softmax_i( TopK( x_t · W_g ) ) · FFN_i(x_t)
# E=32 专家；TopK 把 top-2 之外的打分置 -∞（只激活 2 个专家）
```

每层 FFN 都被替换为 32 个专家、每 token 选 top-2（专家 FFN 隐藏维度 9216）。这就是为什么 456B 总参里**每 token 只激活 45.9B**——绝大部分专家不参与单个 token 的计算。

**(2) 负载均衡：辅助损失 + 全局路由**（论文 §2.1）：

```text
# 辅助损失（防止专家"旱涝不均"）
L_aux = α_aux · (1/E) · Σ_i f_i · m_i
        f_i = 分到第 i 专家的 token 比例；m_i = 该专家平均路由概率
# Global Router：跨 Expert-Parallel(EP) 组做全局 token 分配，降低 token drop
```

> 对照 [[03-deepseek]]：DeepSeekMoE 走的是 **auxiliary-loss-free**（用可学习 bias 调负载，不加 aux loss）+ 共享专家 + 细粒度专家；MiniMax 这里是更**经典的 aux-loss MoE**（32 个较粗粒度专家、top-2、辅助损失）。两者都解决"负载均衡"，但思路不同。

### 5.4 DeepNorm（PostNorm 稳定化）

MiniMax 用 **PostNorm + DeepNorm**，而不是当下主流的 PreNorm（论文 §架构 + §4.2）：

```text
# 论文 §4.2 初始化：DeepNorm 缩放系数（N=层数=80）
α = (2N)^0.25 ,  β = (8N)^(-0.25)        # 残差用 α 放大、初始化用 β 缩小
# 论文消融 Table 5：PostNorm(用 DeepNorm) 全面优于 PreNorm
```

> 这正对应 5.0 里 launcher 暴露的逐层 `coefficient`（α）。**为什么用 PostNorm？** 论文消融显示在它们的设置下 PostNorm 收敛与下游指标更好；DeepNorm 给的 α/β 系数让深层 PostNorm 也能稳定训练（PostNorm 历史上以"深了容易炸"著称，DeepNorm 正是为此而生）。这和 [[03-deepseek]]/[[01-llama]] 的"PreNorm + RMSNorm"是不同的稳定化哲学。

### 5.5 长上下文：训练 1M、推理外推 4M

- **训练侧**用三类并行/kernel 支撑 1M 序列（论文 §3.2）：
  - **LASP+**（Linear Attention Sequence Parallelism Plus）——线性注意力专用的序列并行，充分利用多卡；
  - **varlen ring attention**——给 softmax 层用的变长 ring attention，支持"上下文无限扩展"；
  - **ETP（Expert Tensor Parallel）/ EP** + 计算-通信 overlap——给 MoE 的 all-to-all 通信降开销；
  - **data-packing**：把不同样本沿序列维首尾拼接，避免 1M 长度下 padding 的巨量浪费。
- **推理侧外推到 4M**：因为线性注意力是"固定大小状态 + 递推"，天然适合超长外推；论文 Figure 14 的 4M NIAH 压力测试全绿。

---

## 6. 源码走读（本地 `forks/MiniMax-01`）

> 再次强调：本 fork 不含 `modeling_*.py` 模型本体（走 `trust_remote_code` 远程加载）。本章走读**本地实际存在**的部署脚本，看 MiniMax 在"**怎么把一个 456B、80 层、跨 8 卡的 hybrid-MoE 模型跑起来**"上的工程细节——这些细节反过来印证了架构。

### 6.1 关键文件清单

| 文件 | 作用 | 行数 |
|---|---|---|
| `inference/minimax-text-01.py` | Text-01 推理 launcher：量化白名单 + 8 卡 device_map + 生成 | 100 |
| `inference/minimax-vl-01.py` | VL-01 推理 launcher：多了 vision_tower / projector 的放置 | 123 |
| `MiniMax-Text-01-Model-Card.md` | 规格 + 全部 benchmark 表 | 232 |
| `MiniMax-01.pdf` | 技术报告（68 页，架构/训练/评测） | — |
| `evaluation/MR-NIAH/` | Multi-Round 大海捞针长上下文评测（2K→1M） | — |
| `docs/{vllm,transformers}_deployment_guide*.md` | vLLM / Transformers 部署指南 | — |

### 6.2 走读 `minimax-text-01.py`：量化白名单 = 架构指纹

最有信息量的是 int8 量化配置——它**按层枚举**了两类"不可量化"模块，等于给出了一份逐层结构清单：

```python
# inference/minimax-text-01.py:14-21
"int8": QuantoConfig(
    weights="int8",
    modules_to_not_convert=[
        "lm_head",
        "embed_tokens",
    ] + [f"model.layers.{i}.coefficient" for i in range(hf_config.num_hidden_layers)]          # ← DeepNorm 残差系数 α（逐层）
    + [f"model.layers.{i}.block_sparse_moe.gate" for i in range(hf_config.num_hidden_layers)]   # ← MoE 路由门 gate（逐层）
),
```

读出三件事：① 每层都有 `block_sparse_moe`（**全层 MoE**，无 dense-only 前几层，区别于 [[03-deepseek]] 的前 3 层 dense）；② 每层都有 `coefficient`（**逐层 DeepNorm α**）；③ `coefficient`/`gate` 这种小而敏感的张量保持 FP 精度。

### 6.3 走读 `minimax-text-01.py`：80 层如何切到 8 卡（流水并行）

```python
# inference/minimax-text-01.py:38     约束：层数必须能被卡数整除
assert hf_config.num_hidden_layers % args.world_size == 0   # 80 % 8 == 0 ✓
# inference/minimax-text-01.py:55-63  逐层放卡：embed→cuda:0，norm/lm_head→最后一卡，中间层均分
device_map = {
    'model.embed_tokens': 'cuda:0',
    'model.norm':   f'cuda:{args.world_size - 1}',
    'lm_head':      f'cuda:{args.world_size - 1}',
}
layers_per_device = hf_config.num_hidden_layers // args.world_size            # 80//8 = 10 层/卡
for i in range(args.world_size):
    for j in range(layers_per_device):
        device_map[f'model.layers.{i * layers_per_device + j}'] = f'cuda:{i}'  # 第 i 卡放第 [10i, 10i+10) 层
```

这是教科书式的**层间流水（pipeline-style）切分**：8 张卡各放 10 层，embedding 在 0 卡、最终 norm+输出头在 7 卡。`check_params`（`:34-38`）还强制 int8 时 `world_size >= 8`（`:36`）——456B 模型即便 int8 也得 8×80G 起步。

```python
# inference/minimax-text-01.py:85-90   生成配置：注意 EOS=200020（硬编码）
generation_config = GenerationConfig(max_new_tokens=20, eos_token_id=200020, use_cache=True)
```

### 6.4 走读 `minimax-vl-01.py`：多模态的差异

VL 版基本复用 Text-01 的逻辑，差别集中在**视觉塔的放置与不量化**：

```python
# inference/minimax-vl-01.py:21-29   量化白名单多了视觉三件套
modules_to_not_convert=[
    "vision_tower", "image_newline", "multi_modal_projector",   # ← ViT + 图像换行符 + MLP 投影器，全部 FP 保留
    "lm_head", "embed_tokens",
] + [f"model.layers.{i}.coefficient" ... text_config.num_hidden_layers ...]      # ← 注意：层数取自 text_config
  + [f"model.layers.{i}.block_sparse_moe.gate" ...]
# inference/minimax-vl-01.py:74-84   device_map 前缀变成 language_model.*，且 vision_tower 整个塞 cuda:0
```

> 读出："ViT-MLP-LLM"三段式在权重命名上就是 `vision_tower.* / multi_modal_projector.* / language_model.model.*`；语言塔的层数从 `hf_config.text_config.num_hidden_layers`（`:45`）取——印证 VL 的 LLM 主体就是 Text-01（80 层）。

---

## 7. 训练机制

### 7.1 MiniMax-Text-01 预训练（来源：论文 §4，本地 `MiniMax-01.pdf`）

- **初始化**：Xavier 初始化；DeepNorm 系数 α=(2N)^0.25、β=(8N)^(-0.25)，N=80（论文 §4.2）。
- **优化器**：AdamW，β1=0.9、β2=0.95、weight decay=0.1（§4.2）。
- **学习率**：500 步线性 warmup → 峰值 **2×10⁻⁴** → 常数段（§4.2）。
- **序列长度/批量**：训练序列长 **8192**；**batch size 渐进放大**——16M→32M(@69B tok)→64M(@790B tok)→128M(@4.7T tok)，依据"训练 loss 与临界 batch size（critical batch size）的幂律关系"来定何时翻倍（§4.2，Figure 13）。
- **scaling law 约束规模**：论文把总参约束在 **<500B**，以便 8×80G 单机 int8 跑 1M 序列（§求解的优化问题，式13）——最终落在 456B。
- **MoE vs Dense**：等 FLOPs 对照（Figure 4，各 1T token）显示 MoE 在 HellaSwag/TriviaQA 等上以更少计算达到同等性能。

**长上下文扩展训练**（论文 §4.2 长上下文段）：分**多个阶段**逐步加长，每阶段调整 **RoPE base 频率 + 训练长度**（详见论文 Table 6）；在每阶段**最后 20%** 混入 **10% 高质量长上下文 QA 数据**，并对数据源权重做**线性插值过渡**以稳住分布漂移。这套"分阶段拉长 + 退火混长文"是把上下文做到 1M 的关键。

### 7.2 MiniMax-VL-01 训练（来源：VL card:46 / README:50）

- ViT（303M）从零在 **6.94 亿 (694M) 图文对**上预训练；
- 全流程四阶段共处理 **512B 视觉-语言 token**；
- projector（2 层 MLP）随机初始化，与 Text-01 base 联合训练；动态分辨率 patch + 缩略图拼接成完整图像表示。

### 7.3 MiniMax-M1：大规模 RL + CISPO（来源：[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)，已核实）

M1 = 在 01 hybrid+MoE 底座上做"推理化"的**大规模强化学习**，核心三点：

**(1) 为什么 hybrid 架构对长 CoT RL 特别划算**：推理模型要生成**极长的思维链（CoT）**、并在长上下文环境（如软件工程沙箱）里做 agentic rollout。线性注意力让"生成 + 评分超长序列"的算力大幅下降——论文称 100K 长度下 M1 仅需 DeepSeek-R1 **约 25% 的 FLOPs**。这直接让 RL（要反复 rollout 长序列）变得可负担。

**(2) CISPO（Clipped IS-weight Policy Optimization）**：M1 提出的新 RL 算法。与 PPO/GRPO **裁剪 token 级更新**不同，**CISPO 裁剪的是"重要性采样权重（importance sampling weights）"而非 token 更新**，论文报告其收敛与效果优于其它竞争性 RL 变体（[arXiv:2506.13585](https://arxiv.org/abs/2506.13585) 摘要 / [HF papers](https://huggingface.co/papers/2506.13585)）。

> 对照 [[03-deepseek]]：R1 用 **GRPO**（去掉 critic、用组内相对优势）；MiniMax-M1 用 **CISPO**（裁剪 IS 权重）。两者都在"让长 CoT RL 更稳更省"，但裁剪对象不同——这是 2025 年推理-RL 算法演进的一个有趣分叉。

**(3) 思考预算（thinking budget）与训练成本**：发布 **40K / 80K** 两个思考预算版本（thinking budget = 单次会话可生成的最大输出 token 数，界定推理/分解的"深度"）；其中 40K 是 80K 训练的中间阶段产物。**整套 RL 在 512 张 H800 上仅用 3 周、租用成本约 \$534,700** 完成（[arXiv:2506.13585](https://arxiv.org/abs/2506.13585)）——"hybrid 省 FLOPs → RL 便宜"的最直接证据。80K 版在长程/高复杂任务上优于 40K 版。

---

## 8. 学习要点与横向对比

| 维度 | MiniMax 的做法 | 横向对照 |
|---|---|---|
| **省长序列开销** | **主体换成线性注意力（Lightning）**，O(n²)→O(n)，KV 变成固定大小状态矩阵 KᵀV | [[04-mistral]] 滑窗(局部 softmax) / [[05-gemma]] 局部-全局交替 / [[03-deepseek]] MLA 压 KV cache——只有 MiniMax 把**计算量级**真正降下来 |
| **保留检索能力** | hybrid **7:1**：每 8 层留 1 层 softmax 全注意力兜底 | 纯线性会掉检索；混合是"既要又要"的工程折中 |
| **大 MoE** | 32 专家 top-2、全层 MoE、aux-loss 负载均衡、global router | [[03-deepseek]] auxiliary-loss-free + 共享专家 + 细粒度；[[07-glm]] 大 MoE——MiniMax 走经典 aux-loss 路线 |
| **归一化/稳定性** | **PostNorm + DeepNorm**（逐层 α/β 系数） | [[03-deepseek]]/[[01-llama]] 用 PreNorm+RMSNorm——MiniMax 是少见的"PostNorm 也能稳"的大模型 |
| **超长上下文** | 训练 1M / 推理 4M；LASP+ / varlen ring / data-packing | 多数模型 128K-256K；MiniMax 是"原生百万级"代表 |
| **推理-RL** | M1：CISPO（裁 IS 权重）+ 思考预算 40K/80K；hybrid 让长 CoT RL 便宜 | [[03-deepseek]] R1 用 GRPO——裁剪对象不同的两条 RL 路线 |

> **一句话总结**：要理解"**线性注意力如何在商用规模落地**"，MiniMax 是当前最好的范本——`O = Norm(Q(KᵀV))` 这一行结合律改写是全部魔法的起点；要理解"**为什么不能全用线性注意力**"，记住 hybrid 7:1 的存在理由（检索）；要理解"**长上下文 + 推理 RL 为何能便宜**"，看 M1 用 \$53 万跑完 RL 的账。配合 [[03-deepseek]]（另一条大 MoE + RL 路线）对照阅读，能把"2025 年大模型如何同时追求规模、长度、推理"看得很完整。

---

## 9. 参考

**本地源码 / 资料（`forks/MiniMax-01`，commit `57cf223`）**
- 推理 launcher：`inference/minimax-text-01.py`（量化白名单+架构指纹 `:14-21`、层数整除约束 `:38`、device_map 流水切分 `:55-63`、EOS/生成 `:85-90`）、`inference/minimax-vl-01.py`（视觉塔不量化 `:21-29`、text_config 层数 `:45`、device_map `:74-84`）
- 规格 / benchmark：`MiniMax-Text-01-Model-Card.md`（架构 `:54-67`、核心学术 `:73-90`、RULER/LongBench `:100-124`）、`MiniMax-VL-01-Model-Card.md`（ViT/VL `:44-47`）、`README.md:59-83`
- 技术报告：`MiniMax-01.pdf`（68 页，§2 架构 / §2.2 线性注意力公式 2-3 / §2.2.1 Lightning tiling / §2.1 MoE 式1 + aux loss / §4.2 训练 / §3.2 LASP+/varlen ring/ETP / Figure 3 架构图 / Figure 13 批量幂律 / Figure 14 4M NIAH）
- 许可证：`LICENSE-CODE`（MIT）、`LICENSE-MODEL`（MiniMax Model Agreement，发布日 `:4`）
- 长上下文评测：`evaluation/MR-NIAH/README.md`

**论文 / 官方（联网核实）**
- MiniMax-01（2501.08313）：https://arxiv.org/abs/2501.08313
- MiniMax-M1（2506.13585，CISPO / 40K-80K thinking budget / \$534,700 RL）：https://arxiv.org/abs/2506.13585 · https://huggingface.co/papers/2506.13585
- HF 组织 / 模型：https://huggingface.co/MiniMaxAI · `MiniMaxAI/MiniMax-Text-01` · `MiniMaxAI/MiniMax-VL-01` · `MiniMaxAI/MiniMax-M1-80k`（Apache-2.0，benchmark 已核实）
- GitHub：https://github.com/MiniMax-AI/MiniMax-01
- 报道（M1 规格佐证）：https://www.infoq.com/news/2025/06/minimax-m1/

> **核实状态**：架构数字、benchmark、许可证均经本地 model card + 论文 + HF card 三方交叉核对。**显式存疑项**：① **RoPE base** 论文正文写 1e4，发布模型/card 为 **1e7**（本文以发布配置 1e7 为准）；② 模型**本体 `forward` 代码不在本 fork**（远程 `trust_remote_code` 加载），故"线性注意力/MoE 的逐行实现"以论文公式 + launcher 暴露的模块名佐证，未能给出 `modeling_*.py:行`；③ M1 部分 benchmark（如 TAU-bench retail）出现 40k>80k 的非单调，按 HF card 原值如实记录。
