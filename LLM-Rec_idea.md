# 生成式推荐中的样本工程：LLM for Rec 的创新方向

## 一、问题背景

### 1.1 当前 LLM4Rec 的通用范式

现有的 LLM 生成式推荐项目（P5、GenRec、RecAI、HLLM、OnePiece 等）在训练策略上几乎完全照搬 NLP 领域的通用做法：

- **序列构造**：按时间窗口截断用户行为序列，不做任何难度筛选
- **负采样**：随机采样或 batch 内随机负采样（in-batch negative）
- **数据增广**：照搬 NLP 的 token 级别增广（随机删除、替换），忽略推荐场景的结构特性
- **训练策略**：统一采样、统一训练，所有样本一视同仁

### 1.2 为什么推荐场景需要专门的样本工程

推荐数据与 NLP 数据有本质差异，照搬 NLP 范式会导致系统性缺陷：

| 维度 | NLP 数据 | 推荐数据 |
|------|---------|---------|
| 分布 | 近似均匀（语料经过平衡） | 严重长尾（热门物品占 80% 交互） |
| 序列长度 | 相对集中（128-512 tokens） | 方差极大（新用户 3 条，老用户 3000 条） |
| 负样本定义 | 明确（语言模型中的错误 token） | 模糊（未交互 ≠ 不喜欢） |
| 语义结构 | 天然的语法依存关系 | 隐式的意图-品类-物品层级关系 |
| 监督信号 | 每个 token 都有明确标签 | 仅有隐式反馈（点击/购买），噪声大 |

Shopee 的 OnePiece 团队在工业部署后明确指出：**"样本工程是生成式推荐领域尚未开垦的处女地"**。当前没有任何开源项目在这一方向上有系统性的探索。

---

## 二、核心创新：三项推荐专用的样本工程方法

### 2.1 推荐适配的课程学习（RecCL: Recommendation Curriculum Learning）

#### 动机

现有方法对所有用户序列一视同仁地训练。但一个包含 3 次交互的新用户和一个包含 500 次交互的老用户，其序列的建模难度完全不同。同时，热门物品的交互模式简单且重复，冷门物品则需要模型进行更精细的语义推理。

#### 方法设计

**三维渐进课程体系**：

```
Stage 1: 短序列 + 热门物品 → 学习基本的协同过滤模式
Stage 2: 中等序列 + 混合物品 → 学习序列转移规律
Stage 3: 长序列 + 冷门物品 → 学习细粒度语义推理
```

**具体的难度度量**：

1. **序列难度** $\mathcal{D}_{seq}(u)$：用户交互序列长度的倒数 + 序列熵（品类多样性）
2. **物品难度** $\mathcal{D}_{item}(i)$：物品流行度的倒数（冷门物品更难）
3. **预测难度** $\mathcal{D}_{pred}(u, i)$：协同过滤模型（如 SASRec）对该样本的预测置信度——置信度越低，说明纯粹的协同信号不足以做出判断，这类样本更需要 LLM 的语义能力

**调度策略**：

不是简单的三阶段切换，而是一个连续的采样权重调整过程。训练过程中，每个样本被采样的概率与三者的加权和成正比，权重随训练步数动态变化：早期偏向 $\mathcal{D}_{seq}$ 低的样本（简单序列），后期逐步增加 $\mathcal{D}_{pred}$ 高的样本（需要语义推理的困难样本）。

#### 预期效果

- 训练早期在简单样本上快速收敛，避免冷门物品在模型尚未学好基础模式时引入噪声
- 训练后期在困难样本上精细调优，让 LLM 的语义推理能力在真正需要它的地方发挥作用
- 缓解流行度偏差：困难样本中冷门物品比例高，后期训练自然地增加了对冷门物品的曝光

---

### 2.2 语义感知的负样本分层（SANS: Semantic-Aware Negative Sampling）

#### 动机

现有生成式推荐中，负样本构造方式极为粗糙：
- 随机采样：大部分负样本是"与正样本完全无关的物品"，模型几乎不需要学习就能区分
- In-batch negative：受 batch 内热门物品主导，冷门物品几乎没有机会成为负样本

这导致模型学到的决策边界非常粗糙——只需要区分"相关 vs 无关"，而无法区分"相关 vs 高度相关但不完全匹配"。

#### 方法设计

**三级负样本分层**：

| 层级 | 定义 | 构造方式 | 训练权重 |
|------|------|---------|---------|
| **Easy Negatives** | 与正样本完全无关 | 随机采样（品类不同） | 0.1 |
| **Medium Negatives** | 同类目但不同物品 | 同品类随机采样 | 0.3 |
| **Hard Negatives** | 语义相似但用户不感兴趣 | LLM 生成"用户可能混淆的替代品" | 0.6 |

**Hard Negatives 的 LLM 生成方法**：

给定正样本物品 $i^+$ 及其文本描述（标题、品类、属性），让 LLM 生成语义高度相似但关键的差异点：

```
Prompt 示例：
"用户购买了一个物品，描述如下：{物品标题}
请生成 3 个看起来非常相似、用户可能会混淆、但不完全符合该用户需求的替代品。
要求：(1) 品类相同 (2) 核心功能相似 (3) 但关键属性（如品牌、价格段、适用场景）有差异"
```

LLM 生成的不是真实存在的物品 ID，而是"物品描述文本"。后续通过以下方式将这些描述落地为真实物品：
- 对物品库做文本 embedding（如 sentence-transformer），检索与 LLM 生成描述最相似的 Top-K 真实物品
- 取相似度排名在第 3-10 名的物品（第 1-2 名可能太接近正样本）

**对比学习的损失函数适配**：

在标准的 InfoNCE 损失中，按层级对负样本加权：

$$\mathcal{L} = -\log \frac{\exp(s(q, i^+) / \tau)}{\exp(s(q, i^+) / \tau) + \sum_{k} w_k \cdot \exp(s(q, i^-_k) / \tau)}$$

其中 $w_k$ 为 SANS 层级权重（Hard > Medium > Easy），$\tau$ 为温度参数。

#### 预期效果

- 模型学习到更精细的决策边界，能区分"看起来像但实际不匹配"的物品
- 隐性提升推荐的多样性：模型不会简单地输出最热门的物品，因为它被迫学习了差异化
- Hard negatives 提供了天然的"可解释性锚点"——模型需要理解为什么用户要 A 不要 B

---

### 2.3 推荐语义保持的数据增广（RecAug: Recommendation-Specific Semantic Augmentation）

#### 动机

NLP 的数据增广（EDA：随机删除、同义词替换、随机交换）在推荐序列上直接使用会严重破坏序列的推荐语义：

- **随机删除**：删掉用户的"触发物品"（导致后续购买的导火索），整个序列的因果结构被破坏
- **随机交换**：颠倒因果顺序（先买手机壳再买手机），训练信号变噪声
- **同义词替换**：推荐中没有"同义词"的概念——把 iPhone 14 替换成小米 13，用户的购买意图完全改变

#### 方法设计

**三项推荐专用的增广操作**：

**增广 1：意图保持的序列截断（Intent-Preserving Truncation）**

不是随机删除，而是基于 LLM 判断"该物品是否属于当前会话的核心意图"来选择删除对象。

具体方法：
- 对用户序列中的每个物品，用 LLM 生成一句话描述"用户买这个物品时满足的意图"
- 对语义高度重复的意图（如"补充日用品：买了纸巾"和"补充日用品：买了洗衣液"），随机保留其中一个
- 保留所有"意图转折点"物品（如从"日用品"突然变成"电子产品"），这些是序列的骨架

**增广 2：会话边界感知的分段重排（Session-Boundary-Aware Permutation）**

人的购物行为天然有会话边界（session boundary）。两个会话之间间隔 3 小时和间隔 3 天，语义独立性完全不同。

具体方法：
- 用时间间隔 + LLM 意图分析联合检测会话边界
- 在会话内部：保持时间顺序不变（因果结构不能破坏）
- 在不同会话之间：随机交换会话块的顺序——"先逛街再吃饭"和"先吃饭再逛街"都是合理的

**增广 3：LLM 驱动的物品替换（LLM-Guided Item Substitution）**

不使用简单的规则替换，而是让 LLM 在"保持意图不变"的前提下生成替代物品：

```
Prompt 示例：
"用户在购物会话中的意图是：{LLM总结的意图}
当前物品：{物品标题}
请推荐 2 个可以替代该物品、满足相同意图的其他物品。
要求：保持品类相同、价位接近、满足同一使用场景。"
```

同样，LLM 生成的替代品通过文本 embedding 检索落地到真实物品库。

**增广的组合策略**：

不是随机组合三种增广，而是根据序列的"冗余度"自适应选择：

- 高冗余序列（如连续 5 个零食购买）→ 增广 1（意图保持截断）+ 增广 3（同品类替换）
- 低冗余序列（每个物品代表不同意图）→ 仅增广 2（会话重排），保留所有意图信息

#### 预期效果

- 增广后的序列保持了原始序列的推荐语义结构，不会引入训练噪声
- 语义相同的会话块随机排列，增强了模型对"会话间相对顺序不敏感"的归纳偏置
- LLM 引导的物品替换天然保证了替换后物品与原始物品的功能等价性

---

## 三、三项方法的整体框架

将 RecCL、SANS、RecAug 整合为一个统一的训练框架：

```
训练流程：
1. 数据预处理阶段：
   - 对全量用户序列，计算序列难度、物品难度、预测难度（RecCL 指标）
   - 对每个正样本，用 LLM 生成 Hard Negatives（SANS）
   - 对每个用户序列，生成 2-3 个增广序列变体（RecAug）

2. 训练阶段：
   - 每个 epoch 开始时，根据当前训练步数更新采样权重（RecCL 调度）
   - 按权重采样 batch，batch 内包含原始序列 + 增广序列
   - 对每个样本，正样本为真实下一个物品，负样本按 SANS 三级分层构造
   - 损失函数为 SANS 加权的 InfoNCE + RecAug 的一致性正则项

3. 一致性正则项：
   - 原始序列和增广序列对同一用户的表征应当一致
   - 用 KL 散度约束两者的输出分布
```

---

## 四、预期贡献与差异化

### 4.1 相对于现有工作的差异化

| 现有工作 | 局限 | 本方案的改进 |
|---------|------|------------|
| P5/GenRec | 统一采样，无难度区分 | RecCL 三维渐进课程 |
| RecAI/RecLM | 随机负采样 | SANS 语义分层负采样 |
| E4SRec/BIGRec | 照搬 NLP 数据增广 | RecAug 推荐专用增广 |
| OnePiece | 指出样本工程重要但未探索 | 系统性的样本工程方案 |

### 4.2 学术贡献

- 首次系统性定义生成式推荐中的**样本工程**问题
- 提出三项推荐专用的样本构造方法，每项都有独立的创新性
- 三者的组合产生协同效应：RecCL 决定"何时学"，SANS 决定"学什么差异"，RecAug 决定"从哪些视角学"

### 4.3 工程价值

- 三项方法都是**训练阶段的改进**，不增加推理开销
- 除了 LLM 离线生成 hard negatives 和增广数据外，不需要额外硬件
- 可以与现有的任何 LLM 生成式推荐架构（P5、GenRec、RecAI、HLLM 等）即插即用地组合

---

## 五、数据集与技术栈

### 5.1 数据集选择：Steam

选用 **Steam 游戏推荐数据集**（UCSD Julian McAuley 收集），原因如下：

| 优势 | 说明 |
|------|------|
| **文本质量高** | 游戏有标题、描述、标签、开发商等丰富文本元数据，天然适合 LLM 语义建模 |
| **隐式意图丰富** | 游戏购买/游玩序列反映用户的品味偏好（如"偏爱独立游戏→突然入坑 3A"），序列中的意图转折对 RecCL 课程学习是极好的测试场景 |
| **长尾显著** | Steam 上有大量小众独立游戏（长尾物品），适合验证 SANS 对冷门物品推荐的效果 |
| **时间跨度大** | 用户游玩时长数据可天然用于 RecAug 的会话边界检测（相邻游戏的购买间隔 × 游玩时长的组合信号） |
| **学术认可度高** | Steam 在 RecSys/CIKM/WSDM 等顶会中广泛使用，评审认可度高 |
| **规模适中** | 比 Amazon 全集小、比 ML-1M 大，单卡可训练但足够展示方法优势 |

**Steam 数据集概览**：

| 统计项 | 数值 |
|--------|------|
| 用户数 | ~280K |
| 游戏数 | ~13K |
| 交互数（购买/游玩） | ~3.7M |
| 稀疏度 | ~99.9% |
| 字段 | user_id, game_id, game_title, purchase/play timestamp, hours_played, genre_tags, developer, publisher |

**下载地址**：[https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)

### 5.2 技术栈与成熟框架

采用 **OpenP5（训练管线）+ HuggingFace Transformers（模型层）+ RecBole（数据预处理）** 三位一体架构：

```
技术栈层级：
┌─────────────────────────────────────┐
│  数据预处理  →  RecBole             │  k-core过滤、序列构造、train/val/test划分
│  训练管线    →  OpenP5              │  prompt模板、multi-task batching、评估
│  模型层      →  HuggingFace T5/Llama │  预训练权重加载、分布式训练
│  样本工程    →  本项目的三项创新     │  RecCL / SANS / RecAug（插入到训练管线中）
└─────────────────────────────────────┘
```

**各框架的角色与用法**：

| 框架 | 用途 | 关键 API |
|------|------|---------|
| **RecBole** | Steam 原始 JSON → 用户序列 → k-core 过滤 → leave-one-out 划分 | `create_dataset()`, `data_preparation()` |
| **OpenP5** | 序列 → prompt 模板（如 "The user has played {games}. What game should be recommended next?"），统一 text-to-text 格式 | `PromptTemplate`, `P5Dataset` |
| **HuggingFace Transformers** | T5-small / T5-base 模型加载、训练循环、checkpoint 管理 | `AutoModelForSeq2SeqLM`, `Trainer` |
| **本项目** | RecCL 采样器、SANS 负样本构造器、RecAug 数据增广器 | 三个独立模块，即插即用 |

**框架参考仓库**：

| 仓库 | 地址 | 用途 |
|------|------|------|
| OpenP5 | [github.com/agiresearch/OpenP5](https://github.com/agiresearch/OpenP5) | 训练主框架，prompt 模板系统 |
| P5 (原始) | [github.com/jeykigung/P5](https://github.com/jeykigung/P5) | 经典 text-to-text 推荐范式参考 |
| RecBole | [github.com/RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) | 数据预处理管线 |
| RecBole2.0 | [github.com/RUCAIBox/RecBole2.0](https://github.com/RUCAIBox/RecBole2.0) | 扩展版（更多模型/Dataset） |
| SASRec.pytorch | [github.com/pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) | RecCL 中计算预测难度的协同过滤基线 |

### 5.3 基座模型选择

- **快速验证**：`google/flan-t5-small`（80M 参数）或 `google/flan-t5-base`（250M），单卡 RTX 3090 可在数小时内完成全量训练
- **进阶验证**：`google/flan-t5-large`（780M）或 `Qwen2.5-7B`（decoder-only），验证方法在大模型上的可扩展性
- **Steam 适配说明**：Steam 的游戏文本较长（标题 + 标签 + 描述），T5 的 512 token 输入窗口足够覆盖 10-15 个游戏的序列。若序列过长，RecAug 的意图保持截断可自然压缩

### 5.4 评估维度

| 维度 | 指标 | 说明 |
|------|------|------|
| 准确性 | NDCG@5, NDCG@10, Recall@10, HR@10 | 标准推荐指标 |
| 多样性 | ILS@10, Coverage@10 | 推荐列表的类目/标签覆盖度（Steam 有 genre_tags 字段） |
| 冷门物品 | Tail Recall@10 | 交互数 < 50 的游戏召回率 |
| 新颖性 | Novelty@10 | 推荐物品的平均流行度倒数 |
| 训练效率 | 收敛步数, GPU 时间 | 达到同等 NDCG 所需的训练资源 |
| 幻觉率 | OOD@10 | 生成不在物品库中的游戏 ID 的比例 |

### 5.5 消融实验设计

以 `flan-t5-base` + Steam + OpenP5 作为统一基座：

| 实验组 | 配置 | 验证目标 |
|--------|------|---------|
| **Base** | 纯 OpenP5，不做任何样本工程 | 基线 |
| **+RecCL** | 仅加入课程学习采样器 | RecCL 的有效性 |
| **+SANS** | 仅加入语义分层负采样 | SANS 的有效性 |
| **+RecAug** | 仅加入推荐专用增广 | RecAug 的有效性 |
| **+RecCL+SANS** | 课程学习 + 分层负采样 | 两者的协同效应 |
| **+All** | 三项方法完整组合 | 全面评估 |

预期每项单独都能带来提升，三项组合产生超过加性的协同增益。

### 5.6 关键假设验证

- **H1**：RecCL 能有效加速训练收敛（同等 NDCG@10 下训练步数减少 30%+）
- **H2**：SANS 能显著提升冷门游戏召回率（交互数 < 50 的游戏的 Tail Recall@10 提升 15%+）
- **H3**：RecAug 能在不引入噪声的前提下增强模型鲁棒性（增广序列与原始序列的推荐 Top-10 一致率 > 85%）
- **H4**：三项方法组合在冷门游戏上的提升幅度 > 热门游戏（说明样本工程主要惠及数据稀疏场景）

---

## 六、项目工程结构

### 6.1 目录规划

```
llm-rec-sample-engineering/
├── data/
│   ├── raw/                          # Steam 原始 JSON（不纳入版本控制）
│   │   └── Steam_reviews.json
│   ├── processed/                    # RecBole 预处理后的序列数据
│   │   ├── steam.train.txt           # 用户行为序列（user_id item_id_1 item_id_2 ...）
│   │   ├── steam.valid.txt
│   │   └── steam.test.txt
│   └── preprocess.py                 # Steam JSON → 序列格式 预处理脚本
│
├── sample_engineering/               # ★ 本项目的核心创新模块
│   ├── __init__.py
│   ├── rec_cl.py                     # RecCL: 三维课程学习采样器
│   │   ├── DifficultyScorer          #   计算序列/物品/预测三维难度
│   │   └── CurriculumSampler         #   按难度连续调度的 batch 采样器
│   ├── sans.py                       # SANS: 语义感知负样本分层
│   │   ├── HardNegativeGenerator     #   LLM 生成 hard negatives（离线缓存）
│   │   └── LayeredNegativeSampler    #   三级负样本按权重采样 + 加权 InfoNCE
│   └── rec_aug.py                    # RecAug: 推荐语义保持增广
│       ├── IntentPreservingTruncation #   意图保持的序列截断
│       ├── SessionBoundaryDetector   #   时间+LLM联合会话边界检测
│       ├── SessionPermutation        #   会话块随机重排
│       └── LLMGuidedSubstitution     #   LLM驱动的同意图物品替换
│
├── models/                           # 模型封装（薄封装，不修改架构）
│   ├── __init__.py
│   ├── base_p5.py                    # T5 encoder-decoder 基座（基于 HuggingFace）
│   └── config.py                     # 模型配置（hidden_size, num_layers 等）
│
├── utils/                            # 工具函数
│   ├── metrics.py                    # 评估指标（NDCG, Recall, ILS, Tail Recall, OOD）
│   ├── caching.py                    # LLM 生成结果的磁盘缓存（避免重复调用 LLM）
│   └── steam_utils.py                # Steam 特定工具（genre 解析、时长归一化等）
│
├── config/
│   ├── config.yaml                   # 主配置文件（模型、数据、训练参数）
│   └── steam_prompt_templates.yaml   # Steam 场景的 prompt 模板
│
├── scripts/
│   ├── preprocess.sh                 # 一键数据预处理
│   ├── train_base.sh                 # 训练基线模型
│   ├── train_full.sh                 # 训练完整模型（含三项样本工程）
│   └── evaluate.sh                   # 评估脚本
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Steam 数据探索分析
│   ├── 02_preprocessing.ipynb        # 预处理过程可视化
│   └── 03_result_analysis.ipynb      # 实验结果分析与可视化
│
├── trainer.py                        # 训练主逻辑（改编自 OpenP5，插入样本工程钩子）
├── evaluate.py                       # 独立评估脚本
├── requirements.txt
└── README.md
```

### 6.2 代码改动量估算

| 模块 | 预估代码量 | 说明 |
|------|-----------|------|
| `rec_cl.py` | ~200 行 | 三个难度 Scorer + 连续调度器 |
| `sans.py` | ~250 行 | LLM 调用 + embedding 检索 + 加权损失 |
| `rec_aug.py` | ~300 行 | 三种增广 + 自适应组合策略 |
| `trainer.py`（改动） | ~150 行 | 在 OpenP5 的训练循环中插入三个钩子 |
| `preprocess.py` | ~100 行 | Steam JSON → RecBole 格式 |
| `utils/` | ~150 行 | 指标、缓存、Steam 工具 |
| **总计** | **~1150 行** | 核心创新代码，不含 OpenP5/框架代码 |

### 6.3 三个方法在训练管线中的插入点

```
OpenP5 原始训练循环:
  for batch in dataloader:
      inputs = prompt_template(batch)      # prompt 模板化
      outputs = model(**inputs)            # 前向传播
      loss = cross_entropy(outputs, labels) # 标准损失
      loss.backward()                      # 反向传播

本项目的改进（三个钩子标记为 ★）:
  for batch in curriculum_sampler(data):   # ★ RecCL: 按难度采样
      batch = recaug_augment(batch)        # ★ RecAug: 增广序列变体
      inputs = prompt_template(batch)
      outputs = model(**inputs)
      loss_pos = cross_entropy(outputs, labels)
      loss_neg = sans_weighted_infonce(    # ★ SANS: 分层加权负采样
          outputs, labels, neg_weights
      )
      loss_consistency = kl_divergence(    # RecAug 一致性正则
          outputs_orig, outputs_aug
      )
      loss = loss_pos + loss_neg + λ * loss_consistency
      loss.backward()
```

### 6.4 LLM 调用成本控制

所有涉及外部 LLM 调用（GPT-4o-mini / DeepSeek）的步骤均采用离线预计算 + 磁盘缓存策略：

| 步骤 | LLM 用途 | 调用量 | 缓存策略 |
|------|---------|--------|---------|
| SANS Hard Negatives | 生成语义相似替代品 | 全量正样本 × 3 | 一次生成，存为 JSON，训练时读取 |
| RecAug 意图总结 | 判断每个物品的用户意图 | 全量物品 × 1 | 按 item_id 缓存，不同用户共享 |
| RecAug 物品替换 | LLM 驱动的同意图替代 | 采样部分序列 | 按 (item_id, intent) key 缓存 |
| RecAug 会话边界 | 意图分析辅助时间间隔 | 可选（时间间隔已够用） | 按用户缓存 |

> Steam ~13K 游戏，全量意图总结仅约 13K 次 LLM 调用，用 GPT-4o-mini 成本约 $3-5。Hard negatives 对每个正样本生成 3 个，按采样 50K 训练样本计算约 $15-25。总 LLM 成本控制在 $30 以内（一次性）。

---

## 七、潜在风险与应对

| 风险 | 可能性 | 应对策略 |
|------|--------|---------|
| LLM 生成 hard negatives 的成本过高 | 中 | 预生成并缓存，只在训练开始时做一次；可先用 DeepSeek-V3（便宜）验证，确认有效后再用 GPT-4o 提升质量 |
| Steam 游戏文本过长导致 T5 输入溢出 | 中 | RecAug 的意图保持截断可自然压缩；Steam 标签字段（genre_tags）本身已提供精简语义，优先使用标签替代长描述 |
| RecCL 的难度度量不准确 | 低 | 三个指标互为补充，且采用模糊调度而非硬切换，对单个指标的精度不敏感 |
| 三项方法简单叠加可能互相冲突 | 低 | 消融实验会揭示冲突点；如有冲突，用验证集调优三者权重 |
| 在更大的 LLM 基座上效果不显著 | 中 | 先在 T5-small/base 上充分验证，再迁移到 Qwen2.5-7B；小模型上有效果是大模型的基础 |

---

## 八、参考文献

1. Geng et al. (2022). Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). *RecSys 2022*.
2. Ji et al. (2024). GenRec: Large Language Model for Generative Recommendation. *ECIR 2024*.
3. Fu et al. (2025). OnePiece: Context Engineering Meets Generative Recommendation on Item ID Sequences. *Shopee Technical Report*.
4. Lin et al. (2025). Rec-R1: Bridging Large Language Models and Recommendation Systems via Reinforcement Learning. *TMLR 2025*.
5. Li et al. (2024). RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems. *WWW 2024 Companion*.
6. Wang et al. (2025). HLLM: Hierarchical Large Language Model for Sequential Recommendation. *ByteDance Technical Report*.
7. Wang et al. (2025). Lite-LLM4Rec: Rethinking Large Language Model Architectures for Sequential Recommendations. *IJCNLP-AACL 2025*.
8. Bengio et al. (2009). Curriculum Learning. *ICML 2009*.
9. Robinson et al. (2021). Contrastive Learning with Hard Negative Samples. *ICLR 2021*.
10. Wei & Zou (2019). EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks. *EMNLP-IJCNLP 2019*.
