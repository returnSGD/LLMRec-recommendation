# LLM-Rec Goodreads Cross-Dataset Validation

## 硬件要求
- GPU: Tesla V100 16GB (or higher)
- RAM: 32GB+
- Disk: 20GB free (data ~6.1GB + checkpoints ~5-10GB)

## 环境配置

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 设置 DeepSeek API Key (SANS/RecAug 需要)
export DEEPSEEK_API_KEY="your_key_here"
```

## 快速开始

### 完整流程 (预处理 → 训练 → 评估)
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 仅预处理数据
```bash
python preprocess_goodreads.py \
    --data_dir data \
    --output_dir data/goodreads_processed \
    --k_core 5 --min_seq_len 5 --max_seq_len 50
```

### 仅训练 Baseline
```bash
python trainer.py \
    --config config/config_5090.yaml \
    --mode base \
    --data_dir data/goodreads_processed \
    --output_dir checkpoints/baseline \
    --device cuda
```

### 仅训练 Full (RecCL + SANS + RecAug)
```bash
python trainer.py \
    --config config/config_5090.yaml \
    --mode full \
    --data_dir data/goodreads_processed \
    --output_dir checkpoints/full \
    --device cuda
```

### 消融实验
```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

### 单独评估
```bash
python evaluate.py \
    --checkpoint checkpoints/baseline/final_model.pt \
    --data_dir data/goodreads_processed \
    --config config/config_5090.yaml \
    --batch_size 16 \
    --top_k 5 10 20 \
    --device cuda \
    --output results/metrics.json
```

## 配置说明 (config/config_5090.yaml)

| 参数 | 值 | 说明 |
|------|-----|------|
| base_model | google/flan-t5-base | ~250M 参数, 适合 16GB |
| batch_size | 24 | V100 16GB 安全值 |
| gradient_accumulation | 2 | 有效 batch = 48 |
| fp16 | true | V100 Tensor Cores 加速 |
| epochs | 10 | 充分训练 |
| max_seq_length | 512 | 长文本支持 |

## 目录结构

```
deploy_5090/
├── data/                          # 原始 Goodreads 数据
├── config/config_5090.yaml        # V100 优化配置
├── models/                        # T5 模型封装
├── sample_engineering/            # RecCL, SANS, RecAug
├── utils/                         # 工具函数
├── scripts/                       # 消融实验脚本
├── preprocess_goodreads.py        # 数据预处理
├── trainer.py                     # 训练脚本
├── evaluate.py                    # 评估脚本
├── run_pipeline.sh                # 一键流程
├── run_ablation.sh                # 消融实验
└── requirements.txt               # Python 依赖
```
