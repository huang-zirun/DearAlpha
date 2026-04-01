# DearAlpha

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**WorldQuant Brain Alpha 因子智能挖掘框架**

设计哲学：**AI 生成信号，代码枚举验证，人做最终判断**

</div>

---

## 📖 项目简介

DearAlpha 是一个面向 WorldQuant Brain 平台的 alpha 因子自动化挖掘框架。它融合了人工智能生成、系统化枚举、贝叶斯优化等多种策略，帮助量化研究者高效发现高质量的 alpha 因子。

### 核心特性

- 🤖 **AI 裸信号生成** - 基于 LLM 的经济学直觉生成创新 alpha 表达式
- 🔄 **三阶递进流水线** - 移植经典 day1→day2→day3 挖掘逻辑，支持断点续跑
- 📊 **笛卡尔积穷举** - 系统化覆盖内置模板库的所有参数组合
- 🎯 **分层剪枝** - 两阶段粗筛+精调，减少约 50% 无效模拟
- 🧠 **贝叶斯优化** - 使用 Optuna TPE 采样器高效搜索最优参数
- 💾 **智能断点管理** - 支持中断恢复，避免重复计算

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- WorldQuant Brain 账号
- LLM API 密钥（OpenRouter / Anthropic / Ollama）

### 安装

```bash
# 使用 uv（推荐）
uv venv
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

### 配置

```bash
# 1. 配置 WorldQuant Brain 凭证
cp credential.txt.example credential.txt
# 编辑 credential.txt，填入 ["your_email@worldquant.com", "your_password"]

# 2. 配置 LLM 和挖掘参数
cp configs/default.yaml config.yaml
# 编辑 config.yaml，填入 llm.api_key（OpenRouter key）
# 或设置环境变量: export OPENROUTER_API_KEY=sk-or-...
```

---

## 💡 使用指南

### 6 种挖掘模式

```bash
# 🤖 AI 模式 - LLM 生成裸信号（发现式）
python mine.py ai

# 🔄 Pipeline 模式 - 三阶递进流水线（day1→day2→day3）
python mine.py pipeline

# 📊 Template 模式 - 笛卡尔积穷举内置模板库
python mine.py template

# 🎯 Layered 模式 - 分层剪枝（大字段空间推荐）
python mine.py layered

# 🧠 Bayesian 模式 - 贝叶斯优化数值参数（精调推荐）
python mine.py bayesian

# 📤 Submit 模式 - 提交通过质量门控的 alpha
python mine.py submit
```

### 常用选项

```bash
# AI 模式选项
python mine.py ai --rounds 5               # 跑 5 轮，每轮 5 个信号
python mine.py ai --theme "Momentum"       # 指定经济学主题

# Pipeline 模式选项
python mine.py pipeline                                    # 使用 config.yaml 配置
python mine.py pipeline --fields "close,volume,returns"   # 指定字段
python mine.py pipeline --field-prefix "anl4" --prune-keep 5

# Layered 模式选项
python mine.py layered --keep-fields 5

# Bayesian 模式选项
python mine.py bayesian --n-trials 80

# 提交选项
python mine.py submit --dry-run            # 预览可提交的 alpha
python mine.py submit                      # 实际提交

# 通用选项
python mine.py --verbose ai --rounds 1     # 开启 debug 日志
```

---

## 📚 模式详解

### `ai` — LLM 裸信号生成

让 LLM 基于经济学直觉提出 alpha 假设，直接提交模拟，不受模板约束。适合发现全新结构。

- 每轮自动循环 10 个经济学主题（价值、动量、质量、情绪、波动率……）
- 可使用 `--theme` 指定特定主题
- 支持多厂商 LLM 后端（OpenRouter、Anthropic、Ollama）

### `pipeline` — 三阶递进流水线

移植自 WQ挖掘脚本 的 day1/day2/day3 逻辑，核心是**递进展开 + 层层剪枝**：

```
Stage 1 (day1)
  fields × ts_ops → first_order_factory
  → 模拟 → 通过 stage1_filter → prune（同字段保留前N个）
      ↓
Stage 2 (day2)
  stage1 通过结果 × group_ops → group_second_order_factory
  → 模拟 → 通过 stage2_filter → prune
      ↓
Stage 3 (day3)
  stage2 通过结果 × open/exit events → trade_when_factory
  → 模拟 → 通过 stage3_filter
```

**字段来源**（按优先级）：
1. CLI `--fields close,volume,...`
2. `config.yaml` → `mining.pipeline.fields`
3. `config.yaml` → `mining.pipeline.dataset_id`（从 WQ API 动态拉取）

### `template` — 笛卡尔积穷举

对内置模板库中的每个模板，枚举所有参数组合并全量测试。适合搜索空间小、需要完整覆盖的场景。支持断点续跑。

### `layered` — 分层剪枝

两阶段减少模拟量：
- **Pass 1**：所有字段 × 少量代表性窗口 → 按 |Sharpe| 排名
- **Pass 2**：top-K 字段 × 完整窗口网格

相比全量笛卡尔积节省约 50% 模拟次数。

### `bayesian` — 贝叶斯优化

使用 Optuna TPE 采样器搜索数值/离散参数空间。建立"哪个区域 Sharpe 高"的概率模型，自动引导采样，适合窗口、decay 等连续参数精调。

### `submit` — 提交

从 `results/passing_alphas.jsonl` 读取未提交的 alpha，检查 PROD_CORRELATION，在每日限额内批量提交。

---

## 📁 项目结构

```
DearAlpha/
├── dear_alpha/              # 核心模块
│   ├── brain.py            # WorldQuant Brain REST API 客户端
│   ├── generator.py        # AI 裸信号生成器（多厂商 LLM 支持）
│   ├── factories.py        # 表达式工厂函数
│   │                         - first_order_factory
│   │                         - group_second_order_factory
│   │                         - trade_when_factory
│   │                         - prune
│   ├── miner.py            # 四种挖掘算法 + Checkpoint 断点管理
│   │                         - TemplateMiner
│   │                         - LayeredMiner
│   │                         - BayesianMiner
│   │                         - PipelineMiner
│   ├── evaluator.py        # 质量门控（Sharpe / Fitness / Turnover 过滤）
│   └── submitter.py        # 结果持久化 + 限速提交
├── configs/
│   └── default.yaml        # 默认配置模板
├── WQ挖掘脚本/              # 原始参考脚本（day1/day2/day3）
│   ├── day1.py
│   ├── day2.py
│   ├── day3.py
│   └── machine_lib.py
├── mine.py                 # CLI 入口
├── requirements.txt        # 依赖列表
└── README.md              # 本文件
```

---

## 📊 输出结果

| 文件 | 内容 |
|------|------|
| `results/passing_alphas.jsonl` | 通过质量门控的 alpha（每行一个 JSON） |
| `results/progress.json` | 断点进度（pipeline / template 续跑用） |
| `results/dear_alpha.log` | 完整运行日志 |

### Alpha 记录格式

```json
{
  "expression": "group_rank(ts_mean(close, 22), densify(sector))",
  "alpha_id": "abc123...",
  "metrics": {
    "sharpe": 1.38,
    "fitness": 1.12,
    "turnover": 0.25,
    "margin": 0.15,
    "long_count": 150,
    "short_count": 148
  },
  "recommended_decay": 4,
  "source": "pipeline",
  "stage": "stage2",
  "saved_at": "2026-04-01T10:30:00"
}
```

---

## ⚙️ 配置详解

### LLM 后端切换

在 `config.yaml` 中修改 `llm` 部分：

```yaml
# OpenRouter（多模型，默认）
llm:
  provider: "openrouter"
  api_key: "sk-or-..."
  model: "anthropic/claude-3.5-sonnet"

# 本地 Ollama（零成本）
llm:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model: "deepseek-r1:8b"

# 直接 Anthropic API
llm:
  provider: "anthropic"
  api_key: "sk-ant-..."
  model: "claude-sonnet-4-6"
```

### Pipeline 详细配置

```yaml
mining:
  pipeline:
    # 字段来源（三选一）
    fields: [close, volume, returns, vwap]   # 直接指定
    # dataset_id: "analyst4"                 # 从 WQ API 动态拉取

    field_prefix: ""      # prune 时的字段标识前缀
    prune_keep: 5         # 每个字段最多保留几个表达式进入下一阶
    init_decay: 6

    # 各阶质量阈值（可比最终 gate 宽松）
    stage1_filter:
      min_sharpe: 1.0
      min_fitness: 0.7
    stage2_filter:
      min_sharpe: 1.3
      min_fitness: 1.0
    stage3_filter:
      min_sharpe: 1.3
      min_fitness: 1.0
```

### 质量门控参数

```yaml
quality:
  min_sharpe: 1.25
  min_fitness: 1.0
  min_turnover: 0.01
  max_turnover: 0.70
  min_long_count: 50
  min_short_count: 50
```

---

## 🎯 算法对比与推荐工作流

| 模式 | 搜索策略 | 适合场景 | 断点续跑 |
|------|---------|---------|---------|
| `ai` | LLM 生成 | 发现新结构，无模板约束 | — |
| `pipeline` | 递进展开 + 剪枝 | 系统性三阶挖掘，字段集较大 | ✅ |
| `template` | 笛卡尔积 | 搜索空间小，要求全覆盖 | ✅ |
| `layered` | 两阶段粗筛 + 精调 | 字段多，先淘汰坏字段 | — |
| `bayesian` | TPE 贝叶斯 | 数值参数精调，自适应收敛 | — |

### 推荐工作流

```bash
# 1. 用 pipeline 系统性跑三阶展开
python mine.py pipeline

# 2. 对最优字段做贝叶斯精调
python mine.py bayesian --n-trials 80

# 3. AI 发现完全不同结构的新想法
python mine.py ai --rounds 5

# 4. 检查并提交
python mine.py submit --dry-run
python mine.py submit
```

---

## 🔧 内置模板库

框架包含精心策划的模板库，涵盖常见 alpha 结构：

| 类别 | 模板示例 |
|------|---------|
| **动量** | `group_rank(ts_mean({field}, {window}), densify(sector))` |
| **反转** | `-rank(ts_returns(close, {window}))` |
| **波动率** | `-rank(ts_std_dev(returns, {window}))` |
| **价值** | `-rank(winsorize(ts_backfill({field}, 120), std=4))` |
| **成交量** | `-rank(ts_mean({field}, {window}) / ts_std_dev({field}, {window}))` |
| **相关性** | `rank(ts_corr(returns, {field}, {window}))` |

---

## 📝 依赖项

```
requests>=2.31.0
pyyaml>=6.0
optuna>=3.0.0  # Bayesian 模式需要
pandas>=2.0.0  # Pipeline 动态拉取字段需要
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！请确保：

1. 代码符合 PEP 8 规范
2. 新功能包含适当的测试
3. 更新相关文档

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- WorldQuant Brain 平台提供的 API 支持
- 原始 WQ挖掘脚本 的算法启发
- 开源社区的优秀工具（Optuna、Requests 等）

---

<div align="center">

**Happy Alpha Mining! 🚀**

</div>
