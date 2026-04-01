# DearAlpha

挖掘 WorldQuant Brain 的 alpha 因子。

设计原则：**AI 做信号生成，代码做枚举验证，人做最终判断。**

---

## 安装

```bash
uv venv
uv pip install -r requirements.txt
```

## 配置

```bash
cp credential.txt.example credential.txt
# 填入 ["email@worldquant.com", "password"]

cp configs/default.yaml config.yaml
# 填入 llm.api_key（OpenRouter key）
# 或设置环境变量: export OPENROUTER_API_KEY=sk-or-...
```

---

## 使用

### 6 种挖掘模式

```bash
python mine.py ai        # LLM 生成裸信号（发现式）
python mine.py pipeline  # 三阶递进流水线（day1→day2→day3）
python mine.py template  # 笛卡尔积穷举内置模板库
python mine.py layered   # 分层剪枝（大字段空间推荐）
python mine.py bayesian  # 贝叶斯优化数值参数（精调推荐）
python mine.py submit    # 提交通过质量门控的 alpha
```

### 常用选项

```bash
# AI 模式
python mine.py ai --rounds 5               # 跑 5 轮，每轮 5 个信号
python mine.py ai --theme "Momentum"       # 指定经济学主题

# Pipeline 模式
python mine.py pipeline                                    # 用 config.yaml 配置
python mine.py pipeline --fields "close,volume,returns"   # 指定字段
python mine.py pipeline --field-prefix "anl4" --prune-keep 5

# Layered 模式
python mine.py layered --keep-fields 5

# Bayesian 模式
python mine.py bayesian --n-trials 80

# 提交
python mine.py submit --dry-run            # 预览哪些 alpha 可提交
python mine.py submit

# 通用
python mine.py --verbose ai --rounds 1     # 开启 debug 日志
```

---

## 模式详解

### `ai` — LLM 裸信号生成

让 LLM 基于经济学直觉提出 alpha 假设，直接提交模拟，不受模板约束。适合发现新结构。

每轮自动循环10个经济学主题（价值、动量、质量、情绪、波动率……），可用 `--theme` 指定。

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

每阶都有独立断点（`results/progress.json`），中断后自动从断点恢复，不重跑已完成的模拟。

字段来源（按优先级）：
1. CLI `--fields close,volume,...`
2. `config.yaml` → `mining.pipeline.fields`
3. `config.yaml` → `mining.pipeline.dataset_id`（从 WQ API 动态拉取）

### `template` — 笛卡尔积穷举

对内置模板库中的每个模板，枚举所有参数组合并全量测试。适合搜索空间小、需要完整覆盖的场景。支持断点续跑。

### `layered` — 分层剪枝

两阶段减少模拟量：
- Pass 1：所有字段 × 少量代表性窗口 → 按 |Sharpe| 排名
- Pass 2：top-K 字段 × 完整窗口网格

相比全量笛卡尔积节省约 50% 模拟次数。

### `bayesian` — 贝叶斯优化

用 Optuna TPE 采样器搜索数值/离散参数空间。会建立"哪个区域 Sharpe 高"的概率模型，自动引导采样，适合窗口、decay 等连续参数精调。

### `submit` — 提交

从 `results/passing_alphas.jsonl` 读取未提交的 alpha，检查 PROD_CORRELATION，每日限额内批量提交。

---

## 结果

| 文件 | 内容 |
|------|------|
| `results/passing_alphas.jsonl` | 通过质量门控的 alpha（每行一个 JSON） |
| `results/progress.json` | 断点进度（pipeline / template 续跑用） |
| `results/dear_alpha.log` | 完整运行日志 |

每条 alpha 记录格式：
```json
{
  "expression": "group_rank(ts_mean(close, 22), densify(sector))",
  "alpha_id": "abc123...",
  "metrics": {"sharpe": 1.38, "fitness": 1.12, "turnover": 0.25, ...},
  "recommended_decay": 4,
  "source": "pipeline",
  "stage": "stage2",
  "saved_at": "2026-04-01T10:30:00"
}
```

---

## 项目结构

```
DearAlpha/
├── dear_alpha/
│   ├── brain.py       # WorldQuant Brain REST API 客户端
│   ├── generator.py   # AI 裸信号生成器（多厂商 LLM）
│   ├── factories.py   # 表达式工厂函数（first_order / group / trade_when / prune）
│   ├── miner.py       # 四种挖掘算法 + Checkpoint 断点管理
│   ├── evaluator.py   # 质量门控（Sharpe / Fitness / Turnover 过滤）
│   └── submitter.py   # 结果持久化 + 限速提交
├── configs/
│   └── default.yaml   # 默认配置
├── mine.py            # CLI 入口
└── requirements.txt
```

---

## 算法对比

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

## LLM 后端切换

在 `config.yaml` 中修改 `llm` 部分：

```yaml
# OpenRouter（多模型，默认）
llm:
  provider: "openrouter"
  api_key: "sk-or-..."
  model: "openai/gpt-4o"

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

---

## Pipeline 详细配置

`config.yaml` 中 `mining.pipeline` 部分：

```yaml
mining:
  pipeline:
    # 字段来源（三选一）
    fields: [close, volume, returns, vwap]   # 直接指定
    # dataset_id: "analyst4"                 # 从 WQ API 动态拉取

    field_prefix: "anl4"  # prune 时的字段标识前缀
    prune_keep: 5          # 每个字段最多保留几个表达式进入下一阶
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

---

## 质量门控参数

在 `config.yaml` 的 `quality` 部分调整最终过滤阈值：

```yaml
quality:
  min_sharpe: 1.25
  min_fitness: 1.0
  min_turnover: 0.01
  max_turnover: 0.70
```
