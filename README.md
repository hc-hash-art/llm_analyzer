# Model Kernel Analyzer

静态分析深度学习模型项目源码，生成算子到 kernel 实现路径 (`impl_path`) 的映射。

当前版本目标：
- 基于 AST 的源码级分析（不执行模型代码）
- 从入口脚本开始，递归追踪调用链（函数、方法、类实例化）
- 重点追踪 `forward` / `__call__`
- 提取 `torch` / `torch.nn.functional(F)` / `torch.nn` 算子调用
- 输出结构化 JSON，包含调用图与算子 `impl_path`

## 快速开始

```bash
cd /home/hc/llm_analyzer
python -m src.model_kernel_analyzer.cli --config configs/qwen3.yaml
```

查看**以 `forward` 为入口的调用链**（终端树状 + JSON）可用最小样例：

```bash
python -m src.model_kernel_analyzer.cli --config configs/toy_forward.yaml
```

输出文件默认写入：
- `outputs/qwen3_analysis.json`  
- 样例：`outputs/toy_forward_analysis.json`（字段 `call_chain_forest` 为结构化调用树）

仅写入 JSON、不打印树：

```bash
python -m src.model_kernel_analyzer.cli --config configs/toy_forward.yaml --no-print-chains
```

## 自动解析 torch/transformers 并安装（可选）

如果你希望工具在分析时使用“与模型工程一致”的 `torch`/`transformers` 版本，可以先自动解析版本、创建 venv、安装依赖并在该 venv 里重新跑分析。

### 1) Dry-run：只解析版本

```bash
python -m src.model_kernel_analyzer.setup_env \
  --project-root /home/hc/Qwen3 \
  --analyze-config /home/hc/llm_analyzer/configs/qwen3.yaml \
  --dry-run
```

### 2) 真正执行：创建 venv + 安装 + 重新分析

```bash
python -m src.model_kernel_analyzer.setup_env \
  --project-root /home/hc/Qwen3 \
  --analyze-config /home/hc/llm_analyzer/configs/qwen3.yaml
```

默认会从：
1. `project_root/requirements.txt`
2. `project_root/Dockerfile*`
3. `project_root/README*.md`
中解析 `torch` / `transformers` 的版本/约束，并优先使用 `torch` CPU wheel 源（可通过 `--torch-index-url` 改）。

安装完成后，工具会把 `transformers` 的源码目录也加入静态索引扫描（这样调用链里就能出现 transformers 内部 `*.forward`，并继续下钻到 `torch.*` 算子与 kernel 映射）。

### 仅算子 / kernel 相关调用链（默认）

`analysis.op_chain_only: true`（默认）时，终端与 JSON 中的 `call_chain_forest` **会跳过** `copy/gc/os/argparse/threading` 等胶水调用，  
对中间结点做**路径嫁接**（`via` 里用 `»` 表示穿过的跳过结点），只保留：

- `torch` / `triton` / `functorch`
- `transformers`（便于跟模型；若只要纯算子可排除）
- 工程内 `*.forward`（子模块前向）

排除整库或子模块示例：

```yaml
analysis:
  op_chain_only: true
  op_chain_exclude_modules:
    - transformers.trainer_utils   # 去掉 set_seed 等
    # - transformers               # 只要 torch，整库不展示
```

需要额外算作「算子相关」的第三方模块（如 `numpy`）：

```yaml
  op_chain_include_modules:
    - numpy
```

关闭过滤（恢复完整 Python 调用树）：

```yaml
  op_chain_only: false
```

## 面向扩展的设计

核心步骤解耦为可独立测试模块：
- `entry_discovery.py`：入口发现
- `project_index.py`：项目符号索引（函数/类/方法）
- `call_graph.py`：调用链构建（call graph + class graph）
- `op_extractor.py`：算子调用抽取
- `kernel_mapper.py`：算子映射到 `impl_path`
- `pipeline.py`：流程编排

## 说明

`/home/hc/Qwen3` 示例仓库主要是 README、示例和评测脚本，并不包含完整模型算子源码。  
因此第一版会：
- 对仓库内可见 Python 代码做“行精确”静态追踪
- 对外部库调用（如 `transformers`、`torch`）记录源码路径（若可定位）
- 对 PyTorch kernel 具体实现路径先给出启发式结果与来源说明，后续可接入更强规则库
# llm_analyzer
