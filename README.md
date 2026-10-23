# Logic2Text / Spider：本地 Auto-CoT 推理与 BLEC 测试

这个精简包只包含两个数据集的推理、示例构建和 BLEC 评价所需文件。沿用根目录的 `run_inference.py`、`run_demo.py`，保留 Logic2Text 的 `logic_prompt_rule` 和 Spider 的 `sql_prompt`，通过 `local_decoder.py` 调用本地模型。

## 1. 安装与模型

在本仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

本地模型服务需要事先启动，并提供 OpenAI 兼容接口。默认使用：

- 地址：`http://localhost:11434/v1`
- 模型：`qwen3.6:latest`
- 温度：`0`
- 输出上限：`10240` tokens，沿用原 Logic2Text / Spider 限制。

快捷脚本默认传入 `reasoning_effort=none`，关闭模型原生 thinking，保留提示词中的 “Let's think step by step”。这是为了避免原生 thinking 耗尽输出预算、最终答案为空。需要保留服务端默认行为时：

```bash
export REASONING_EFFORT=default
```

修改模型或地址：

```bash
export LOCAL_MODEL_NAME='你的模型名称'
export LOCAL_MODEL_BASE_URL='http://localhost:11434/v1'
```

`run_demo.py` 默认使用仓库内的 `models/all-MiniLM-L6-v2` 编码器，不需要首次运行下载。包含一套原始 `model.safetensors` 权重、分词器和配置，共约 91.6 MB；未包含重复的 PyTorch BIN 权重。下载仓库时请保留该目录。需要换用其他完整的 sentence-transformers 模型目录时设置：

```bash
export ENCODER='/你的路径/编码器模型目录'
```

本包不包含 Qwen 大模型权重或虚拟环境。推理沿用 PyTorch DataLoader，示例构建使用句向量和 KMeans，因此保留 `torch`、`numpy`、`sentence-transformers`、`scikit-learn`；BLEC 还需要 `pandas` 和 `nltk`。当前固定测试模板的 Spider 评价不需要下载 NLTK 语料。可选绘图需要额外安装 `matplotlib`，默认不绘图。已验证环境的直接依赖版本记录在 `environment-tested.json`，安装文件不锁定 CUDA 构建。

## 2. 最短运行流程

### Zero-shot-CoT 推理与评价

Logic2Text：

```bash
bash run.sh logic2text zero
bash run.sh logic2text eval-zero
```

Spider：

```bash
bash run.sh spider zero
bash run.sh spider eval-zero
```

先试运行少量样本：

```bash
LIMIT=3 bash run.sh logic2text zero
bash run.sh logic2text eval-zero
```

`LIMIT=3` 的评价分母是 3，只能验证流程；正式测试不设置 `LIMIT`，运行全量数据。同一个输出目录再次运行会覆盖之前结果，正式实验前请自行备份。

### Auto-CoT 示例构建、推理与评价

先在训练集上生成候选推理结果，再聚类选取示例，最后在测试集上推理：

```bash
bash run.sh logic2text train
bash run.sh logic2text demo
bash run.sh logic2text auto
bash run.sh logic2text eval-auto

bash run.sh spider train
bash run.sh spider demo
bash run.sh spider auto
bash run.sh spider eval-auto
```

`train` 默认使用随机种子 1 抽取的 256 条训练样本，调用与测试相同的模型生成候选；`demo` 不调用大语言模型，而是用编码器和 KMeans 聚类，按原有长度与非空条件挑选至多 8 条示例。它不会按 BLEC 结果或 `match` 字段筛选正确样本。若选不出示例，可增加训练候选：

```bash
TRAIN_LIMIT=1024 bash run.sh logic2text train
bash run.sh logic2text demo
```

`TRAIN_LIMIT=0` 表示使用全量训练集。不要用测试集的预测结果构建正式测试示例。本包不附带历史 demo，避免混用来源。

**以前的 `relogic_logic2text_train` 示例**是把训练集参考文本同时写入 `prediction`、`generated_critique` 和 `label` 后再聚类得到的，属于训练集参考答案示例；上面的 `train → demo` 流程则使用模型生成的候选，两者的示例来源不同。

## 3. 直接使用原入口

快捷脚本只是组合参数。需要调整生成长度等设置时，直接运行原入口。以下以 Logic2Text 为例；Spider 对应替换数据集名和输出路径即可。

```bash
# Zero-shot-CoT；默认读取附带的 test.json
python3 run_inference.py \
  --dataset relogic_logic2text --method zero_shot_cot \
  --model qwen3.6:latest --reasoning_effort none \
  --output_dir output/logic2text_zero

# 训练集候选；--dataset_path 是本精简包新增的输入覆盖参数
python3 run_inference.py \
  --dataset relogic_logic2text --method zero_shot_cot \
  --dataset_path dataset/relogic_data/logic2text/raw/train.json \
  --limit_dataset_size 256 \
  --model qwen3.6:latest --reasoning_effort none \
  --output_dir output/logic2text_train

# 构建示例；离线时追加 --encoder /完整模型目录
python3 run_demo.py \
  --task relogic_logic2text \
  --pred_file output/logic2text_train/relogic_logic2text.jsonl \
  --demo_save_dir demos/logic2text.json

# Auto-CoT
python3 run_inference.py \
  --dataset relogic_logic2text --method auto_cot \
  --demo_path demos/logic2text.json \
  --model qwen3.6:latest --reasoning_effort none \
  --output_dir output/logic2text_auto

# BLEC
python3 evaluate_blec.py --dataset logic2text \
  --input output/logic2text_auto/relogic_logic2text.jsonl
```

原入口不传 `--reasoning_effort` 时保留服务端行为；它与快捷脚本的默认设置有这一处区别。多模型服务可通过 `--model` 明确指定；只有一个模型时原入口支持自动识别。端点可通过 `--base_url` 指定。

## 4. 输出与评价口径

以 Logic2Text Auto-CoT 为例：

| 文件 | 内容 |
| --- | --- |
| `demos/logic2text.json` | 聚类得到的示例 |
| `output/logic2text_auto/relogic_logic2text.jsonl` | 问题、参考文本、模型输出、提取答案及结束原因 |
| `output/logic2text_auto/relogic_logic2text_log_for_evaluation.json` | BLEC 可读的逻辑表达式与预测文本 |
| `output/logic2text_auto/relogic_logic2text_blec.json` | BLEC 汇总与逐条错误 |

BLEC 汇总中 `passed / total` 是一致性通过率，`blec_pass_rate` 为 0–1 的比例。模型输出中的 `match` 只是预测与参考文本的严格字符串相等判断，不能替代 BLEC。

- Logic2Text：仅对操作词、比较词等自然语言关键词使用不区分大小写的匹配；沿用当前工作区的规则修改及测试，数值匹配规则不变。
- Spider：沿用现有规则，使用附带的测试 SQL 模板；`missing_sql_template` 和空预测计为失败。
- BLEC 通过表示规则未发现已知错误，不保证语义完全正确，也不直接等于基于人工一致性真值标签的检测准确率。
- 当前原入口在整轮推理结束后保存结果；若请求中途异常会退出，该轮未完成结果不会自动保存。此包不实现断点续跑，勿依赖原有 `--resume_id` 对这两个数据集续跑。

## 5. 数据与文件范围

数据来自当前工作区的 reLogic 数据副本，只保留这两个任务实际读取的字段，顺序和值均未改变：

| 数据集 | 训练集 | 测试集 | 保留字段 |
| --- | ---: | ---: | --- |
| Logic2Text | 8566 | 1092 | `logic_str`, `sent` |
| Spider | 5600 | 1034 | `query`, `question` |

不需要表格原文、数据库文件、SQL 执行环境、图结构、mutation 数据或开发集，因为这里进行的是逻辑/SQL 到自然语言的生成与 BLEC 规则评价。数据条数和文件校验值见 `dataset/manifest.json`。Spider SQL 模板随包提供，用于附带测试集；自定义 SQL 需要通过 `evaluate_blec.py --template` 提供对应模板。

```text
run_inference.py / run_demo.py / utils.py / local_decoder.py
run.sh                       简短运行命令
evaluate_blec.py              批量 BLEC 评价
requirements.txt              合并后的必要依赖
BLEC-main/BLEC/               两个任务的 BLEC 实现
BLEC-main/tests/              Spider 测试模板与关键词大小写测试
dataset/relogic_data/         精简后的训练集、测试集
dataset/manifest.json         数据来源校验信息
models/all-MiniLM-L6-v2/      随仓库提供的本地编码器
licenses/                    上游许可证与声明
THIRD_PARTY.md                上游来源与本包修改说明
```

未包含其他任务数据、远程付费模型后端、历史实验、日志、冗余缓存或密钥。原入口文件中保留了原有其他任务分支，但本包只支持表中的两个数据集，不包含其他任务依赖与数据。

## 6. 验证

安装依赖后，可在不启动本地模型的情况下运行接口模拟集成检查：

```bash
python3 tests/smoke_test.py
```

此检查使用模拟模型验证两个任务的 Zero-shot-CoT、Auto-CoT、训练输入覆盖与 BLEC 流程，不代表真实模型效果，也不下载编码器。真实 `run_demo.py` 默认读取随仓库提供的编码器。
