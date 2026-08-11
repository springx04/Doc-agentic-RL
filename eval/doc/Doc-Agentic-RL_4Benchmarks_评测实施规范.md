# Doc-Agentic-RL：DocVQA 2026 / LongDocURL / MP-DocVQA / DUDE 评测实施规范

> 目标：本文件用于直接交给代码 Agent，在现有 `springx04/Doc-agentic-RL` 项目上接入四个公开文档理解 benchmark，并完成可复现的评测。  
> 本规范优先满足：**不泄漏 GT、不重复实现已有 Agent、不下载无用数据、不预渲染整套 PDF、不把训练 reward 当官方 benchmark 分数、总磁盘控制在约 100 GB 内。**
>
> 核实日期：2026-08-10。

---

## 0. 必须遵守的总原则

### 0.1 只评以下四个 benchmark

1. **DocVQA 2026 validation**
2. **LongDocURL public benchmark**
3. **MP-DocVQA validation**
4. **DUDE validation**

不要额外下载或接入：

- DocVQA 2026 test；
- MP-DocVQA test；
- DUDE train / test 的评测样本；
- LongDocURL 的预渲染 PNG 包；
- 任意 benchmark 的预计算 OCR；
- 任意 benchmark 官方 baseline 模型权重；
- 8B 以上模型或所谓“8B 以上赛道数据”。

特别注意：**DocVQA 2026 的 “Up to 8B” 是参赛系统参数量类别，不是独立数据 split。**  
所有参数类别使用同一 validation/test 数据。因此，本项目 Qwen3-VL 4B/8B 的本地评测只下载 **公开带答案的 validation** 即可，不存在“≤8B 专用数据文件”。

---

## 0.2 不改变现有 Agent 推理协议

现有项目的 document-agent 推理协议保持不变：

- 使用项目已有 document tools；
- 使用项目已有 `generate_with_retool.generate`；
- 最终答案继续输出：

```text
<final>ANSWER</final>
```

不要为了适配 benchmark 将项目主协议改成：

```text
FINAL ANSWER: ...
```

DocVQA 2026 官方 evaluator 要求 `FINAL ANSWER:`，应当在**离线评分适配器**里把项目已经抽取出的 `<final>` 内容包装成：

```text
FINAL ANSWER: {prediction}
```

然后送入官方 evaluator。

---

## 0.3 不把项目 reward 当论文 benchmark 主分数

项目当前 `document_reward.py` 中的 `anls` / `quality` 是训练与 rollout 诊断指标，其中有项目自己的 normalization。论文中的 benchmark 主分数必须用对应 benchmark 官方/兼容官方的 scorer 独立计算。

因此每次评测分两层：

```text
Agent rollout
    │
    ├─ project reward / process diagnostics
    │      └─ 只用于 Agent 行为分析
    │
    └─ extracted <final> answer
           │
           └─ benchmark official scorer
                  └─ 论文主结果
```

---

## 0.4 GT 与 evidence 不能进入 Agent 可见状态

以下信息只允许保存在评测 sidecar 中，**禁止放入模型 prompt，也不要写入可能被 generation logic 消费的导航字段**：

- gold answer page；
- evidence pages；
- evidence bbox；
- answer type；
- gold locating coordinates；
- detailed evidence；
- gold page index。

尤其不要给 MP-DocVQA eval row 写项目已有的 `metadata["answer_page"]`，因为项目 rollout/evidence 逻辑已经认识该字段。  
应改存：

```text
gold/mpdocvqa.jsonl
```

并在 rollout 完成后进行 post-hoc 统计。

---

# 1. 当前项目中必须复用的组件

以当前公开仓库 `springx04/Doc-agentic-RL` 为准，评测实现必须复用以下部分。

## 1.1 Document tools

文件：

```text
toolcall-rl/tools/document_tools.py
toolcall-rl/tool_sandbox.py
```

当前已有工具：

```text
render_page
crop_region
zoom_region
parse_document
detect_layout
ocr_region
extract_table
chart_to_table
```

不要为 benchmark 重写 OCR / PDF renderer / table parser。

当前 `render_page`：

- PDF page number 是 **1-based**；
- 默认渲染 DPI 为 144；
- PDF 使用 PyMuPDF。

因此所有 benchmark adapter 最终均应把文档表示成项目能够直接读取的单个 PDF 路径。

---

## 1.2 Agent generation

文件：

```text
toolcall-rl/generate_with_retool.py
```

保持当前：

```text
generate_with_retool.generate
```

作为多轮 tool-call generation implementation。

不要使用 LongDocURL 官方 repo 的 Qwen2-VL inference runner；  
不要使用 DUDE baseline inference；  
不要写一套新的“benchmark 专用 Agent”。

Benchmark 应只替换：

```text
document_path
question
label
metadata.task_id
```

推理行为仍由项目自己的 Agent 完成。

---

## 1.3 Final answer extraction

复用：

```python
from document_reward import extract_final_answer
```

优先从 rollout metadata 的 `final_action` 中抽取最终答案，避免直接正则扫完整 trajectory。

---

## 1.4 当前工具预算

当前 `toolcall-rl/tool_sandbox.py` 的真实配置是：

```python
TOOL_CONFIGS = {
    "max_turns": 10,
    "max_tool_calls": 8,
    "max_obs_chars": 8192,
    "tool_concurrency": 32,
}
```

**四个 benchmark 主结果必须使用完全相同的 tool budget。**

不要针对 LongDocURL 单独把工具调用上限改为 20，也不要针对 MP-DocVQA 降到 4。否则主表不可公平比较。

如果后续专门做 budget ablation，可以单独跑，不覆盖主结果。

---

# 2. 最终新增目录结构

在仓库中新增：

```text
toolcall-rl/
└── eval_benchmarks/
    ├── README.md
    ├── common.py
    ├── prepare_docvqa2026.py
    ├── prepare_longdocurl.py
    ├── prepare_mpdocvqa.py
    ├── prepare_dude.py
    ├── export_eval_predictions.py
    ├── score_docvqa2026.py
    ├── score_longdocurl.py
    ├── score_mpdocvqa.py
    ├── score_dude.py
    ├── summarize_agent_metrics.py
    ├── validate_prepared_data.py
    └── vendors/
        ├── DocVQA2026/
        ├── LongDocURL/
        └── DUDEeval/
```

数据不要放进 Git repo。使用：

```text
/data/doc_agentic_benchmarks/
├── raw/
│   ├── docvqa2026/
│   ├── longdocurl/
│   ├── mpdocvqa/
│   └── dude/
├── documents/
│   ├── docvqa2026/
│   ├── longdocurl/
│   ├── mpdocvqa/
│   └── dude/
├── eval/
│   ├── docvqa2026.jsonl
│   ├── longdocurl.jsonl
│   ├── mpdocvqa.jsonl
│   └── dude.jsonl
├── gold/
│   ├── docvqa2026.jsonl
│   ├── longdocurl.jsonl
│   ├── mpdocvqa.jsonl
│   └── dude.jsonl
└── results/
    ├── docvqa2026/
    ├── longdocurl/
    ├── mpdocvqa/
    └── dude/
```

如果机器没有 `/data`，统一把 `BENCH_ROOT` 换成一个剩余空间最大的盘；**不要在代码内部散落多个绝对路径**。

---

# 3. 环境变量与依赖

在项目环境中执行：

```bash
cd /path/to/Doc-agentic-RL

export REPO_ROOT="$(pwd)"
export BENCH_ROOT="/data/doc_agentic_benchmarks"
export HF_HOME="${BENCH_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${BENCH_ROOT}/hf_datasets_cache"

mkdir -p \
  "${BENCH_ROOT}/raw" \
  "${BENCH_ROOT}/documents" \
  "${BENCH_ROOT}/eval" \
  "${BENCH_ROOT}/gold" \
  "${BENCH_ROOT}/results" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}"
```

安装评测 adapter 必要依赖：

```bash
python -m pip install -U \
  huggingface_hub \
  datasets \
  pyarrow \
  pillow \
  pymupdf \
  img2pdf \
  python-Levenshtein
```

不要安装四个 benchmark 官方 baseline 的完整训练依赖。

---

# 4. 官方 scorer 代码下载

只 clone scorer/code，小于数据量很多：

```bash
cd "${REPO_ROOT}/toolcall-rl/eval_benchmarks"
mkdir -p vendors
cd vendors

git clone --depth 1 https://github.com/VLR-CVC/DocVQA2026.git
git clone --depth 1 https://github.com/dengc2023/LongDocURL.git
git clone --depth 1 https://github.com/Jordy-VL/DUDEeval.git
```

用途：

```text
DocVQA2026/eval_utils.py
    → DocVQA 2026 官方 correctness evaluator

LongDocURL/utils/calculate_metrics.py
LongDocURL/utils/utils_score_v3.py
    → LongDocURL generalized accuracy

DUDEeval/evaluate_submission.py
    → DUDE 官方 ANLS
```

MP-DocVQA 不需要下载完整官方 framework；本规范直接实现其标准 ANLS，并额外统计 Agent 导航指标。

---

# 5. 磁盘预算

## 5.1 必须下载的数据量

| Benchmark | 下载内容 | 下载量 |
|---|---|---:|
| DocVQA 2026 | `val.parquet` | ~1.19 GB |
| LongDocURL | `pdf_files.tar.gz` | ~2.63 GB |
| LongDocURL | `LongDocURL_public.jsonl` | ~8.14 MB |
| MP-DocVQA | validation 29 parquet shards | ~4.69 GB |
| DUDE | train/val/test PDF binary archive（公开端只提供合包） | ~21.2 GB 临时 |
| DUDE | public GT JSON | ~14.1 MB |

关键点：

- LongDocURL **不下载** `png_files_p1.tar.gz`；
- LongDocURL **不下载** `png_files_p2.tar.gz`；
- DocVQA 2026 **不下载** `test.parquet`（约 4 GB）；
- MP-DocVQA **不下载** test shards；
- DUDE 的 21.2 GB 合包只作为临时下载文件，抽出 val PDF 后立即删除；
- 不缓存全数据集 OCR；
- 不把所有 PDF 预渲染成 PNG。

---

## 5.2 空间安全阈值

开始 DUDE 下载前执行：

```bash
df -h "${BENCH_ROOT}"
```

必须至少保留：

```text
>= 45 GB free
```

才开始 DUDE 下载/抽取。

最终建议：

```text
raw + materialized documents + scorer + results < 30~40 GB
```

实际生成的 tool render cache 需要定期清理。

---

# 6. 统一 eval schema

四个 adapter 最终都必须生成：

```text
${BENCH_ROOT}/eval/<benchmark>.jsonl
```

每行 schema 固定为：

```json
{
  "prompt": "Document path: /abs/path/doc.pdf\nQuestion: ...\n\n...",
  "label": {
    "answers": ["answer1", "answer2"],
    "metric": "anls"
  },
  "metadata": {
    "task_id": "benchmark:question_id",
    "benchmark": "benchmark_name",
    "question_id": "source_question_id",
    "doc_id": "source_doc_id",
    "document_path": "/abs/path/doc.pdf",
    "page_count": 12
  }
}
```

`task_id` 必须全局唯一。

格式固定为：

```text
docvqa2026:<question_id>
longdocurl:<question_id>
mpdocvqa:<questionId>
dude:<questionId>
```

### 6.1 不允许出现在 eval metadata 的 GT 字段

不要写：

```text
answer_page
target_page
answer_bbox
evidence_pages
detailed_evidences
answers_page_bounding_boxes
answer_type
```

这些信息写到：

```text
${BENCH_ROOT}/gold/<benchmark>.jsonl
```

---

# 7. `common.py` 必须实现的函数

创建：

```text
toolcall-rl/eval_benchmarks/common.py
```

至少实现以下接口，其他 adapter 统一调用，不要各写一遍。

```python
from pathlib import Path
from typing import Iterable
import json
import fitz


def jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pdf_page_count(path: Path) -> int:
    with fitz.open(path) as doc:
        return len(doc)


def make_task_id(benchmark: str, question_id: str) -> str:
    return f"{benchmark}:{question_id}"


def make_prompt(document_path: Path, question: str, extra_instruction: str = "") -> str:
    base = (
        f"Document path: {document_path}\n"
        f"Question: {question}\n\n"
        "Inspect the document with the available tools before answering. "
        "Use only evidence from the document. "
        "Return only the concise answer inside <final>...</final>."
    )
    if extra_instruction:
        base += "\n" + extra_instruction.strip()
    return base


def make_eval_row(
    *,
    benchmark: str,
    question_id: str,
    doc_id: str,
    document_path: Path,
    page_count: int,
    question: str,
    answers: list[str],
    extra_instruction: str = "",
) -> dict:
    return {
        "prompt": make_prompt(document_path, question, extra_instruction),
        "label": {"answers": answers, "metric": "anls"},
        "metadata": {
            "task_id": make_task_id(benchmark, question_id),
            "benchmark": benchmark,
            "question_id": str(question_id),
            "doc_id": str(doc_id),
            "document_path": str(document_path.resolve()),
            "page_count": int(page_count),
        },
    }
```

同时实现一个 PDF materializer：

```python
def images_to_pdf(images, output_pdf: Path) -> None:
    ...
```

要求：

1. 每次只处理一个 document；
2. 使用临时目录保存 page PNG；
3. PNG 使用无损保存；
4. 使用 `img2pdf.convert(page_paths)` 合成 PDF；
5. page order 严格等于输入 image list 顺序；
6. 临时 PNG 在单个 PDF 完成后立即删除；
7. 如果 `output_pdf` 已存在，先验证能被 PyMuPDF 打开且 page_count 正确，再跳过；
8. 不允许一次将整个 benchmark 的所有页面 PNG 落盘。

建议实现：

```python
import img2pdf
import tempfile
from PIL import Image


def images_to_pdf(images, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docbench_pages_") as td:
        page_paths = []
        for i, image in enumerate(images, start=1):
            if image is None:
                raise ValueError(f"missing page image {i}")
            if not isinstance(image, Image.Image):
                raise TypeError(f"page {i} is not PIL.Image: {type(image)}")
            p = Path(td) / f"{i:04d}.png"
            image.save(p, format="PNG")
            page_paths.append(str(p))
        output_pdf.write_bytes(img2pdf.convert(page_paths))
```

---

# 8. DocVQA 2026

## 8.1 下载

创建目录：

```bash
mkdir -p "${BENCH_ROOT}/raw/docvqa2026"
```

只下载 validation：

```bash
hf download \
  VLR-CVC/DocVQA-2026 \
  val.parquet \
  --repo-type dataset \
  --local-dir "${BENCH_ROOT}/raw/docvqa2026"
```

校验：

```bash
sha256sum "${BENCH_ROOT}/raw/docvqa2026/val.parquet"
```

期望：

```text
cfdcf75a3929881e5ff21f917c314005a6c1727430e8b8662f34dd9f4b82124f
```

不要执行：

```bash
hf download VLR-CVC/DocVQA-2026 ...
```

而不指定 `val.parquet`，因为这会把 test 一起下载。

---

## 8.2 原始字段

当前 validation 是 document-centric parquet。

一条 document row 包含：

```text
doc_id
doc_category
preview
document          # list[PIL.Image], 每个元素是一页
questions:
    question_id[]
    question[]
answers:
    question_id[]
    answer[]
```

当前 HF validation：

```text
25 document rows
8 domains
document page count range roughly 1..181
```

不要假设每个 document 只有一页。

---

## 8.3 `prepare_docvqa2026.py`

输入：

```text
${BENCH_ROOT}/raw/docvqa2026/val.parquet
```

输出：

```text
${BENCH_ROOT}/documents/docvqa2026/<doc_id>.pdf
${BENCH_ROOT}/eval/docvqa2026.jsonl
${BENCH_ROOT}/gold/docvqa2026.jsonl
```

读取方式：

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files=str(val_parquet),
    split="train",
)
```

逐 document 执行：

```python
doc_id = row["doc_id"]
category = row["doc_category"]
images = row["document"]

question_ids = row["questions"]["question_id"]
questions = row["questions"]["question"]
answer_qids = row["answers"]["question_id"]
answers = row["answers"]["answer"]
```

必须先做：

```python
assert question_ids == answer_qids
assert len(question_ids) == len(questions) == len(answers)
```

将 `images` 按原顺序合成：

```text
documents/docvqa2026/<doc_id>.pdf
```

然后对每个 question 建一行 eval。

### DocVQA 2026 的额外 prompt instruction

追加：

```text
Follow the DocVQA 2026 answer formatting rules for the content inside <final>:
- If the question is unanswerable, return exactly Unknown.
- For multiple answers, preserve document order and separate items with ", ".
- Use standardized abbreviated units with one space between number and unit.
- Attach % directly to the number.
- Format dates as YYYY-MM-DD.
- Do not use thousands separators.
- Do not add explanatory prose.
```

注意：

- `<final>` 内只放答案内容；
- 不在 `<final>` 内加 `FINAL ANSWER:`；
- scorer 再负责包装。

### Gold sidecar

每行：

```json
{
  "task_id": "docvqa2026:maps_2_q1",
  "question_id": "maps_2_q1",
  "doc_id": "maps_2",
  "doc_category": "maps",
  "answers": ["Macadam & Gravel"]
}
```

如果 source 只给单个 answer，也统一保存成 list。

---

## 8.4 DocVQA 2026 官方评分

创建：

```text
score_docvqa2026.py
```

从：

```text
results/docvqa2026/predictions.jsonl
gold/docvqa2026.jsonl
```

按 `task_id` join。

然后加载官方 evaluator：

```python
import sys
from pathlib import Path

vendor = Path(__file__).parent / "vendors" / "DocVQA2026"
sys.path.insert(0, str(vendor))

from eval_utils import evaluate_docvqa_prediction
```

对于项目预测：

```python
project_pred = prediction["final_answer"]
raw_for_official = f"FINAL ANSWER: {project_pred}"

is_correct, extracted = evaluate_docvqa_prediction(
    raw_for_official,
    gold["answers"],
)
```

最后：

```python
accuracy = correct_count / total_count
```

同时输出 per-domain：

```text
business report
comics
engineering drawing
infographics
maps
science paper
science poster
slide
```

最终写：

```text
results/docvqa2026/official_metrics.json
```

格式：

```json
{
  "benchmark": "docvqa2026",
  "split": "val",
  "num_questions": 80,
  "accuracy": 0.0,
  "by_domain": {},
  "missing_predictions": 0,
  "protocol_invalid": 0
}
```

`num_questions` 不要硬编码 80；上面只是输出示例。真实值必须由 gold sidecar 长度计算。

---

# 9. LongDocURL

## 9.1 下载

只下载 PDF 和一个 QA 文件：

```bash
mkdir -p "${BENCH_ROOT}/raw/longdocurl"

hf download \
  dengchao/LongDocURL \
  LongDocURL_public.jsonl \
  pdf_files.tar.gz \
  --repo-type dataset \
  --local-dir "${BENCH_ROOT}/raw/longdocurl"
```

校验 PDF archive：

```bash
sha256sum "${BENCH_ROOT}/raw/longdocurl/pdf_files.tar.gz"
```

期望：

```text
0cfc0984417f7a3d309c68a9afd9243768bf5767b1132860846ab4a5d940d8ab
```

禁止下载：

```text
png_files_p1.tar.gz
png_files_p2.tar.gz
```

因为项目已有 `render_page`，预渲染 33k+ 页没有必要。

---

## 9.2 解压 PDF

```bash
mkdir -p "${BENCH_ROOT}/documents/longdocurl"

tar -xzf \
  "${BENCH_ROOT}/raw/longdocurl/pdf_files.tar.gz" \
  -C "${BENCH_ROOT}/documents/longdocurl"
```

随后不要依赖一个猜测的固定子目录名。

`prepare_longdocurl.py` 启动时递归索引：

```python
pdf_index = {
    p.stem: p
    for p in documents_root.rglob("*.pdf")
}
```

对所有 `doc_no` 必须：

```python
assert doc_no in pdf_index
```

PDF 包解压成功后，可以删除 archive：

```bash
rm "${BENCH_ROOT}/raw/longdocurl/pdf_files.tar.gz"
```

前提是：

```bash
python toolcall-rl/eval_benchmarks/prepare_longdocurl.py ...
```

已经完成文件存在性验证。

---

## 9.3 原始 QA 字段

官方 public JSONL 每行包含至少：

```text
question_id
doc_no
total_pages
start_end_idx
question_type
question
answer
detailed_evidences
evidence_pages
evidence_sources
answer_format
task_tag
images
pdf_path
subTask
```

LongDocURL 页面索引的 benchmark annotation 使用 **1-based page indexing**；项目 `render_page` 也是 1-based，因此 post-hoc evidence page statistics 不做 `+1/-1` 转换。

---

## 9.4 `prepare_longdocurl.py`

输出：

```text
eval/longdocurl.jsonl
gold/longdocurl.jsonl
```

不要重新生成 PDF。

对每行：

```python
qid = str(row["question_id"])
doc_id = str(row["doc_no"])
pdf_path = pdf_index[doc_id]
question = row["question"]
answer = row["answer"]
answer_format = row["answer_format"]
```

`answer` 标准化成 list：

```python
if isinstance(answer, list):
    answers = [str(x) for x in answer]
else:
    answers = [str(answer)]
```

eval row 只保留：

```text
task_id
benchmark
question_id
doc_id
document_path
page_count
```

Gold sidecar 保存：

```json
{
  "task_id": "longdocurl:<qid>",
  "question_id": "<qid>",
  "doc_id": "<doc_no>",
  "answer": "...",
  "answer_format": "...",
  "task_tag": "...",
  "question_type": "...",
  "subTask": "...",
  "evidence_pages": [...],
  "evidence_sources": [...]
}
```

`detailed_evidences` 可以保存在 sidecar，但绝不进入 eval metadata。

---

## 9.5 LongDocURL 样本数验收

必须：

```python
assert len(eval_rows) == 2325
```

并且：

```python
assert len(set(task_id)) == 2325
```

所有 referenced PDF 必须存在。

---

## 9.6 LongDocURL 官方 generalized accuracy

官方 repo 的：

```text
utils/calculate_metrics.py
```

要求 result JSONL 每行至少包含：

```text
pred
answer
answer_format
```

由于项目最终 `<final>` 已要求 concise answer，因此**不要再调用官方 baseline 中额外的 GPT-4o/qwen-turbo “short-answer extraction”**。  
否则会：

- 引入额外模型；
- 违反 ≤8B agent system 的参数/服务口径；
- 增加不可复现因素；
- 改变最终答案。

创建：

```text
score_longdocurl.py
```

将 prediction + gold join 后生成：

```text
results/longdocurl/official_input.jsonl
```

每行：

```json
{
  "question_id": "...",
  "pred": "<final answer>",
  "answer": "<gold answer>",
  "answer_format": "<official answer_format>",
  "...": "可以保留其他官方字段"
}
```

如果 rollout 失败或没有合法 final：

```text
pred = "Fail to extract"
```

不要从分母里删掉失败样本。

随后执行：

```bash
cd "${REPO_ROOT}/toolcall-rl/eval_benchmarks/vendors/LongDocURL"

python utils/calculate_metrics.py \
  --results_file "${BENCH_ROOT}/results/longdocurl/official_input.jsonl"
```

官方代码会输出：

```text
Avg. acc
Rectified Avg. acc
```

主表报告：

```text
Rectified Avg. acc
```

原因：官方脚本会按完整 2,325 样本纠正缺失样本。

实现 `score_longdocurl.py` 时应直接 import `eval_score`，同时写一个结构化 JSON，而不是只依赖 stdout：

```python
from utils.utils_score_v3 import eval_score
```

对所有 2325 条：

```python
score = 0.0 if pred == "Fail to extract" else eval_score(
    gold_answer,
    pred,
    answer_format,
)
```

输出：

```json
{
  "benchmark": "longdocurl",
  "num_questions": 2325,
  "generalized_accuracy": 0.0,
  "understanding": 0.0,
  "reasoning": 0.0,
  "locating": 0.0,
  "missing_predictions": 0
}
```

其中 Understanding / Reasoning / Locating 按 `task_tag` 分组。

---

# 10. MP-DocVQA

## 10.1 数据源选择

官方数据下载入口需要通过 RRC；为了让 Agent 可以无交互、可复现地自动下载 validation，使用当前 Hugging Face 转换版：

```text
lmms-lab/MP-DocVQA
```

只取 validation。

当前 validation：

```text
5,187 QA
927 documents
29 parquet shards
约 4.69 GB
```

这与官方 MP-DocVQA validation 规模对应。

---

## 10.2 下载

```bash
mkdir -p "${BENCH_ROOT}/raw/mpdocvqa"

hf download \
  lmms-lab/MP-DocVQA \
  --repo-type dataset \
  --include "data/val-*.parquet" \
  --local-dir "${BENCH_ROOT}/raw/mpdocvqa"
```

下载后确认：

```bash
find "${BENCH_ROOT}/raw/mpdocvqa" \
  -name 'val-*.parquet' \
  -type f | wc -l
```

必须为：

```text
29
```

不要下载：

```text
data/test-*.parquet
```

---

## 10.3 原始字段

validation parquet 包含：

```text
questionId
question
doc_id
page_ids
answers
answer_page_idx
data_split
image_1
...
image_20
```

其中：

- `page_ids` 可能是字符串化 list；
- `answers` 可能是字符串化 list；
- `answer_page_idx` 是 gold page 在该多页 document 中的索引；
- `answer_page_idx` 按原 benchmark 表示为 **0-based**；
- 项目工具 page number 是 **1-based**；
- 最多 20 页。

---

## 10.4 为什么必须将 page images 合成 PDF

项目 document tools 的接口以：

```text
document_path
page_number
```

工作。

如果把 MP-DocVQA 每个 page image 当成独立 `document_path`，Agent 就无法通过一个文档路径自主导航 1..N 页。

因此 validation adapter 必须：

```text
page images
    ↓
one multi-page PDF per doc_id
    ↓
existing render_page / crop / OCR tools
```

不要修改主 tool API 来适配 dataset。

---

## 10.5 `prepare_mpdocvqa.py`

使用 streaming 方式读取本地 parquet，避免额外生成一个完整 Arrow cache：

```python
from datasets import load_dataset

shards = sorted(raw_root.rglob("val-*.parquet"))

ds = load_dataset(
    "parquet",
    data_files={"validation": [str(x) for x in shards]},
    split="validation",
    streaming=True,
)
```

实现：

```python
import ast

def parse_maybe_literal(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value
```

逐 row：

```python
qid = str(row["questionId"])
doc_id = str(row["doc_id"])
page_ids = parse_maybe_literal(row["page_ids"])
answers = parse_maybe_literal(row["answers"])
gold_idx = int(row["answer_page_idx"])
```

验证：

```python
assert isinstance(page_ids, list)
assert isinstance(answers, list)
assert 0 <= gold_idx < len(page_ids)
assert 1 <= len(page_ids) <= 20
```

### PDF dedup

同一 `doc_id` 有多个 QA，PDF 只生成一次。

维护：

```python
materialized_docs: set[str]
```

当 `doc_id` 第一次出现：

```python
images = [
    row[f"image_{i}"]
    for i in range(1, len(page_ids) + 1)
]
```

验证：

```python
assert all(image is not None for image in images)
```

然后：

```text
documents/mpdocvqa/<doc_id>.pdf
```

严格按 `image_1 ... image_N` 顺序合成。

同一个 `doc_id` 后续 row：

- 不重复写 PDF；
- 但必须验证 `page_ids` 与第一次见到的 page_ids 完全一致。

### eval row

只写：

```text
question
answers
doc_id
questionId
document_path
page_count
```

### gold sidecar

写：

```json
{
  "task_id": "mpdocvqa:<questionId>",
  "question_id": "<questionId>",
  "doc_id": "<doc_id>",
  "answers": ["..."],
  "page_ids": ["..."],
  "gold_answer_page_0based": 3,
  "gold_answer_page_1based": 4
}
```

注意：`gold_answer_page_1based` **只存在 sidecar**。

---

## 10.6 MP-DocVQA 验收

必须满足：

```text
eval rows       = 5,187
unique doc_id   = 927
unique task_id  = 5,187
```

每个 materialized PDF：

```python
pdf_page_count(pdf) == len(page_ids)
```

完成以后可删除原 validation parquet，以省约 4.69 GB：

```bash
rm -rf "${BENCH_ROOT}/raw/mpdocvqa/data"
```

删除前必须先运行：

```bash
python toolcall-rl/eval_benchmarks/validate_prepared_data.py \
  --benchmark mpdocvqa \
  --bench-root "${BENCH_ROOT}"
```

并通过。

---

## 10.7 MP-DocVQA 官方主指标：ANLS

主表报告：

```text
ANLS
```

实现 `score_mpdocvqa.py`。

按标准 DocVQA/DUDE ANLS 做：

```python
def normalized_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def nls(pred: str, gt: str) -> float:
    p = normalized_text(pred)
    g = normalized_text(gt)
    if p == "" and g == "":
        return 1.0
    if p == "" or g == "":
        return 0.0
    dist = Levenshtein.distance(p, g)
    score = 1.0 - dist / max(len(p), len(g))
    return score if score >= 0.5 else 0.0


def anls(pred: str, refs: list[str]) -> float:
    return max(nls(pred, ref) for ref in refs)
```

总分：

```python
mean(per_question_anls)
```

不要用项目 `document_reward.normalize_answer()` 替代这套官方兼容 scorer。

---

## 10.8 不强行声称 APPA

MP-DocVQA 官方还有 APPA，但项目当前 final answer protocol 并不要求模型显式输出一个 `predicted_page_idx`。

因此主实验中：

- **不要把“最后访问页”直接叫 APPA**；
- 不改 `<final>` schema；
- 不向模型泄漏 gold page。

额外报告 Agent-native page metrics：

```text
Gold Page Visited Rate
Gold Page First-Hit Rate
Mean First-Hit Tool Step
Pages Visited / Question
```

如果将来单独实现显式 page prediction head，再报告官方 APPA。

---

# 11. DUDE

## 11.1 为什么只评 validation

DUDE 当前公开 GT JSON：

```text
train  = 23,728
val    = 6,315
test   = 11,402
```

公开描述明确说明 ground truth 用于 train/validation；test 用于 leaderboard。

因此本地论文可复现评测：

```text
DUDE validation = 6,315 QA
```

不跑 train，不跑 test。

---

## 11.2 下载 annotation

```bash
mkdir -p "${BENCH_ROOT}/raw/dude"
```

下载：

```bash
curl -L \
  "https://zenodo.org/records/7763635/files/2023-03-23_DUDE_gt_test_PUBLIC.json?download=1" \
  -o "${BENCH_ROOT}/raw/dude/2023-03-23_DUDE_gt_test_PUBLIC.json"
```

这个文件虽然名字包含 `test_PUBLIC`，但里面有 `data_split`，按：

```text
data_split == "val"
```

提取 validation。

---

## 11.3 下载 PDF binary archive

当前 HF 公开 loader 把 train/val/test binary 放在一个 21.2 GB archive 中。

只下载这个 archive，不下载 OCR config：

```bash
hf download \
  jordyvl/DUDE_loader \
  data/DUDE_train-val-test_binaries.tar.gz \
  --repo-type dataset \
  --local-dir "${BENCH_ROOT}/raw/dude"
```

校验：

```bash
sha256sum \
  "${BENCH_ROOT}/raw/dude/data/DUDE_train-val-test_binaries.tar.gz"
```

期望：

```text
1506384a93022a2da6b180270345a6928ea7347842eaa2d2e177190c4fd29cae
```

不要通过：

```python
load_dataset("jordyvl/DUDE_loader", ...)
```

直接加载完整数据，因为这样会自动解压并缓存 train/val/test + OCR，浪费空间。

---

## 11.4 只从 tar.gz 中抽 validation PDF

`prepare_dude.py` 必须先读取 annotation：

```python
root = json.load(...)
all_rows = root["data"]
val_rows = [x for x in all_rows if x["data_split"] == "val"]
val_doc_ids = {str(x["docId"]) for x in val_rows}
```

验收：

```python
assert len(val_rows) == 6315
```

随后**流式遍历 tar.gz，只 copy validation PDF**。

不要：

```python
tar.extractall(...)
```

实现逻辑：

```python
from pathlib import PurePosixPath
import tarfile
import shutil

with tarfile.open(archive, "r:gz") as tar:
    for member in tar:
        if not member.isfile():
            continue
        if not member.name.lower().endswith(".pdf"):
            continue

        path = PurePosixPath(member.name)
        tokens = set(path.parts)
        tokens.add(path.stem)

        matches = tokens.intersection(val_doc_ids)
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            raise RuntimeError(
                f"ambiguous DUDE pdf member: {member.name}, matches={matches}"
            )

        doc_id = next(iter(matches))
        dst = documents_root / f"{doc_id}.pdf"

        src = tar.extractfile(member)
        if src is None:
            raise RuntimeError(f"cannot read {member.name}")

        with dst.open("wb") as f:
            shutil.copyfileobj(src, f)
```

如果 archive 内路径格式发生变化，不要靠 substring：

```python
if doc_id in member.name
```

因为可能误匹配。

只使用 path component 或 file stem 精确匹配。

遍历完成：

```python
missing = val_doc_ids - extracted_doc_ids
assert not missing, sorted(missing)[:20]
```

并检查：

```python
assert len(extracted_doc_ids) == len(val_doc_ids)
```

然后立刻删除 21.2 GB archive：

```bash
rm \
 "${BENCH_ROOT}/raw/dude/data/DUDE_train-val-test_binaries.tar.gz"
```

---

## 11.5 DUDE eval / gold

DUDE row 主要字段：

```text
docId
questionId
question
answers
answers_page_bounding_boxes
answers_variants
answer_type
data_split
```

eval row：

```python
answers = row["answers"]
```

如果：

```python
answers == []
```

项目内部 label 使用：

```python
[""]
```

避免空 reference list 导致项目 rollout diagnostic scorer 异常。

但 official gold sidecar 保留 source：

```text
answers == []
```

---

## 11.6 DUDE 的不可回答问题适配

项目 `<final>` parser 不适合真正的空字符串：

```text
<final></final>
```

所以 DUDE prompt 追加：

```text
If the document does not contain an answer, return exactly Unknown inside <final>.
```

离线生成 DUDE official submission 时：

```python
EMPTY_SENTINELS = {
    "",
    "unknown",
    "unanswerable",
    "not answerable",
    "not-answerable",
    "n/a",
}
```

只有当模型 prediction **本身精确等于**其中之一时：

```python
submission_answer = ""
```

不要根据 gold `answer_type` 决定是否置空，因为那会是评测泄漏。

---

## 11.7 生成官方 DUDE validation GT

从原 GT JSON 构造：

```text
results/dude/dude_val_gt.json
```

保留原 root 的其他字段，只替换：

```python
root["data"] = val_rows
```

不要把字段改名。

---

## 11.8 生成 DUDE submission

官方格式是 JSON root array：

```json
[
  {
    "questionId": "...",
    "answers": ["Yes"],
    "answers_confidence": [1]
  }
]
```

本项目没有 calibrated confidence，因此固定：

```python
"answers_confidence": [1]
```

这不会影响默认 ANLS，因为不加 `-c`。

对于普通单答案：

```python
"answers": [pred]
```

对于映射后的 unanswerable：

```python
"answers": [""]
```

不要提交：

```text
answers = "text"
```

必须是 list。

---

## 11.9 DUDE 官方评分

```bash
cd "${REPO_ROOT}/toolcall-rl/eval_benchmarks/vendors/DUDEeval"

python evaluate_submission.py \
  -g="${BENCH_ROOT}/results/dude/dude_val_gt.json" \
  -s="${BENCH_ROOT}/results/dude/dude_val_submission.json" \
  -o="${BENCH_ROOT}/results/dude/official"
```

默认 ANLS threshold：

```text
0.5
```

主表报告：

```text
Overall ANLS
```

不要加：

```text
-c
```

除非后续真的有 calibrated answer confidence。

---

# 12. `export_eval_predictions.py`

不要用现有 `export_rollout_workflows.py --limit 1000000` 来导出全 benchmark，因为该脚本还会生成一个很大的 Markdown timeline。

新增轻量脚本：

```text
toolcall-rl/eval_benchmarks/export_eval_predictions.py
```

输入：

```text
<output_dir>/dump_details/rollout_data/eval_0.pt
```

输出：

```text
${BENCH_ROOT}/results/<benchmark>/predictions.jsonl
```

实现时复用：

```python
from export_rollout_workflows import _find_samples
from document_reward import extract_final_answer
```

核心逻辑：

```python
payload = torch.load(
    eval_pt,
    map_location="cpu",
    weights_only=False,
)

samples = _find_samples(payload)

for sample in samples:
    metadata = sample.get("metadata") or {}
    task_id = metadata.get("task_id")
    if not task_id:
        raise RuntimeError("eval sample missing metadata.task_id")

    response = str(sample.get("response", ""))
    final_answer, protocol_valid = extract_final_answer(
        response,
        metadata,
    )

    tool_execution = metadata.get("tool_execution") or {}
    nav = metadata.get("navigation_state") or {}

    row = {
        "task_id": task_id,
        "benchmark": metadata.get("benchmark"),
        "question_id": metadata.get("question_id"),
        "doc_id": metadata.get("doc_id"),
        "final_answer": final_answer,
        "protocol_valid": bool(protocol_valid),
        "rollout_status": (
            sample.get("rollout_status")
            or metadata.get("rollout_status")
            or ""
        ),
        "tool_call_count": int(
            metadata.get(
                "tool_call_count",
                tool_execution.get("call_count", 0),
            ) or 0
        ),
        "valid_tool_call_count": int(
            metadata.get(
                "valid_tool_call_count",
                tool_execution.get("valid_call_count", 0),
            ) or 0
        ),
        "tool_error_count": int(
            metadata.get(
                "tool_error_count",
                tool_execution.get("error_count", 0),
            ) or 0
        ),
        "visited_pages": metadata.get(
            "visited_pages",
            nav.get("visited_pages", []),
        ),
        "rendered_pages": metadata.get(
            "rendered_pages",
            nav.get("rendered_pages", []),
        ),
        "ocr_pages": metadata.get(
            "ocr_pages",
            nav.get("ocr_pages", []),
        ),
        "supporting_pages": metadata.get(
            "supporting_pages",
            nav.get("supporting_pages", []),
        ),
        "tool_calls": tool_execution.get("calls", []),
    }
```

### 必须处理失败 rollout

如果：

```text
final_answer == ""
```

不要丢掉该样本。

仍输出一行：

```json
{
  "final_answer": "",
  "protocol_valid": false,
  ...
}
```

官方 scorer 按错误处理。

---

## 12.1 一题只能有一个 benchmark prediction

主 benchmark 评测必须：

```text
n_samples_per_eval_prompt = 1
```

不要沿用项目某些 RL eval 中的：

```text
16 samples / prompt
```

否则：

- 计算量乘 16；
- 需要额外定义 pass@k / majority vote；
- 与标准 benchmark 单次预测设置不一致。

如果现有 eval launch script 中存在：

```text
--n-samples-per-eval-prompt 16
```

在 benchmark eval 专用 launcher 中改为：

```text
--n-samples-per-eval-prompt 1
```

训练脚本保持原值，不要全局修改。

---

# 13. 与现有 Qwen3-VL 4B/8B evaluation launcher 的连接方式

## 13.1 不重新写模型服务

你的本地项目已经有可工作的 Qwen3-VL 4B/8B + tool-agent rollout 路径。  
Benchmark 接入只替换 eval data，不另写 Transformers/vLLM/SGLang 单轮推理。

代码 Agent 必须先定位**当前项目真实用于已有自建测试集评测的 launcher**，复制成：

```text
toolcall-rl/eval_benchmarks/run_eval_only.sh
```

复制以后只做以下 benchmark 相关修改：

1. `EVAL_DATA` 改为参数；
2. `OUTPUT_DIR` 改为参数；
3. `--n-samples-per-eval-prompt 1`；
4. 使用 eval-only/no-update 模式；
5. 保持当前 Qwen3-VL 模型加载配置；
6. 保持当前 4×A100 GPU topology；
7. 保持现有 tool generation hook；
8. 保持 `generate_with_retool.generate`；
9. 保持当前 tool budget；
10. 不启动 RL update。

当前项目 `baseline_eval.py` 已将 eval-only 的必要训练控制定义为：

```text
--num-rollout 0
--lr-decay-iters 1
```

benchmark launcher 必须保持等价约束，确保：

```text
optimizer steps = 0
parameter update = 0
```

### 重要

当前公开 repo 的某些旧脚本默认按 8 GPU 分 actor/rollout。  
**不要把那个 8-GPU partition 生搬到本机。**

Benchmark launcher 应复制你本地已经验证成功的 4×A100 Qwen3-VL eval launch 配置，只替换上面列出的 benchmark 参数。

这不是让 Agent 自行重新设计 GPU 配置；这是明确要求**复用当前已成功运行自建 200-sample eval 的同一个 GPU/model launcher**。

---

## 13.2 `run_eval_only.sh` 的外部接口必须固定

无论内部现有 launcher 如何，新增 wrapper 最终必须支持：

```bash
bash toolcall-rl/eval_benchmarks/run_eval_only.sh \
  --eval-data "/abs/path/eval.jsonl" \
  --output-dir "/abs/path/output" \
  --model "/abs/path/model_or_checkpoint"
```

wrapper 要：

- 检查 eval data 存在；
- 创建 output dir；
- 检查 model path；
- 启动当前项目 eval-only；
- 运行结束后必须存在：

```text
<output-dir>/dump_details/rollout_data/eval_0.pt
```

不存在则退出非零。

---

# 14. 全量运行前的 smoke test

不要直接跑 14k 条问题。

每个 benchmark 从 prepared eval JSONL 取前 8 条：

```bash
head -n 8 \
  "${BENCH_ROOT}/eval/docvqa2026.jsonl" \
  > "${BENCH_ROOT}/eval/docvqa2026_smoke8.jsonl"
```

其他三个同样。

逐 benchmark 跑 8 条。

Smoke 验收：

```text
1. 8 条均能加载 document_path
2. render_page 能打开 PDF
3. page_number 1-based 正常
4. 至少一条发生真实 tool call
5. 无系统性 tool import error
6. 结果中 task_id 不丢失
7. export_eval_predictions.py 能导出 8 条
8. scorer 能得到 8 条分数
9. 不出现 gold evidence/page 泄漏到 prompt
10. output 目录不会覆盖其他 benchmark
```

只有四个 smoke 全过后才跑全量。

---

# 15. `validate_prepared_data.py`

该脚本要在正式 rollout 前统一检查。

参数：

```bash
python toolcall-rl/eval_benchmarks/validate_prepared_data.py \
  --benchmark longdocurl \
  --bench-root "${BENCH_ROOT}"
```

必须检查：

### 所有 benchmark

```text
eval JSONL 可解析
gold JSONL 可解析
task_id unique
eval task_id == gold task_id 集合
document_path 是 absolute path
document_path exists
PDF 可被 PyMuPDF 打开
page_count >= 1
label.answers 是 list
prompt 不包含 gold answer
prompt 不包含 evidence_pages
prompt 不包含 answer_page
```

“prompt 不包含 gold answer”不要简单做 substring 全拒绝，因为 question 本身可能自然包含答案词；应检查是否由 adapter 直接拼接了 label/GT 字段，而不是做语义判断。

### LongDocURL

```text
rows == 2325
```

### MP-DocVQA

```text
rows == 5187
unique docs == 927
```

### DUDE

```text
rows == 6315
```

### DocVQA 2026

```text
source documents == 25
```

并检查 question IDs 唯一。

---

# 16. Agent-level 统一指标

创建：

```text
summarize_agent_metrics.py
```

对四个 benchmark 都报告：

```text
Protocol Valid Rate
Completion Rate
Mean Tool Calls
Mean Valid Tool Calls
Tool Error Rate
Mean Unique Pages Visited
Mean Rendered Pages
Mean OCR Pages
Duplicate Page Call Rate（如果 metadata 已提供）
No-Information-Gain Call Rate（如果 metadata 已提供）
```

这些指标只从 rollout metadata 计算。

不要用 GT 得出“tool useful”然后反向影响 prediction。

---

## 16.1 Evidence/navigation 指标

### MP-DocVQA

gold 有单一 answer page。

计算：

```python
gold = gold_answer_page_1based
visited = set(pred_record["visited_pages"])

gold_page_visited = gold in visited
```

报告：

```text
Gold Page Visited Rate
```

再从 `tool_calls` 中按 execution order 找第一次出现该 page 的 call：

```text
Mean Gold Page First-Hit Tool Step
```

如果从未访问：

```text
first_hit = null
```

不从平均值中静默删掉。可以同时报告：

```text
Hit-conditioned Mean First-Hit Step
```

和：

```text
Unconditional First-Hit normalized score
```

---

### LongDocURL

gold `evidence_pages` 可能多页。

令：

```python
G = set(gold_evidence_pages)
V = set(visited_pages)
```

计算：

```python
evidence_recall = len(G & V) / len(G)
evidence_precision = len(G & V) / len(V) if V else 0
```

报告：

```text
Evidence Page Recall
Evidence Page Precision
Any Evidence Hit Rate
All Evidence Hit Rate
```

这些是 Agent 分析指标，不替代 generalized accuracy。

---

### DocVQA 2026 / DUDE

如果没有稳定公开 page-level GT，就不要编造 evidence recall。

只报告：

```text
tool efficiency
pages visited
final benchmark score
```

---

# 17. 评分输出格式

每个 benchmark result dir 最终固定：

```text
results/<benchmark>/
├── predictions.jsonl
├── official_metrics.json
├── agent_metrics.json
└── per_sample_scores.jsonl
```

DUDE 额外：

```text
dude_val_gt.json
dude_val_submission.json
official/results.json
```

LongDocURL 额外：

```text
official_input.jsonl
```

---

# 18. 最终统一 summary

生成：

```text
${BENCH_ROOT}/results/summary.json
${BENCH_ROOT}/results/summary.md
```

`summary.md` 主表固定：

```markdown
| Benchmark | Split | #Q | Official metric | Score | Protocol valid | Mean tool calls | Mean pages visited |
|---|---|---:|---|---:|---:|---:|---:|
| DocVQA 2026 | val | ... | Accuracy | ... | ... | ... | ... |
| LongDocURL | public | 2325 | Generalized Accuracy | ... | ... | ... | ... |
| MP-DocVQA | val | 5187 | ANLS | ... | ... | ... | ... |
| DUDE | val | 6315 | ANLS | ... | ... | ... | ... |
```

另起 Agent navigation 表：

```markdown
| Benchmark | Evidence metric | Value |
|---|---|---:|
| LongDocURL | Evidence Page Recall | ... |
| LongDocURL | Any Evidence Hit Rate | ... |
| MP-DocVQA | Gold Page Visited Rate | ... |
| MP-DocVQA | Mean First-Hit Tool Step | ... |
```

不要把自建集和公开 benchmark 混成一个平均分。

---

# 19. 推荐的完整执行顺序

## Phase A：安装与 scorer

```bash
cd "${REPO_ROOT}"

python -m pip install -U \
  huggingface_hub \
  datasets \
  pyarrow \
  pillow \
  pymupdf \
  img2pdf \
  python-Levenshtein

cd toolcall-rl/eval_benchmarks/vendors

git clone --depth 1 https://github.com/VLR-CVC/DocVQA2026.git
git clone --depth 1 https://github.com/dengc2023/LongDocURL.git
git clone --depth 1 https://github.com/Jordy-VL/DUDEeval.git
```

---

## Phase B：下载 DocVQA 2026

```bash
hf download \
  VLR-CVC/DocVQA-2026 \
  val.parquet \
  --repo-type dataset \
  --local-dir "${BENCH_ROOT}/raw/docvqa2026"
```

---

## Phase C：下载 LongDocURL

```bash
hf download \
  dengchao/LongDocURL \
  LongDocURL_public.jsonl \
  pdf_files.tar.gz \
  --repo-type dataset \
  --local-dir "${BENCH_ROOT}/raw/longdocurl"

mkdir -p "${BENCH_ROOT}/documents/longdocurl"

tar -xzf \
  "${BENCH_ROOT}/raw/longdocurl/pdf_files.tar.gz" \
  -C "${BENCH_ROOT}/documents/longdocurl"
```

---

## Phase D：下载 MP-DocVQA validation

```bash
hf download \
  lmms-lab/MP-DocVQA \
  --repo-type dataset \
  --include "data/val-*.parquet" \
  --local-dir "${BENCH_ROOT}/raw/mpdocvqa"
```

---

## Phase E：下载 DUDE annotation + binary archive

```bash
curl -L \
  "https://zenodo.org/records/7763635/files/2023-03-23_DUDE_gt_test_PUBLIC.json?download=1" \
  -o "${BENCH_ROOT}/raw/dude/2023-03-23_DUDE_gt_test_PUBLIC.json"

hf download \
  jordyvl/DUDE_loader \
  data/DUDE_train-val-test_binaries.tar.gz \
  --repo-type dataset \
  --local-dir "${BENCH_ROOT}/raw/dude"
```

---

## Phase F：prepare

```bash
cd "${REPO_ROOT}"

python toolcall-rl/eval_benchmarks/prepare_docvqa2026.py \
  --bench-root "${BENCH_ROOT}"

python toolcall-rl/eval_benchmarks/prepare_longdocurl.py \
  --bench-root "${BENCH_ROOT}"

python toolcall-rl/eval_benchmarks/prepare_mpdocvqa.py \
  --bench-root "${BENCH_ROOT}"

python toolcall-rl/eval_benchmarks/prepare_dude.py \
  --bench-root "${BENCH_ROOT}" \
  --delete-archive
```

---

## Phase G：validate

```bash
for b in docvqa2026 longdocurl mpdocvqa dude; do
  python toolcall-rl/eval_benchmarks/validate_prepared_data.py \
    --benchmark "$b" \
    --bench-root "${BENCH_ROOT}"
done
```

---

## Phase H：smoke

对每个 benchmark：

```bash
head -n 8 \
  "${BENCH_ROOT}/eval/${B}.jsonl" \
  > "${BENCH_ROOT}/eval/${B}_smoke8.jsonl"
```

用现有 Qwen3-VL eval-only launcher 跑 8 条。

---

## Phase I：full eval

顺序建议：

```text
1. DocVQA 2026
2. MP-DocVQA
3. DUDE
4. LongDocURL
```

原因：

- 先用较短 benchmark 验证整个 scorer；
- LongDocURL 文档最长、总 tool latency 通常最大，放最后。

每个 benchmark 使用独立 output dir。

示例：

```bash
bash toolcall-rl/eval_benchmarks/run_eval_only.sh \
  --eval-data "${BENCH_ROOT}/eval/mpdocvqa.jsonl" \
  --output-dir "${BENCH_ROOT}/results/mpdocvqa/run" \
  --model "${MODEL_PATH}"
```

---

## Phase J：export prediction

```bash
python toolcall-rl/eval_benchmarks/export_eval_predictions.py \
  --eval-pt \
  "${BENCH_ROOT}/results/mpdocvqa/run/dump_details/rollout_data/eval_0.pt" \
  --output \
  "${BENCH_ROOT}/results/mpdocvqa/predictions.jsonl"
```

其他 benchmark 同理。

---

## Phase K：official scoring

```bash
python toolcall-rl/eval_benchmarks/score_docvqa2026.py \
  --bench-root "${BENCH_ROOT}"

python toolcall-rl/eval_benchmarks/score_longdocurl.py \
  --bench-root "${BENCH_ROOT}"

python toolcall-rl/eval_benchmarks/score_mpdocvqa.py \
  --bench-root "${BENCH_ROOT}"

python toolcall-rl/eval_benchmarks/score_dude.py \
  --bench-root "${BENCH_ROOT}"
```

---

## Phase L：agent metrics

```bash
for b in docvqa2026 longdocurl mpdocvqa dude; do
  python toolcall-rl/eval_benchmarks/summarize_agent_metrics.py \
    --benchmark "$b" \
    --bench-root "${BENCH_ROOT}"
done
```

最后生成 summary。

---

# 20. 失败恢复与断点续跑

所有 prepare 脚本必须可重复运行。

### PDF 已存在时

不要直接跳过。

先：

```python
with fitz.open(pdf) as doc:
    assert len(doc) == expected_pages
```

通过才跳过。

### eval JSONL 已存在时

prepare 应重新生成到临时文件：

```text
xxx.jsonl.tmp
```

全部成功后 atomic rename：

```python
tmp.replace(final)
```

避免中途失败留下半份 manifest。

### rollout 已存在时

如果：

```text
eval_0.pt
```

已存在，不自动覆盖。

要求用户/Agent显式设置：

```text
--overwrite
```

否则退出。

---

# 21. 不允许的实现

代码 Agent 不得做以下事情：

1. 把 benchmark 的所有 page 预渲染后永久保存为 PNG；
2. 下载 LongDocURL PNG package；
3. 下载 DocVQA 2026 test；
4. 下载 MP-DocVQA test；
5. 用 DUDE train 充当 eval；
6. 把 `answer_page` / `evidence_pages` 放进模型 prompt；
7. 把 gold evidence 写进导航状态；
8. 根据 gold answer type 后处理模型答案；
9. 使用额外大模型做 LongDocURL answer extraction；
10. 用 GPT/API grader 替代官方 metric；
11. 用项目 reward `quality` 充当官方 benchmark score；
12. 把多个 rollout 取 best-of-16 后称作标准单次 benchmark；
13. 为每个 benchmark 改 tool budget；
14. 因长文档而一次把所有页面图片输入 Qwen3-VL；
15. 修改训练数据或训练 reward 以“适配评测集”；
16. 将 public benchmark validation 用于 RL/SFT 后再报告同一 split 的结果。

---

# 22. 结果完整性检查

正式论文结果生成前，必须输出：

```text
evaluation_integrity.json
```

至少记录：

```json
{
  "model": "...",
  "model_parameter_class": "<=8B",
  "checkpoint": "...",
  "git_commit": "...",
  "tool_budget": {
    "max_turns": 10,
    "max_tool_calls": 8,
    "max_obs_chars": 8192
  },
  "samples_per_question": 1,
  "benchmarks": {
    "docvqa2026": {
      "split": "val",
      "num_questions": 0
    },
    "longdocurl": {
      "split": "public",
      "num_questions": 2325
    },
    "mpdocvqa": {
      "split": "val",
      "num_questions": 5187
    },
    "dude": {
      "split": "val",
      "num_questions": 6315
    }
  }
}
```

实际 DocVQA 2026 question 数由 adapter 从 validation 计算，不硬编码。

---

# 23. 数据/评分来源（供 Agent 核实，不要改成其他下载源）

## DocVQA 2026

Dataset:

```text
https://huggingface.co/datasets/VLR-CVC/DocVQA-2026
```

Validation file:

```text
https://huggingface.co/datasets/VLR-CVC/DocVQA-2026/blob/main/val.parquet
```

Official evaluator:

```text
https://github.com/VLR-CVC/DocVQA2026
```

当前 `val.parquet`：

```text
~1.19 GB
SHA256 cfdcf75a3929881e5ff21f917c314005a6c1727430e8b8662f34dd9f4b82124f
```

---

## LongDocURL

Dataset:

```text
https://huggingface.co/datasets/dengchao/LongDocURL
```

Official code:

```text
https://github.com/dengc2023/LongDocURL
```

Paper:

```text
ACL 2025 Main
https://aclanthology.org/2025.acl-long.57/
```

PDF archive：

```text
~2.63 GB
SHA256 0cfc0984417f7a3d309c68a9afd9243768bf5767b1132860846ab4a5d940d8ab
```

---

## MP-DocVQA

Official framework:

```text
https://github.com/rubenpt91/MP-DocVQA-Framework
```

Automatable validation mirror:

```text
https://huggingface.co/datasets/lmms-lab/MP-DocVQA
```

Validation：

```text
5,187 questions
927 documents
29 parquet shards
~4.69 GB
```

---

## DUDE

Official project:

```text
https://github.com/duchallenge-team/dude
```

Validation/test annotations:

```text
https://zenodo.org/records/7763635
```

PDF binaries:

```text
https://huggingface.co/datasets/jordyvl/DUDE_loader
```

Official evaluator:

```text
https://github.com/Jordy-VL/DUDEeval
```

Current binary archive：

```text
~21.2 GB
SHA256 1506384a93022a2da6b180270345a6928ea7347842eaa2d2e177190c4fd29cae
```

Validation：

```text
6,315 questions
```

---

# 24. 最终 Definition of Done

只有同时满足以下条件才算四 benchmark 接入完成：

- [ ] DocVQA 2026 只下载/使用 validation。
- [ ] LongDocURL 只下载 PDF + QA JSONL，不下载 PNG。
- [ ] MP-DocVQA 只下载 validation 29 shards。
- [ ] DUDE 只评 validation 6,315 QA，train/test PDF 未被持久化。
- [ ] 四个 benchmark 都转换为一个 document PDF + question 的 Agent 输入。
- [ ] 不给模型任何 gold page/evidence。
- [ ] 保持项目原 tool list。
- [ ] 保持项目原 `<final>` protocol。
- [ ] 保持统一 `max_turns=10`、`max_tool_calls=8`。
- [ ] 每题只生成一个 benchmark prediction。
- [ ] eval-only 期间 optimizer step 为 0。
- [ ] 每条 prediction 有唯一 `task_id`。
- [ ] rollout 失败样本仍在官方分母内。
- [ ] DocVQA 2026 使用官方 `eval_utils.py`。
- [ ] LongDocURL 使用官方 `utils_score_v3.py` / `calculate_metrics.py`。
- [ ] DUDE 使用官方 `evaluate_submission.py`。
- [ ] MP-DocVQA 使用标准 threshold=0.5 ANLS。
- [ ] MP-DocVQA page metric 不错误命名为官方 APPA。
- [ ] LongDocURL/MP-DocVQA 额外报告 evidence/page navigation statistics。
- [ ] 四个 benchmark 均有 `official_metrics.json`。
- [ ] 四个 benchmark 均有 `agent_metrics.json`。
- [ ] 生成统一 `summary.md`。
- [ ] 全流程磁盘使用保持在约 100 GB 限制以内，并清理临时 archive/render cache。
