from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from local_decoder import LocalDecoder

ROOT = Path(__file__).resolve().parents[1]
BLEC_ROOT = ROOT / "BLEC-main"
sys.path.insert(0, str(BLEC_ROOT))


LOGIC2TEXT_PROMPT = """
Task: Translate the logic expression into natural language.

Requirements:
- Explicit semantic trigger words for every logic operator must be preserved.
- Comparison relations and superlative meanings must not be omitted.
- Do not replace key operations or entities with pronouns or vague references.
- The original logic should be recoverable through keyword matching.

Mandatory mappings:
count -> number of; avg -> average; sum -> total / sum; max -> highest / maximum;
min -> lowest / minimum; argmax -> row/entity with highest;
argmin -> row/entity with lowest; greater -> greater than / more than;
less -> less than / lower than; eq -> equal to; not_eq -> not equal to;
all -> all; most -> most; only -> only one; diff -> difference between.

For nth operators, explicitly include ordinal words. For filter operators, explicitly
include that, whose, or with. End with the tag "answer {natural language sentence}".
""".strip()

SPIDER_PROMPT = """
Task: Generate a natural language question based on the provided SQL query.
Output a clear and concise question that the SQL query is designed to answer.
End with the tag "answer {natural language question}".
""".strip()


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    task_name: str
    test_path: Path
    train_path: Path
    expected_count: int


DATASETS = {
    "logic2text": DatasetSpec(
        id="logic2text",
        name="Logic2Text",
        task_name="relogic_logic2text",
        test_path=ROOT / "dataset/relogic_data/logic2text/raw/test.json",
        train_path=ROOT / "dataset/relogic_data/logic2text/raw/train.json",
        expected_count=1092,
    ),
    "spider": DatasetSpec(
        id="spider",
        name="Spider",
        task_name="relogic_spider",
        test_path=ROOT / "dataset/relogic_data/spider/raw/test.json",
        train_path=ROOT / "dataset/relogic_data/spider/raw/train.json",
        expected_count=1034,
    ),
}


def load_cases(dataset_id: str) -> list[dict[str, str]]:
    spec = DATASETS[dataset_id]
    rows = json.loads(spec.test_path.read_text(encoding="utf-8"))
    if dataset_id == "logic2text":
        return [
            {"case_id": f"logic2text:{index}", "logic": str(row["logic_str"]), "label": str(row["sent"])}
            for index, row in enumerate(rows)
        ]
    return [
        {"case_id": f"spider:{index}", "logic": str(row["query"]), "label": str(row["question"])}
        for index, row in enumerate(rows)
    ]


def extract_answer(text: str) -> str:
    marker = "answer "
    lowered = text.lower()
    position = lowered.rfind(marker)
    result = text[position + len(marker):].strip() if position >= 0 else ""
    if result.startswith("{") and result.endswith("}"):
        result = result[1:-1].strip()
    return result


def load_demo_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[str] = []
    for row in payload.get("demo", []):
        blocks.append(
            f"{row['question']} {row['rationale']} The answer is {row['pred_ans']}."
        )
    if not blocks:
        raise ValueError(f"Auto-CoT example library is empty: {path}")
    return "\n\n".join(blocks) + "\n\n"


def build_prompt(dataset_id: str, logic: str, method_id: str, demo_text: str = "") -> str:
    instruction = LOGIC2TEXT_PROMPT if dataset_id == "logic2text" else SPIDER_PROMPT
    prompt = f"Input:{logic}\n{instruction}"
    if method_id == "auto_cot":
        prompt = demo_text + prompt
    return prompt + " Let's think step by step."


class BlecEvaluator:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        if dataset_id == "logic2text":
            from BLEC.Logic2text import BLECLogic2text
            self.impl: Any = BLECLogic2text()
        else:
            from BLEC.Spider import BLECSpider
            self.impl = BLECSpider(
                template_path=str(BLEC_ROOT / "tests/template_to_names_test.json")
            )

    def evaluate(self, logic: str, prediction: str, label: str) -> tuple[list[str], str]:
        if not prediction.strip():
            return ["empty_prediction"], "empty_prediction"
        if self.dataset_id == "spider" and logic not in self.impl.template_to_names:
            return ["missing_sql_template"], "missing_sql_template"
        if self.dataset_id == "logic2text":
            errors = self.impl.evaluate(logic, prediction, label)
        else:
            errors = self.impl.evaluate(prediction, logic)
        return list(errors), "evaluated"


def make_decoder(model_id: str) -> LocalDecoder:
    import os

    reasoning = os.getenv("RELOGIC_REASONING_EFFORT", "none")
    return LocalDecoder(
        base_url=os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1"),
        model_name=model_id,
        api_key=os.getenv("LOCAL_MODEL_API_KEY"),
        timeout=float(os.getenv("RELOGIC_MODEL_TIMEOUT", "600")),
        temperature=float(os.getenv("RELOGIC_TEMPERATURE", "0")),
        reasoning_effort=None if reasoning == "default" else reasoning,
    )


def probe_model(model_id: str) -> None:
    available = make_decoder(model_id).list_models()
    if model_id not in available:
        raise ValueError(
            f"Configured model {model_id!r} is not available from the local endpoint; "
            f"advertised models: {', '.join(available) or '(none)'}"
        )
