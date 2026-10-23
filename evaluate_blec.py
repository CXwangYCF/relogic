"""Batch adapter for the sibling BLEC repository; does not modify BLEC rules."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BLEC_ROOT = ROOT / "BLEC-main"


def load_rows(path):
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [r for group in data.values() for r in group]
    rows = []
    for i, r in enumerate(records):
        logic = r.get("logic")
        if logic is None:
            # The original inference entry stores Input:<logic> before a newline.
            question = r.get("question", "")
            if not question.startswith("Input:"):
                raise ValueError(f"Record {i} has no logic or Input: field")
            logic = question[len("Input:"):].split("\n", 1)[0].strip()
        if not isinstance(logic, str) or not logic.strip():
            raise ValueError(f"Record {i} has an invalid logic field")
        rows.append({"idx": r.get("idx", i), "logic": logic,
                     "pred": r.get("pred", r.get("prediction", "")) or "",
                     "label": r.get("label", "") or ""})
    if not rows:
        raise ValueError("No records found")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["logic2text", "spider"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template", type=Path,
                        default=BLEC_ROOT / "tests/template_to_names_test.json")
    args = parser.parse_args()
    rows = load_rows(args.input)
    os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
    sys.path.insert(0, str(BLEC_ROOT))
    if args.dataset == "logic2text":
        from BLEC.Logic2text import BLECLogic2text
        evaluator = BLECLogic2text()
    else:
        from BLEC.Spider import BLECSpider
        evaluator = BLECSpider(template_path=str(args.template))

    results = []
    for row in rows:
        pred = row["pred"].strip()
        status = "evaluated"
        if not pred:
            errors = ["empty_prediction"]
            status = "empty_prediction"
        elif args.dataset == "spider" and row["logic"] not in evaluator.template_to_names:
            # BLECSpider.evaluate otherwise prints an error and returns [],
            # which must not be interpreted as a successful evaluation.
            errors = ["missing_sql_template"]
            status = "missing_sql_template"
        elif args.dataset == "logic2text":
            # This repository's wrapper forwards its first two arguments as-is
            # to logic_matching(logic, nl, truth), despite its parameter names.
            errors = evaluator.evaluate(row["logic"], pred, row["label"])
        else:
            errors = evaluator.evaluate(pred, row["logic"])
        results.append({**row, "errors": errors, "status": status, "passed": not errors})

    passed = sum(r["passed"] for r in results)
    summary = {"dataset": args.dataset,
               "metric_variant": ("logic2text_case_insensitive_operator_keywords"
                                  if args.dataset == "logic2text" else "spider_original"),
               "input": str(args.input.resolve()),
               "total": len(results), "passed": passed, "failed": len(results) - passed,
               "blec_pass_rate": passed / len(results),
               "empty_predictions": sum(r["status"] == "empty_prediction" for r in results),
               "missing_templates": sum(r["status"] == "missing_sql_template" for r in results)}
    output = args.output or args.input.with_name(args.input.stem + "_blec.json")
    if output.resolve() == args.input.resolve():
        parser.error("Output must differ from input")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": results},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"BLEC pass rate: {passed}/{len(results)} = {passed / len(results):.2%}")
    print(f"Details saved to: {output}")


if __name__ == "__main__":
    main()
