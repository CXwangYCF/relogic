from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from typing import Any

from .runtime import (
    BlecEvaluator,
    DATASETS,
    ROOT,
    build_prompt,
    extract_answer,
    load_cases,
    load_demo_text,
    make_decoder,
    probe_model,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@dataclass
class ConflictError(Exception):
    message: str


class TaskService:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or Path(
            os.getenv("RELOGIC_OUTPUT_DIR", str(ROOT / "task_output"))
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = os.getenv("LOCAL_MODEL_NAME", "qwen3.6:latest")
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._pause_events: dict[str, threading.Event] = {}
        self._reconcile_interrupted()

    def _task_dir(self, task_id: str) -> Path:
        return self.output_dir / task_id

    def _meta_path(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "task.json"

    def _results_path(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "results.json"

    def _read_meta(self, task_id: str) -> dict[str, Any] | None:
        path = self._meta_path(task_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _read_results(self, task_id: str) -> list[dict[str, Any]]:
        path = self._results_path(task_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def _write_meta(self, task: dict[str, Any]) -> None:
        task["updated_at"] = now_iso()
        atomic_json(self._meta_path(task["task_id"]), task)

    def _write_results(self, task_id: str, rows: list[dict[str, Any]]) -> None:
        atomic_json(self._results_path(task_id), rows)
        jsonl = self._task_dir(task_id) / "results.jsonl"
        temporary = jsonl.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda item: int(item["case_id"].split(":")[-1])):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(jsonl)

    def _reconcile_interrupted(self) -> None:
        for path in self.output_dir.glob("*/task.json"):
            task = json.loads(path.read_text(encoding="utf-8"))
            if task.get("status") in {"running", "pausing"}:
                task["status"] = "interrupted"
                task["quality_status"] = "partial"
                task["error"] = "Engine restarted before the task reached a terminal state."
                task["updated_at"] = now_iso()
                atomic_json(path, task)

    def _dataset_hash(self, dataset_id: str) -> str:
        return sha256_file(DATASETS[dataset_id].test_path)

    def _demo_dir(self, dataset_id: str) -> Path:
        safe_model = self.model_id.replace("/", "_").replace(":", "_")
        return ROOT / "artifacts/autocot" / safe_model / dataset_id

    def _demo_status(self, dataset_id: str) -> dict[str, Any]:
        directory = self._demo_dir(dataset_id)
        demo = directory / "demo.json"
        manifest = directory / "manifest.json"
        if not demo.exists() or not manifest.exists():
            return {
                "id": "auto_cot",
                "name": "Auto-CoT",
                "available": False,
                "artifact_sha256": None,
                "unavailable_reason": "正式示例库尚未准备",
            }
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            actual = sha256_file(demo)
            valid = (
                payload.get("dataset_id") == dataset_id
                and payload.get("model_id") == self.model_id
                and payload.get("demo_sha256") == actual
                and payload.get("train_sha256") == sha256_file(DATASETS[dataset_id].train_path)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            valid = False
            actual = None
        return {
            "id": "auto_cot",
            "name": "Auto-CoT",
            "available": valid,
            "artifact_sha256": actual if valid else None,
            "unavailable_reason": None if valid else "示例库清单或哈希校验失败",
        }

    def capabilities(self) -> dict[str, Any]:
        datasets = []
        for dataset_id, spec in DATASETS.items():
            rows = load_cases(dataset_id)
            datasets.append(
                {
                    "id": dataset_id,
                    "name": spec.name,
                    "sample_count": len(rows),
                    "expected_count": spec.expected_count,
                    "sha256": self._dataset_hash(dataset_id),
                    "methods": [
                        {
                            "id": "zero_shot_cot",
                            "name": "Zero-shot-CoT",
                            "available": True,
                            "artifact_sha256": None,
                            "unavailable_reason": None,
                        },
                        self._demo_status(dataset_id),
                    ],
                }
            )
        return {
            "suite": {"id": "relogic", "name": "reLogic BLEC", "version": "v1"},
            "target_model": {
                "id": self.model_id,
                "name": self.model_id,
                "transport": "openai-compatible-local",
            },
            "evaluator": {"id": "blec", "name": "BLEC deterministic rules"},
            "datasets": datasets,
            "protocol": {
                "temperature": float(os.getenv("RELOGIC_TEMPERATURE", "0")),
                "max_tokens": 10240,
                "reasoning_effort": os.getenv("RELOGIC_REASONING_EFFORT", "none"),
                "prompt_version": "relogic-formal-v1",
            },
        }

    def _active_task(self) -> dict[str, Any] | None:
        for task in self.list_tasks(limit=200):
            if task["status"] in {"running", "pausing"}:
                return task
        return None

    def create(self, request: dict[str, str], idempotency_key: str | None = None) -> tuple[dict[str, Any], bool]:
        model_id = request.get("model_id")
        dataset_id = request.get("dataset_id")
        method_id = request.get("method_id")
        if model_id != self.model_id:
            raise ValueError("model_id does not match engine capabilities")
        if dataset_id not in DATASETS:
            raise ValueError("unknown dataset_id")
        if method_id not in {"zero_shot_cot", "auto_cot"}:
            raise ValueError("unknown method_id")
        method = next(item for item in self.capabilities()["datasets"] if item["id"] == dataset_id)
        method = next(item for item in method["methods"] if item["id"] == method_id)
        if not method["available"]:
            raise ValueError(method["unavailable_reason"] or "method unavailable")
        probe_model(model_id)

        with self._lock:
            if idempotency_key:
                for task in self.list_tasks(limit=200):
                    if task.get("idempotency_key") == idempotency_key:
                        return self.get(task["task_id"]), True
            active = self._active_task()
            config = {"model_id": model_id, "dataset_id": dataset_id, "method_id": method_id}
            if active:
                if all(active["config"].get(key) == value for key, value in config.items()):
                    return self.get(active["task_id"]), True
                raise ConflictError("another formal evaluation is already running")

            task_id = f"relogic-{uuid.uuid4().hex[:12]}"
            cases = load_cases(dataset_id)
            task = {
                "task_id": task_id,
                "status": "running",
                "quality_status": "partial",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "idempotency_key": idempotency_key,
                "config": {
                    **config,
                    "model_name": self.model_id,
                    "dataset_name": DATASETS[dataset_id].name,
                    "method_name": method["name"],
                    "sample_count": len(cases),
                    "dataset_sha256": self._dataset_hash(dataset_id),
                    "demo_sha256": method.get("artifact_sha256"),
                },
                "attempt": 1,
                "error": None,
            }
            self._write_meta(task)
            self._write_results(task_id, [])
            self._start(task_id, retry_only=False)
            return self.get(task_id), False

    def _start(self, task_id: str, retry_only: bool) -> None:
        pause = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(task_id, pause, retry_only),
            daemon=True,
            name=f"relogic-{task_id}",
        )
        self._pause_events[task_id] = pause
        self._threads[task_id] = thread
        thread.start()

    def _run(self, task_id: str, pause: threading.Event, retry_only: bool) -> None:
        task = self._read_meta(task_id)
        if not task:
            return
        dataset_id = task["config"]["dataset_id"]
        method_id = task["config"]["method_id"]
        cases = load_cases(dataset_id)
        existing = {row["case_id"]: row for row in self._read_results(task_id)}
        if retry_only:
            selected = [case for case in cases if existing.get(case["case_id"], {}).get("status") == "inference_error"]
        else:
            selected = [case for case in cases if case["case_id"] not in existing]

        demo_text = ""
        if method_id == "auto_cot":
            demo_text = load_demo_text(self._demo_dir(dataset_id) / "demo.json")
        evaluator = BlecEvaluator(dataset_id)
        try:
            decoder = make_decoder(task["config"]["model_id"])
            for case in selected:
                if pause.is_set():
                    break
                try:
                    response, finish_reason = decoder.decode(
                        build_prompt(dataset_id, case["logic"], method_id, demo_text),
                        10240,
                    )
                    prediction = extract_answer(response)
                    errors, status = evaluator.evaluate(case["logic"], prediction, case["label"])
                    row = {
                        **case,
                        "prediction": prediction,
                        "finish_reason": finish_reason,
                        "status": status,
                        "blec_errors": errors,
                        "passed": not errors,
                        "error": None,
                    }
                except Exception as exc:
                    row = {
                        **case,
                        "prediction": "",
                        "finish_reason": None,
                        "status": "inference_error",
                        "blec_errors": [],
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                with self._lock:
                    existing[case["case_id"]] = row
                    self._write_results(task_id, list(existing.values()))
                    current = self._read_meta(task_id)
                    if current:
                        self._write_meta(current)
        except Exception as exc:
            task = self._read_meta(task_id) or task
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            self._write_meta(task)
            return

        task = self._read_meta(task_id) or task
        if pause.is_set():
            task["status"] = "paused"
        else:
            task["status"] = "completed"
        task["error"] = None
        self._write_meta(task)
        self._finalize_quality(task_id)

    def _summary(self, task: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = int(task["config"]["sample_count"])
        evaluated = sum(row["status"] != "inference_error" for row in rows)
        passed = sum(bool(row.get("passed")) for row in rows)
        inference_errors = sum(row["status"] == "inference_error" for row in rows)
        empty = sum(row["status"] == "empty_prediction" for row in rows)
        missing = sum(row["status"] == "missing_sql_template" for row in rows)
        return {
            "total": total,
            "processed": len(rows),
            "evaluated": evaluated,
            "blec_passed": passed,
            "blec_failed": max(0, evaluated - passed),
            "blec_pass_rate": round(passed / evaluated * 100, 4) if evaluated else None,
            "inference_errors": inference_errors,
            "empty_predictions": empty,
            "missing_templates": missing,
        }

    def _finalize_quality(self, task_id: str) -> None:
        task = self._read_meta(task_id)
        if not task:
            return
        summary = self._summary(task, self._read_results(task_id))
        task["quality_status"] = (
            "qualified"
            if task["status"] == "completed"
            and summary["processed"] == summary["total"]
            and summary["inference_errors"] == 0
            and summary["missing_templates"] == 0
            else "partial"
        )
        self._write_meta(task)

    def get(self, task_id: str) -> dict[str, Any] | None:
        task = self._read_meta(task_id)
        if not task:
            return None
        rows = self._read_results(task_id)
        summary = self._summary(task, rows)
        total = summary["total"]
        processed = summary["processed"]
        task["execution_status"] = {
            "total": total,
            "processed": processed,
            "in_progress": 1 if task["status"] in {"running", "pausing"} else 0,
            "pending": max(0, total - processed),
            "inference_errors": summary["inference_errors"],
            "progress": round(processed / total * 100, 2) if total else 0,
        }
        task["result"] = {"summary": summary, "quality_status": task["quality_status"]}
        return task

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        tasks = []
        for path in self.output_dir.glob("*/task.json"):
            task = json.loads(path.read_text(encoding="utf-8"))
            tasks.append(task)
        tasks.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return [self.get(task["task_id"]) for task in tasks[:limit]]

    def pause(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._read_meta(task_id)
            if not task:
                return None
            if task["status"] == "running":
                task["status"] = "pausing"
                self._write_meta(task)
                self._pause_events.get(task_id, threading.Event()).set()
            return self.get(task_id)

    def retry_failures(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._read_meta(task_id)
            if not task:
                return None
            if self._active_task():
                raise ConflictError("another formal evaluation is already running")
            errors = [row for row in self._read_results(task_id) if row["status"] == "inference_error"]
            if not errors:
                raise ValueError("task has no inference errors to retry")
            task["status"] = "running"
            task["quality_status"] = "partial"
            task["attempt"] = int(task.get("attempt", 1)) + 1
            task["error"] = None
            self._write_meta(task)
            self._start(task_id, retry_only=True)
            return self.get(task_id)

    def samples(self, task_id: str, offset: int, limit: int) -> dict[str, Any] | None:
        if not self._read_meta(task_id):
            return None
        rows = sorted(self._read_results(task_id), key=lambda item: int(item["case_id"].split(":")[-1]))
        sanitized = [
            {
                "case_id": row["case_id"],
                "logic": row["logic"],
                "label": row["label"],
                "prediction": row["prediction"],
                "status": row["status"],
                "blec_errors": row["blec_errors"],
                "passed": row["passed"],
                "error": row["error"],
            }
            for row in rows[offset: offset + limit]
        ]
        return {"items": sanitized, "total": len(rows)}

    def artifact(self, task_id: str) -> str | None:
        page = self.samples(task_id, 0, 10_000)
        if page is None:
            return None
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in page["items"])
