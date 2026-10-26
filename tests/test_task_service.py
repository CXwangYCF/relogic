import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from relogic_service.service import TaskService


CASES = [
    {"case_id": "logic2text:0", "logic": "eq { a ; b }", "label": "a equals b"},
    {"case_id": "logic2text:1", "logic": "count { rows }", "label": "number of rows"},
    {"case_id": "logic2text:2", "logic": "greater { a ; b }", "label": "a is greater than b"},
]


class FakeEvaluator:
    def __init__(self, _dataset_id):
        pass

    def evaluate(self, _logic, prediction, _label):
        return ([] if prediction else ["empty_prediction"], "evaluated" if prediction else "empty_prediction")


class FlakyDecoder:
    failures = {1}

    def decode(self, prompt, _max_tokens):
        index = 0 if "eq {" in prompt else 1 if "count {" in prompt else 2
        if index in self.failures:
            raise RuntimeError("temporary model failure")
        return "Reasoning. answer {formal result}", "stop"


class BlockingDecoder:
    entered = threading.Event()
    release = threading.Event()

    def decode(self, _prompt, _max_tokens):
        self.entered.set()
        self.release.wait(timeout=2)
        return "Reasoning. answer {formal result}", "stop"


class TaskServiceTest(unittest.TestCase):
    def test_checkpoint_retry_and_quality(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "relogic_service.service.load_cases", return_value=CASES
        ), patch(
            "relogic_service.service.make_decoder", return_value=FlakyDecoder()
        ), patch(
            "relogic_service.service.BlecEvaluator", FakeEvaluator
        ), patch(
            "relogic_service.service.probe_model"
        ):
            service = TaskService(Path(tmp))
            task, reused = service.create(
                {
                    "model_id": service.model_id,
                    "dataset_id": "logic2text",
                    "method_id": "zero_shot_cot",
                },
                "same-request",
            )
            self.assertFalse(reused)
            service._threads[task["task_id"]].join(timeout=5)
            completed = service.get(task["task_id"])
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["result"]["summary"]["processed"], 3)
            self.assertEqual(completed["result"]["summary"]["inference_errors"], 1)
            self.assertEqual(completed["quality_status"], "partial")

            same, reused = service.create(
                {
                    "model_id": service.model_id,
                    "dataset_id": "logic2text",
                    "method_id": "zero_shot_cot",
                },
                "same-request",
            )
            self.assertTrue(reused)
            self.assertEqual(same["task_id"], task["task_id"])

            FlakyDecoder.failures = set()
            service.retry_failures(task["task_id"])
            service._threads[task["task_id"]].join(timeout=5)
            retried = service.get(task["task_id"])
            self.assertEqual(retried["result"]["summary"]["inference_errors"], 0)
            self.assertEqual(retried["result"]["summary"]["processed"], 3)
            self.assertEqual(retried["quality_status"], "qualified")

            artifact = service.artifact(task["task_id"])
            rows = [json.loads(line) for line in artifact.splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertNotIn("generated_critique", rows[0])

    def test_autocot_is_unavailable_without_frozen_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = TaskService(Path(tmp))
            dataset = next(row for row in service.capabilities()["datasets"] if row["id"] == "logic2text")
            auto = next(row for row in dataset["methods"] if row["id"] == "auto_cot")
            self.assertFalse(auto["available"])
            self.assertTrue(auto["unavailable_reason"])

    def test_pause_keeps_completed_checkpoint(self):
        BlockingDecoder.entered.clear()
        BlockingDecoder.release.clear()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "relogic_service.service.load_cases", return_value=CASES
        ), patch(
            "relogic_service.service.make_decoder", return_value=BlockingDecoder()
        ), patch(
            "relogic_service.service.BlecEvaluator", FakeEvaluator
        ), patch(
            "relogic_service.service.probe_model"
        ):
            service = TaskService(Path(tmp))
            task, _ = service.create(
                {
                    "model_id": service.model_id,
                    "dataset_id": "logic2text",
                    "method_id": "zero_shot_cot",
                }
            )
            self.assertTrue(BlockingDecoder.entered.wait(timeout=2))
            service.pause(task["task_id"])
            BlockingDecoder.release.set()
            service._threads[task["task_id"]].join(timeout=5)
            paused = service.get(task["task_id"])
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["result"]["summary"]["processed"], 1)
            self.assertEqual(paused["quality_status"], "partial")

    def test_restart_marks_running_task_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "task-1"
            root.mkdir()
            (root / "task.json").write_text(json.dumps({
                "task_id": "task-1",
                "status": "running",
                "quality_status": "partial",
                "config": {"sample_count": 3},
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }), encoding="utf-8")
            (root / "results.json").write_text("[]", encoding="utf-8")
            service = TaskService(Path(tmp))
            self.assertEqual(service.get("task-1")["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
