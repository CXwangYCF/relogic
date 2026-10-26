from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .service import ConflictError, TaskService


class CreateTaskRequest(BaseModel):
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    method_id: str = Field(min_length=1)


service = TaskService()
app = FastAPI(title="reLogic formal BLEC evaluation engine", version="1.0.0")


def handle_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ConflictError):
        return HTTPException(status_code=409, detail=exc.message)
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "suite": "relogic"}


@app.get("/api/v1/eval/capabilities")
def capabilities() -> dict:
    return service.capabilities()


@app.post("/api/v1/eval/tasks", status_code=202)
def create_task(
    payload: CreateTaskRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    try:
        task, reused = service.create(payload.model_dump(), idempotency_key)
    except Exception as exc:
        raise handle_error(exc)
    if reused:
        response.headers["X-Task-Reused"] = "true"
    return task


@app.get("/api/v1/eval/tasks")
def list_tasks(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    return {"items": service.list_tasks(limit)}


@app.get("/api/v1/eval/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    task = service.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.post("/api/v1/eval/tasks/{task_id}/pause")
def pause_task(task_id: str) -> dict:
    task = service.pause(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.post("/api/v1/eval/tasks/{task_id}/retry-failures", status_code=202)
def retry_failures(task_id: str) -> dict:
    try:
        task = service.retry_failures(task_id)
    except Exception as exc:
        raise handle_error(exc)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/api/v1/eval/tasks/{task_id}/result")
def result(task_id: str) -> dict:
    task = service.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/api/v1/eval/tasks/{task_id}/samples")
def samples(
    task_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    payload = service.samples(task_id, offset, limit)
    if payload is None:
        raise HTTPException(status_code=404, detail="task not found")
    return payload


@app.get("/api/v1/eval/tasks/{task_id}/artifacts/results.jsonl")
def artifact(task_id: str) -> Response:
    payload = service.artifact(task_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="task not found")
    return Response(
        content=payload,
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{task_id}-results.jsonl"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("relogic_service.app:app", host="127.0.0.1", port=8201)
