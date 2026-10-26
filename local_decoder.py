"""OpenAI-compatible local inference, using only the Python standard library."""

import json
import os
import urllib.error
import urllib.request


class LocalDecoder:
    def __init__(self, base_url=None, model_name=None, api_key=None,
                 timeout=600, temperature=0.0, reasoning_effort=None):
        self.base_url = (base_url or os.getenv("LOCAL_MODEL_BASE_URL")
                         or "http://localhost:11434/v1").rstrip("/")
        self.api_key = api_key or os.getenv("LOCAL_MODEL_API_KEY") or "ollama"
        self.timeout = timeout
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        # Local traffic must not accidentally go through an HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.model_name = model_name or os.getenv("LOCAL_MODEL_NAME")
        if not self.model_name:
            models = self._request("models").get("data", [])
            if len(models) != 1:
                raise ValueError("Specify --model from: " +
                                 ", ".join(model["id"] for model in models))
            self.model_name = models[0]["id"]

    def _request(self, path, payload=None):
        request = urllib.request.Request(
            self.base_url + "/" + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Local model HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Cannot reach local model at {self.base_url}: {exc}") from exc

    def decode(self, input_text, max_tokens=10240, temperature=None):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        response = self._request("chat/completions", payload)
        choice = response["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        if not content:
            message = choice["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            details = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "finish_reason": choice.get("finish_reason"),
                "reasoning_chars": len(reasoning),
                "usage": response.get("usage"),
            }
            raise RuntimeError(
                "Local model returned an empty final answer: " + json.dumps(details) +
                ". For run_inference.py, try --reasoning_effort none to disable "
                "native thinking while keeping the CoT prompt. If finish_reason is "
                "length, the generation was truncated; adjust --max_length_cot "
                "(or --max_length_direct for direct answers)."
            )
        return content, choice.get("finish_reason")

    def list_models(self):
        """Return model ids advertised by the configured local endpoint."""
        return [row["id"] for row in self._request("models").get("data", [])]
