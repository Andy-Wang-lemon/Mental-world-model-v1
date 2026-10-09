"""REST API 服务（stdlib HTTP）。"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.factory import create_llm_client
from mental_world_model.model import MentalWorldModel

HandlerFactory = Callable[[MentalWorldModel, LlmClient], type[BaseHTTPRequestHandler]]


def evaluate_request(
    model: MentalWorldModel,
    llm_client: LlmClient,
    body: dict,
) -> dict:
    """处理 /v1/evaluate 请求体并返回报告字典。"""
    yuanwen = str(body.get("yuanwen", "")).strip()
    if not yuanwen:
        raise ValueError("yuanwen is required")

    explain = body.get("explain", "none")
    role_context = str(body.get("role_context", "")).strip()
    focus_raw = body.get("focus_person", body.get("focusPerson", ""))
    focus_person = str(focus_raw or "").strip() or None
    result = model.evaluate_from_yuanwen(
        yuanwen,
        llm_client,
        update_seeds=False,
        role_context=role_context,
        focus_person=focus_person,
    )
    include_explanations = explain in ("all", "nonzero")
    verbose = bool(body.get("verbose", False))
    if verbose:
        report = result.to_report_dict(
            include_explanations=include_explanations,
            verbose=True,
        )
        if explain == "all":
            report["selected_explanation"] = result.explain_all_scores(only_nonzero=False)
        elif explain == "nonzero":
            report["selected_explanation"] = result.explain_all_scores(only_nonzero=True)
        return report
    return result.to_event_report_dict()


def create_handler(
    model: MentalWorldModel,
    llm_client: LlmClient,
) -> type[BaseHTTPRequestHandler]:
    """创建绑定 model/llm 的请求处理器。"""

    class EvaluateHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/v1/evaluate":
                self._send_json(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid json"})
                return

            try:
                report = evaluate_request(model, llm_client, data)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
                return
            except Exception as error:
                self._send_json(500, {"error": str(error)})
                return

            self._send_json(200, report)

    return EvaluateHandler


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    definitions_dir: Path,
    use_mock: bool = False,
) -> None:
    """启动 REST API 服务。"""
    from mental_world_model.llm.env_file import load_project_dotenv

    load_project_dotenv()

    model = MentalWorldModel.load(definitions_dir)
    llm_client = create_llm_client(use_mock=use_mock)
    handler = create_handler(model, llm_client)
    server = ThreadingHTTPServer((host, port), handler)

    print(f"mwm API listening on http://{host}:{port}")
    print(
        'POST /v1/evaluate  body: {"yuanwen": "...", "focus_person": "陈默", '
        '"explain": "none|all|nonzero"}'
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
