"""CLI 与 REST API 测试。"""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from mental_world_model.api.server import create_handler, evaluate_request
from mental_world_model.cli import main
from mental_world_model.llm.mock_client import MockLlmClient
from mental_world_model.model import MentalWorldModel
from http.server import ThreadingHTTPServer


class TestCli:
    def test_evaluate_with_mock(self, tmp_path):
        root = Path(__file__).resolve().parents[1]
        output = tmp_path / "report.json"
        code = main(
            [
                "--mock",
                "--definitions",
                str(root / "definitions"),
                "evaluate",
                "--text",
                "测试原文",
                "--output",
                str(output),
            ]
        )
        assert code == 0
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "events" in data
        assert isinstance(data["events"], list)
        assert len(data["events"]) == 1
        first = data["events"][0]
        assert "objective_event" in first
        assert "self_grasping" in first
        assert "dharma_grasping" in first
        assert "subjective_alambanas" in first
        assert "branches" in first

    def test_evaluate_auto_report_path(self, tmp_path, monkeypatch, capsys):
        root = Path(__file__).resolve().parents[1]
        monkeypatch.setattr("mental_world_model.cli.get_project_root", lambda: tmp_path)

        code = main(
            [
                "--mock",
                "--definitions",
                str(root / "definitions"),
                "evaluate",
                "--text",
                "测试原文",
            ]
        )
        assert code == 0

        reports_dir = tmp_path / "reports"
        assert reports_dir.is_dir()
        report_files = sorted(reports_dir.glob("report_*.json"))
        assert len(report_files) == 1
        assert report_files[0].name.startswith("report_")
        assert report_files[0].name.endswith(".json")

        data = json.loads(report_files[0].read_text(encoding="utf-8"))
        assert "events" in data

        captured = capsys.readouterr()
        assert "评估完成" in captured.out
        assert "结果已保存：" in captured.out
        assert f"reports/{report_files[0].name}" in captured.out


class TestApi:
    def test_evaluate_request(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        llm = MockLlmClient(default_response={"score": 0.0})
        report = evaluate_request(model, llm, {"yuanwen": "测试", "explain": "none"})
        assert "events" in report
        assert "score_breakdowns" not in report

    def test_evaluate_request_focus_person(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        llm = MockLlmClient(default_response={"score": 0.0})
        report = evaluate_request(
            model,
            llm,
            {"yuanwen": "测试", "focus_person": "陈默", "explain": "none"},
        )
        assert report.get("focus_person") == "陈默"
        assert report["events"][0].get("focus_person") == "陈默"

    def test_evaluate_request_requires_yuanwen(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        import pytest

        with pytest.raises(ValueError):
            evaluate_request(model, MockLlmClient(), {"yuanwen": ""})


class TestApiHttp:
    def test_http_evaluate_endpoint(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        llm = MockLlmClient(default_response={"score": 0.0})
        handler = create_handler(model, llm)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            payload = json.dumps({"yuanwen": "测试 HTTP", "explain": "none"}).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/evaluate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            assert response.status == 200
            assert "events" in body
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_health(self):
        root = Path(__file__).resolve().parents[1]
        model = MentalWorldModel.load(root / "definitions")
        handler = create_handler(model, MockLlmClient())
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
            assert body == {"status": "ok"}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
