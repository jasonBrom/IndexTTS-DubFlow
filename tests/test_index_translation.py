from __future__ import annotations

import inspect
import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import original_dubber.index_translation as index_module
from original_dubber.index_translation import (
    HOMURA_MODEL,
    PUBLIC_API_BASE,
    PUBLIC_MODEL,
    IndexTranslator,
    completion_endpoint,
    glossary_pairs,
    translation_content,
)
from original_dubber.models import DubConfig, Segment
from original_dubber.pipeline import OriginalVoiceDubber
from original_dubber.translation import translate_segments


def result(text="Hello!", reason="stop"):
    return {"choices": [{"finish_reason": reason, "message": {"content": text}}]}


@pytest.fixture
def server():
    requests = []
    replies = [(200, result())]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), payload))
            status, body = replies.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{service.server_port}/v1", requests, replies
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=2)


def test_local_http_preserves_timeline_glossary_and_manual_edits(server):
    base, requests, _ = server
    locked = Segment(0, 0, 2, "你好", "人工译文", translation_locked=True)
    pending = Segment(1, 3, 5, "你来了", speaker="Alice")
    disabled = Segment(2, 6, 7, "唱歌", enabled=False)
    translate_segments(
        [locked, pending, disabled], backend="index", source_language="zh", target_language="EN",
        index_api_base=base, index_api_key="local-key", index_model="custom-alias",
        glossary="Alice=艾丽丝\nIndexTTS:IndexTTS", style="口语", context_size=1,
    )
    assert locked.target_text == "人工译文"
    assert pending.target_text == "Hello!"
    assert pending.start == 3 and pending.end == 5 and pending.speaker == "Alice"
    assert not disabled.target_text
    assert len(requests) == 1
    path, headers, payload = requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer local-key"
    assert payload["model"] == "custom-alias"
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    prompt = payload["messages"][0]["content"]
    assert "Alice -> 艾丽丝" in prompt and "人工译文" in prompt and "口语" in prompt


def test_homura_budget_and_no_neighbor_leak_when_context_disabled(server):
    base, requests, _ = server
    segment = Segment(1, 2, 4.5, "我们去看电影吧")
    translate_segments(
        [Segment(0, 0, 1, "上下文", "already"), segment],
        backend="homura", source_language="zh", target_language="EN", index_api_base=base,
        index_syllables_per_second=4, context_size=0, style="", glossary="电影=cinema",
    )
    payload = requests[0][2]
    assert payload["model"] == HOMURA_MODEL
    assert payload["temperature"] == 0.3
    prompt = payload["messages"][0]["content"]
    assert "10 个音节" in prompt and "电影 -> cinema" in prompt
    assert "上下文" not in prompt


def test_public_preset_never_forwards_other_credentials(monkeypatch):
    captured = []

    def open_request(request, **_kwargs):
        captured.append(request)
        return io.BytesIO(json.dumps(result()).encode())

    monkeypatch.setattr(index_module.urllib.request, "urlopen", open_request)
    translate_segments(
        [Segment(0, 0, 1, "你好")], backend="index_public", source_language="zh", target_language="EN",
        index_api_base="http://custom.invalid", index_model="custom", index_api_key="private-secret",
        api_key="another-secret",
    )
    assert captured[0].full_url == PUBLIC_API_BASE + "/chat/completions"
    assert captured[0].get_header("Authorization") is None
    assert json.loads(captured[0].data)["model"] == PUBLIC_MODEL


def test_transient_http_retry(server, monkeypatch):
    base, requests, replies = server
    replies[:] = [(429, {"error": "rate limited"}), (200, result("OK"))]
    delays = []
    monkeypatch.setattr(index_module.time, "sleep", delays.append)
    segment = Segment(0, 0, 1, "你好")
    IndexTranslator(api_base=base).translate([segment], source_language="zh", target_language="EN")
    assert segment.target_text == "OK" and len(requests) == 2 and delays == [1]


def test_invalid_key_is_not_retried_or_disclosed(server):
    base, requests, replies = server
    replies[:] = [(401, {"error": "private-secret"})]
    segment = Segment(0, 0, 1, "你好")
    with pytest.raises(RuntimeError, match="401") as error:
        IndexTranslator(api_base=base).translate([segment], source_language="zh", target_language="EN")
    assert "private-secret" not in str(error.value)
    assert len(requests) == 1 and segment.target_text == ""


@pytest.mark.parametrize("body", [result("partial", "length"), result("", "stop"),
                                  result(None), result("<think>still thinking"), {}, {"choices": []},
                                  result("blocked", "content_filter"), {"choices": [None]}])
def test_bad_responses_are_not_accepted(body):
    with pytest.raises(RuntimeError):
        translation_content(body)


def test_truncated_result_leaves_current_segment_unmodified(server):
    base, _, replies = server
    replies[:] = [(200, result("partial", "length"))]
    segment = Segment(0, 0, 1, "你好")
    with pytest.raises(RuntimeError, match="length"):
        IndexTranslator(api_base=base).translate([segment], source_language="zh", target_language="EN")
    assert segment.target_text == "" and segment.target_language == ""


def test_thinking_is_removed_but_quotation_marks_preserved():
    assert translation_content(result('<think>hidden</think>"Hello"')) == '"Hello"'


@pytest.mark.parametrize("base,expected", [
    ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
    ("http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"),
    ("https://example.com/api/v1/chat/completions", "https://example.com/api/v1/chat/completions"),
])
def test_endpoint_normalization(base, expected):
    assert completion_endpoint(base) == expected


@pytest.mark.parametrize("base", ["", "file:///tmp/a", "https://user:secret@example.com", "https://a/?key=secret"])
def test_invalid_endpoint_rejected(base):
    with pytest.raises(ValueError):
        completion_endpoint(base)


def test_glossary_formats():
    assert glossary_pairs('{"电影":"cinema"}') == "电影 -> cinema"
    assert glossary_pairs("电影=cinema\n人：person") == "电影 -> cinema、人 -> person"
    with pytest.raises(ValueError):
        glossary_pairs("invalid line")


@pytest.mark.parametrize("rate", [0, float("nan"), float("inf"), 13])
def test_invalid_syllable_rate_rejected(rate):
    with pytest.raises(ValueError):
        IndexTranslator(syllables_per_second=rate)


def test_config_validation_redaction_and_cache_invalidation(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fixture")
    config = DubConfig(str(video), str(tmp_path / "out"), "index-test", translation_backend="homura")
    config.index_api_key = "secret"
    config.validate()
    assert config.public_dict()["index_api_key"] == "***"
    dubber = OriginalVoiceDubber(config)
    first = dubber._analysis_key()
    config.index_syllables_per_second = 5
    assert dubber._analysis_key() != first
    second = dubber._analysis_key()
    config.index_model = "IndexTeam/Index-Homura-9B"
    assert dubber._analysis_key() != second
    config.index_api_base = ""
    with pytest.raises(ValueError):
        config.validate()


def test_timeout_retries_are_bounded(monkeypatch):
    attempts = []

    def fail(*_args, **_kwargs):
        attempts.append(1)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(index_module.urllib.request, "urlopen", fail)
    monkeypatch.setattr(index_module.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="连接失败"):
        IndexTranslator().translate([Segment(0, 0, 1, "Hi")], source_language="en", target_language="ZH")
    assert len(attempts) == 3


def test_gradio_inputs_reach_config_and_presets_select_correct_models(tmp_path):
    from original_dubber.ui import _make_config, build_app

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video fixture")
    app = build_app()
    try:
        analyze = next(f for f in app.fns.values() if f.fn and f.fn.__name__ == "analyze_ui")
        preset = next(f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "update_index_settings")
        parameters = list(inspect.signature(_make_config).parameters)
        assert len(analyze.inputs) == len(parameters)
        values = dict(zip(parameters, [c.value for c in analyze.inputs], strict=True))
        values.update(source_mode="本地路径", local_path=str(video), authorized=True,
                      index_api_key="private", index_syllables_per_second=5.2, index_max_tokens=2048)
        for label, backend, family in [
            ("Index-Translate 官方公网 API", "index_public", "Translate"),
            ("Index-Translate 本地/自建服务", "index", "Translate"),
            ("Index-Homura 音节控制（自建服务）", "homura", "Homura"),
        ]:
            updates = preset(label)
            assert updates[0]["visible"] is True
            assert updates[2]["visible"] == (backend != "index_public")
            assert updates[4]["visible"] == (backend == "homura")
            values.update(translation_backend=label, index_model=updates[3]["value"])
            config = _make_config(*[values[name] for name in parameters])
            config.validate()
            assert config.translation_backend == backend
            assert config.index_model == f"IndexTeam/Index-{family}-2B"
            assert config.index_syllables_per_second == 5.2 and config.index_max_tokens == 2048
            assert config.public_dict()["index_api_key"] == "***"
    finally:
        app.close()
