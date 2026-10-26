from types import SimpleNamespace

from google.genai import types

import app.ask as ask_module


def test_ask_sends_previous_messages_to_the_api(monkeypatch):
    requests = []

    class FakeModels:
        def generate_content(self, *, model, contents):
            requests.append((model, contents))
            return SimpleNamespace(usage_metadata=None, text="新しい回答")

    class FakeClient:
        models = FakeModels()

    def fake_client(api_key):
        assert api_key == "test-key"
        return FakeClient()

    monkeypatch.setattr(ask_module.genai, "Client", fake_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    history = [
        types.Content(role="user", parts=[types.Part(text="前の質問")]),
        types.Content(role="model", parts=[types.Part(text="前の回答")]),
    ]

    answer = ask_module.ask("次の質問", history)

    assert answer == "新しい回答"
    assert requests == [
        (
            ask_module.MODEL,
            [
                *history,
                types.Content(role="user", parts=[types.Part(text="次の質問")]),
            ],
        )
    ]


def test_main_keeps_history_between_interactive_questions(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    questions = iter(["", "二つ目の質問", "終了"])

    def fake_input(prompt):
        assert prompt == "\n質問> "
        return next(questions)

    monkeypatch.setattr("builtins.input", fake_input)
    calls = []

    def fake_ask(question, history):
        calls.append((question, list(history)))
        return f"{question}への回答"

    monkeypatch.setattr(ask_module, "ask", fake_ask)

    ask_module.main()

    assert [question for question, _ in calls] == ["最初の質問", "二つ目の質問"]
    assert calls[0][1] == []
    assert calls[1][1] == [
        types.Content(role="user", parts=[types.Part(text="最初の質問")]),
        types.Content(role="model", parts=[types.Part(text="最初の質問への回答")]),
    ]
    assert "[終了]" in capsys.readouterr().out
