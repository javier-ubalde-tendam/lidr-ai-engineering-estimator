from app.sessions.compression import AnchorDetector, CompressionPolicy
from app.sessions.models import ConversationHistory, Message

MAX_TURNS = 2
ANCHOR_TURN = "We signed the contract yesterday, so the scope is final."


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, list[Message]]] = []

    def summarize(self, *, previous_summary: str | None, evicted: list[Message]) -> str | None:
        self.calls.append((previous_summary, evicted))
        return f"summary of {len(evicted) // 2} turn(s)"


def _policy(summarizer: FakeSummarizer) -> CompressionPolicy:
    return CompressionPolicy(anchor_detector=AnchorDetector(mode="heuristic"), summarizer=summarizer)  # type: ignore[arg-type]


def _history_with_turns(texts: list[str]) -> ConversationHistory:
    history = ConversationHistory(max_turns=MAX_TURNS)
    for index, text in enumerate(texts):
        history.append(user=text, assistant=f"reply {index}")
    return history


TURNS = ["Turn 0: a web portal for gyms.", ANCHOR_TURN, "Turn 2: add a calendar.", "Turn 3: add payments.", "Turn 4: add reports."]


def test_anchor_goes_to_anchors_and_the_rest_is_summarised_with_a_single_call():
    summarizer = FakeSummarizer()
    history = _history_with_turns(TURNS)

    _policy(summarizer).apply(history)

    assert [m.content for m in history.anchors] == [ANCHOR_TURN, "reply 1"]
    assert len(summarizer.calls) == 1  # una sola llamada con todos los turnos expulsados que no eran anclas
    assert [m.content for m in summarizer.calls[0][1]] == [TURNS[0], "reply 0", TURNS[2], "reply 2"]
    assert history.summary == "summary of 2 turn(s)"
    # La ventana queda en su límite y con los turnos más recientes
    assert [m.content for m in history.messages] == [TURNS[3], "reply 3", TURNS[4], "reply 4"]


def test_the_previous_summary_is_passed_to_the_summarizer():
    summarizer = FakeSummarizer()
    history = _history_with_turns(TURNS[:3])
    history.summary = "old summary"

    _policy(summarizer).apply(history)

    assert summarizer.calls[0][0] == "old summary"


def test_apply_is_idempotent():
    summarizer = FakeSummarizer()
    history = _history_with_turns(TURNS)
    policy = _policy(summarizer)

    policy.apply(history)
    snapshot = history.model_copy(deep=True)
    policy.apply(history)

    assert history == snapshot
    assert len(summarizer.calls) == 1


def test_nothing_happens_while_the_window_is_not_exceeded():
    summarizer = FakeSummarizer()
    history = _history_with_turns(TURNS[:MAX_TURNS])

    _policy(summarizer).apply(history)

    assert summarizer.calls == []
    assert history.summary is None
    assert history.anchors == []
    assert len(history.messages) == 2 * MAX_TURNS


def test_applying_after_every_turn_keeps_the_window_bounded():
    summarizer = FakeSummarizer()
    policy = _policy(summarizer)
    history = ConversationHistory(max_turns=MAX_TURNS)

    for index, text in enumerate(TURNS):
        history.append(user=text, assistant=f"reply {index}")
        policy.apply(history)
        assert len(history.messages) <= 2 * MAX_TURNS

    assert len(history.anchors) == 2
    assert history.summary is not None
    # Cada par expulsado sin ancla provoca una llamada (aquí, dos turnos expulsados en momentos distintos)
    assert len(summarizer.calls) == 2


def test_history_compressed_is_logged_with_the_counters():
    from structlog.testing import capture_logs

    history = _history_with_turns(TURNS)

    with capture_logs() as logs:
        _policy(FakeSummarizer()).apply(history)

    event = next(log for log in logs if log["event"] == "history_compressed")
    assert event["promoted_anchors"] == 1
    assert "signed_contract" in event["anchor_rules"]
    assert event["evicted_to_summary"] == 4
    assert event["anchors_count"] == 2
    assert event["recent_messages"] == 4
    assert event["summary_chars"] == len("summary of 2 turn(s)")
