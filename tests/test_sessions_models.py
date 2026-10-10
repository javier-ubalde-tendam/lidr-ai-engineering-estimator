from app.sessions.models import SUMMARY_PREFIX, ConversationHistory, Message, ProjectMetadata


def test_history_append_adds_a_user_assistant_pair():
    history = ConversationHistory(max_turns=6)

    history.append(user="describe the project", assistant="here is the estimation")

    assert [m.role for m in history.messages] == ["user", "assistant"]
    assert history.messages[0].content == "describe the project"
    assert history.messages[1].content == "here is the estimation"


def test_append_no_longer_trims_the_window():
    history = ConversationHistory(max_turns=2)

    for i in range(4):  # 4 turnos = 8 mensajes, el doble del límite (2*2=4)
        history.append(user=f"user {i}", assistant=f"assistant {i}")

    # Qué se olvida lo decide la política de compresión, no append()
    assert len(history.messages) == 8
    assert history.anchors == []
    assert history.summary is None


def test_to_messages_list_orders_system_summary_anchors_and_recent_window():
    history = ConversationHistory(max_turns=2, summary="Earlier facts.")
    history.anchors = [Message(role="user", content="anchor user"), Message(role="assistant", content="anchor reply")]
    history.append(user="recent user", assistant="recent reply")

    messages = history.to_messages_list("system prompt")

    assert [m["content"] for m in messages] == [
        "system prompt",
        f"{SUMMARY_PREFIX}\nEarlier facts.",
        "anchor user",
        "anchor reply",
        "recent user",
        "recent reply",
    ]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("[Earlier conversation summary")


def test_to_messages_list_omits_the_summary_message_when_there_is_no_summary():
    history = ConversationHistory(max_turns=2)
    history.append(user="hello", assistant="hi")

    assert len(history.to_messages_list("system prompt")) == 3


def test_to_messages_list_prepends_the_system_prompt_without_storing_it():
    history = ConversationHistory(max_turns=6)
    history.append(user="hello", assistant="hi")

    messages = history.to_messages_list("you are an estimator")

    assert messages[0] == {"role": "system", "content": "you are an estimator"}
    assert len(messages) == 3  # system + 1 par
    # El system prompt no se guarda en el historial: se regenera cada vez a partir del metadata
    assert all(m.role != "system" for m in history.messages)


def test_metadata_is_empty_when_no_facts_are_known():
    assert ProjectMetadata().is_empty()


def test_metadata_is_not_empty_with_at_least_one_fact():
    assert not ProjectMetadata(project_name="Acme").is_empty()


def test_merge_with_keeps_previous_scalars_when_update_has_none():
    previous = ProjectMetadata(project_name="Acme", assumed_team_size=5)
    update = ProjectMetadata(project_name=None, assumed_team_size=None)

    merged = previous.merge_with(update)

    assert merged.project_name == "Acme"
    assert merged.assumed_team_size == 5


def test_merge_with_overwrites_scalars_when_update_has_a_value():
    previous = ProjectMetadata(project_name="Acme", assumed_team_size=5)
    update = ProjectMetadata(project_name="Renamed", assumed_team_size=8)

    merged = previous.merge_with(update)

    assert merged.project_name == "Renamed"
    assert merged.assumed_team_size == 8


def test_merge_with_unions_technologies_case_insensitively_without_duplicates():
    previous = ProjectMetadata(mentioned_technologies=["React", "Postgres"])
    update = ProjectMetadata(mentioned_technologies=["react", "Docker"])

    merged = previous.merge_with(update)

    assert merged.mentioned_technologies == ["React", "Postgres", "Docker"]
