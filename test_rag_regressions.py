import os
from unittest.mock import MagicMock

os.environ.setdefault("GROQ_API_KEY", "dummy_for_testing")

from fastapi.testclient import TestClient

import main
from main import HistoryItem


if not main._index_ready:
    main._build_index()


def _mock_groq(content: str = "Mock response") -> MagicMock:
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_groq.chat.completions.create.return_value = mock_response
    main._groq = mock_groq
    return mock_groq


def test_skills_query_prefers_skills_context() -> None:
    chunks, meta = main._retrieve_chunks("skills", [])
    assert chunks, meta
    top_chunk = chunks[0].lower()
    assert "technical skills" in top_chunk or "core technical competencies" in top_chunk


def test_recent_achievements_surfaces_achievement_context() -> None:
    chunks, meta = main._retrieve_chunks("recent achievements", [])
    assert chunks, meta
    joined = "\n".join(chunks[:3]).lower()
    assert any(term in joined for term in [
        "hackathon",
        "placed in two competitive hackathons",
        "1st place",
        "3rd place",
        "150,000+ users",
        "scholarship",
    ])


def test_broad_profile_question_surfaces_summary_context() -> None:
    chunks, meta = main._retrieve_chunks("what does he do", [])
    assert chunks, meta
    joined = "\n".join(chunks[:3]).lower()
    assert any(term in joined for term in [
        "profile and demographics",
        "who is daniyal",
        "work experience",
        "data scientist",
        "research developer",
    ])


def test_recruiter_question_surfaces_skills_experience_context() -> None:
    chunks, meta = main._retrieve_chunks("what would a tech recruiter care about", [])
    assert chunks, meta
    joined = "\n".join(chunks[:4]).lower()
    assert any(term in joined for term in [
        "technical skills",
        "work experience",
        "projects",
        "education",
        "achievements",
    ])


def test_ceo_question_surfaces_impact_context() -> None:
    chunks, meta = main._retrieve_chunks("what would a ceo care about", [])
    assert chunks, meta
    joined = "\n".join(chunks[:4]).lower()
    assert any(term in joined for term in [
        "impact",
        "achievements",
        "frugality",
        "deliver results",
        "projects",
        "leadership",
    ])


def test_cto_question_surfaces_technical_context() -> None:
    chunks, meta = main._retrieve_chunks("what would a cto care about", [])
    assert chunks, meta
    joined = "\n".join(chunks[:4]).lower()
    assert any(term in joined for term in [
        "technical skills",
        "machine learning",
        "ai",
        "projects",
        "work experience",
    ])


def test_engineer_question_surfaces_build_context() -> None:
    chunks, meta = main._retrieve_chunks("what would an engineer care about", [])
    assert chunks, meta
    joined = "\n".join(chunks[:4]).lower()
    assert any(term in joined for term in [
        "technical skills",
        "projects",
        "work experience",
        "ai",
        "implementation",
    ])


def test_follow_up_retrieval_query_includes_prior_assistant_answer() -> None:
    history = [
        HistoryItem(role="assistant", content="He placed in two competitive hackathons and shipped systems serving 150,000+ users."),
    ]
    retrieval_query = main._build_retrieval_query("how is that related to ai", history)
    assert "placed in two competitive hackathons" in retrieval_query.lower()


def test_chat_uses_raw_user_message_not_data_wrapper() -> None:
    mock_groq = _mock_groq()
    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "message": "how is that related to ai",
            "history": [
                {"role": "assistant", "content": "He placed in two competitive hackathons and shipped systems serving 150,000+ users."},
            ],
        },
    )

    assert response.status_code == 200
    messages = mock_groq.chat.completions.create.call_args.kwargs["messages"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "how is that related to ai"
    assert "USER_DATA_TO_PROCESS" not in messages[-1]["content"]


def test_chat_caps_default_response_to_85_words() -> None:
    long_reply = " ".join([f"word{i}" for i in range(1, 131)])
    _mock_groq(long_reply)
    client = TestClient(main.app)

    response = client.post(
        "/chat",
        json={"message": "what has he built", "history": []},
    )

    assert response.status_code == 200
    reply = response.json()["reply"]
    assert len(reply.split()) == 85


def test_chat_allows_long_response_when_user_asks_for_detail() -> None:
    long_reply = " ".join([f"word{i}" for i in range(1, 131)])
    _mock_groq(long_reply)
    client = TestClient(main.app)

    response = client.post(
        "/chat",
        json={"message": "please explain in detail what he has built", "history": []},
    )

    assert response.status_code == 200
    reply = response.json()["reply"]
    assert len(reply.split()) == 130