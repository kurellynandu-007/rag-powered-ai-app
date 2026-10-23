from types import SimpleNamespace

import pytest

import app
from rag_pipeline import RetrievedChunk, UnsupportedFileTypeError


class FakeCompletions:
    def __init__(self, content: str = "A grounded answer.") -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeGroqClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def _passage(
    text: str = "Route A1 is a fictional demo route.",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_filename="routes.txt",
        page_number=2,
        distance=0.1,
        similarity=0.9,
    )


def test_answer_from_documents_retrieves_four_and_calls_configured_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    passage = _passage()
    client = FakeGroqClient()

    def fake_retrieve(collection, query, top_k, *, embedding_model):
        calls.append(
            {
                "collection": collection,
                "query": query,
                "top_k": top_k,
                "embedding_model": embedding_model,
            }
        )
        return [passage]

    monkeypatch.setattr(app, "retrieve_documents", fake_retrieve)
    monkeypatch.setattr(app, "create_groq_client", lambda: client)
    monkeypatch.setattr(app, "get_groq_model", lambda: "configured-model")

    answer, sources = app.answer_from_documents(
        "Where does A1 go?",
        collection="fake-collection",
        embedding_model="fake-embedding",
        history=[{"role": "user", "content": "Prior question"}],
    )

    assert answer == "A grounded answer."
    assert sources == [passage]
    assert calls == [
        {
            "collection": "fake-collection",
            "query": "Where does A1 go?",
            "top_k": 4,
            "embedding_model": "fake-embedding",
        }
    ]
    request = client.completions.calls[0]
    assert request["model"] == "configured-model"
    request_messages = request["messages"]
    assert isinstance(request_messages, list)
    assert "Source: routes.txt, page 2" in request_messages[-1]["content"]


def test_answer_from_documents_does_not_call_groq_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "retrieve_documents", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        app,
        "create_groq_client",
        lambda: pytest.fail("Groq must not be called without retrieved evidence"),
    )

    answer, sources = app.answer_from_documents(
        "Unknown question", collection=object(), embedding_model=object()
    )

    assert answer == app.NOT_FOUND_ANSWER
    assert sources == []


def test_generate_grounded_answer_includes_prompt_examples_and_history() -> None:
    client = FakeGroqClient()
    history = [{"role": "user", "content": "What is route A1?"}]

    app.generate_grounded_answer(
        "And where does it go?",
        [_passage()],
        client,
        "configured-model",
        history=history,
    )

    messages = client.completions.calls[0]["messages"]
    assert isinstance(messages, list)
    system_prompt = messages[0]["content"]
    assert "untrusted" in system_prompt
    assert "Example 1" in system_prompt
    assert "Example 2" in system_prompt
    assert messages[1] == history[0]
    assert "page 2" in messages[2]["content"]


def test_generate_grounded_answer_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        app.generate_grounded_answer(
            " ",
            [_passage()],
            FakeGroqClient(),
            "configured-model",
        )


def test_validate_upload_accepts_pdf_and_rejects_empty_or_unsupported() -> None:
    validated = app.validate_upload("sample.PDF", b"%PDF")
    assert validated.filename == "sample.PDF"
    assert validated.content == b"%PDF"

    with pytest.raises(UnsupportedFileTypeError, match="Upload a PDF or TXT"):
        app.validate_upload("sample.docx", b"not supported")
    with pytest.raises(app.UploadValidationError, match="is empty"):
        app.validate_upload("empty.txt", b"")


def test_validate_upload_rejects_files_over_20_mb() -> None:
    with pytest.raises(app.UploadValidationError, match="20 MB upload limit"):
        app.validate_upload("large.txt", b"x" * (app.MAX_UPLOAD_BYTES + 1))


def test_list_indexed_documents_counts_chunks_without_requiring_document_text() -> None:
    class FakeCollection:
        def get(self, *, include):
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {"source_filename": "routes.txt"},
                    {"source_filename": "routes.txt"},
                    {"source_filename": "guide.pdf"},
                    {},
                ]
            }

    assert app.list_indexed_documents(FakeCollection()) == {
        "guide.pdf": 1,
        "routes.txt": 2,
    }


def test_retrieve_from_collections_merges_and_limits_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_collection = object()
    upload_collection = object()
    retrieved_by_collection = {
        knowledge_collection: [
            _passage("kb close"),
            RetrievedChunk("kb far", "kb.txt", None, 0.6, 0.4),
        ],
        upload_collection: [
            RetrievedChunk("upload closest", "upload.txt", 1, 0.05, 0.95),
            RetrievedChunk("upload close", "upload.txt", 1, 0.2, 0.8),
            RetrievedChunk("upload far", "upload.txt", 1, 0.65, 0.35),
        ],
    }
    queried = []

    def fake_retrieve(collection, query, *, top_k, embedding_model):
        queried.append((collection, query, top_k, embedding_model))
        return retrieved_by_collection[collection]

    monkeypatch.setattr(app, "retrieve_documents", fake_retrieve)

    passages = app.retrieve_from_collections(
        knowledge_collection,
        upload_collection,
        "query",
        "embedding-model",
    )

    assert [passage.text for passage in passages] == [
        "upload closest",
        "kb close",
        "upload close",
        "kb far",
    ]
    assert queried == [
        (knowledge_collection, "query", 4, "embedding-model"),
        (upload_collection, "query", 4, "embedding-model"),
    ]


def test_fresh_knowledge_base_is_indexed_from_bundled_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "demo.txt").write_text(
        "A bundled demo fact.", encoding="utf-8"
    )
    indexed: list[tuple[object, object]] = []

    class EmptyCollection:
        def get(self, *, where, include):
            assert where == {"source_id": "demo.txt"}
            assert include == ["metadatas"]
            return {"ids": []}

    def fake_ingest(collection, root, *, embedding_model):
        indexed.append((root, embedding_model))
        return {"demo.txt": 1}

    monkeypatch.setattr(app, "KNOWLEDGE_BASE_PATH", knowledge_base)
    monkeypatch.setattr(app, "ingest_knowledge_base", fake_ingest)

    app._ensure_knowledge_base_indexed(EmptyCollection(), "offline-model")

    assert indexed == [(knowledge_base, "offline-model")]


def test_api_error_messages_do_not_expose_raw_exception_text() -> None:
    import httpx
    from groq import RateLimitError

    secret_like_detail = "secret-token-should-not-be-shown"
    error = RateLimitError(
        secret_like_detail,
        response=httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        ),
        body=None,
    )

    message = app._api_error_message(error)

    assert "rate-limiting" in message
    assert secret_like_detail not in message
