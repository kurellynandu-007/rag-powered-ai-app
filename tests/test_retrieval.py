import hashlib
import re
from pathlib import Path
from typing import Sequence

import pytest

from rag_pipeline import (
    ExtractedPage,
    RetrievedChunk,
    create_vector_collection,
    index_document,
    retrieve_documents,
)


class DeterministicEmbeddingModel:
    """Offline token-vector model for deterministic ChromaDB integration tests."""

    dimensions = 4096

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        assert convert_to_numpy
        assert normalize_embeddings
        vectors: list[list[float]] = []
        for sentence in sentences:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-z0-9]+", sentence.casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[index] += 1.0
            magnitude = sum(value * value for value in vector) ** 0.5
            if magnitude:
                vector = [value / magnitude for value in vector]
            vectors.append(vector)
        return vectors


def _temporary_collection(tmp_path: Path, name: str):
    return create_vector_collection(
        collection_name=name,
        chroma_db_path=tmp_path / "chroma",
    )


class FakeEmbeddingModel:
    def encode(self, sentences, *, convert_to_numpy=True, normalize_embeddings=True):
        assert convert_to_numpy
        assert normalize_embeddings
        return [[1.0, 0.0] for _ in sentences]


class FakeCollection:
    def __init__(self, result):
        self.result = result
        self.query_args = None

    def query(self, *, query_embeddings, n_results, include):
        self.query_args = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include,
        }
        return self.result


def test_retrieve_documents_returns_source_and_cosine_metadata() -> None:
    collection = FakeCollection(
        {
            "documents": [["sample route", "unrelated passage"]],
            "metadatas": [
                [
                    {"source_filename": "routes.txt", "page_number": 2},
                    {"source_filename": "other.txt"},
                ]
            ],
            "distances": [[0.2, 0.9]],
        }
    )

    passages = retrieve_documents(
        collection,
        "route to the library",
        embedding_model=FakeEmbeddingModel(),
    )

    assert passages == [
        RetrievedChunk(
            text="sample route",
            source_filename="routes.txt",
            page_number=2,
            distance=0.2,
            similarity=0.8,
        )
    ]
    assert collection.query_args == {
        "query_embeddings": [[1.0, 0.0]],
        "n_results": 4,
        "include": ["documents", "metadatas", "distances"],
    }


def test_retrieve_documents_returns_empty_list_when_no_match() -> None:
    collection = FakeCollection(
        {
            "documents": [["distant passage"]],
            "metadatas": [[{"source_filename": "other.txt"}]],
            "distances": [[0.95]],
        }
    )

    assert retrieve_documents(
        collection, "question", embedding_model=FakeEmbeddingModel()
    ) == []


def test_retrieve_documents_returns_empty_list_for_empty_collection() -> None:
    collection = FakeCollection(
        {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    )

    assert retrieve_documents(
        collection, "demo query", embedding_model=FakeEmbeddingModel()
    ) == []


def test_unique_synthetic_fact_is_indexed_and_retrieved(tmp_path: Path) -> None:
    collection = _temporary_collection(tmp_path, "orbit-fact")
    embedding_model = DeterministicEmbeddingModel()
    unique_fact = (
        "Project ORBIT-731 uses the internal launch code BLUE-KITE-482."
    )

    indexed_count = index_document(
        [ExtractedPage(unique_fact, page_number=7)],
        "orbit_plan.txt",
        collection,
        embedding_model=embedding_model,
        source_id="orbit-plan",
    )
    results = retrieve_documents(
        collection,
        "What internal launch code does Project ORBIT-731 use?",
        embedding_model=embedding_model,
    )

    assert indexed_count == 1
    assert len(results) == 1
    assert results[0].text == unique_fact
    assert results[0].source_filename == "orbit_plan.txt"
    assert results[0].page_number == 7
    assert results[0].distance is not None
    assert results[0].similarity is not None


def test_unrelated_query_returns_no_document_evidence(tmp_path: Path) -> None:
    collection = _temporary_collection(tmp_path, "unrelated-query")
    embedding_model = DeterministicEmbeddingModel()
    index_document(
        [
            ExtractedPage(
                "Project ORBIT-731 uses the internal launch code BLUE-KITE-482."
            )
        ],
        "orbit_plan.txt",
        collection,
        embedding_model=embedding_model,
        source_id="orbit-plan",
    )

    assert retrieve_documents(
        collection,
        "What are the ingredients in a chocolate cake?",
        embedding_model=embedding_model,
    ) == []


def test_empty_chroma_collection_returns_no_fabricated_evidence(
    tmp_path: Path,
) -> None:
    collection = _temporary_collection(tmp_path, "empty-retrieval")

    assert retrieve_documents(
        collection,
        "What is the launch code?",
        embedding_model=DeterministicEmbeddingModel(),
    ) == []


def test_reingesting_document_does_not_create_duplicate_chunks(
    tmp_path: Path,
) -> None:
    collection = _temporary_collection(tmp_path, "duplicate-ingestion")
    embedding_model = DeterministicEmbeddingModel()
    pages = [
        ExtractedPage(
            "Project ORBIT-731 uses the internal launch code BLUE-KITE-482."
        )
    ]

    first_count = index_document(
        pages,
        "orbit_plan.txt",
        collection,
        embedding_model=embedding_model,
        source_id="orbit-plan",
    )
    second_count = index_document(
        pages,
        "orbit_plan.txt",
        collection,
        embedding_model=embedding_model,
        source_id="orbit-plan",
    )

    assert first_count == second_count == 1
    assert collection.count() == 1
    stored = collection.get(
        where={"source_id": "orbit-plan"},
        include=["documents", "metadatas"],
    )
    assert stored["documents"] == [
        "Project ORBIT-731 uses the internal launch code BLUE-KITE-482."
    ]


@pytest.mark.parametrize(
    ("query", "top_k", "max_distance"),
    [("", 4, 0.7), ("  ", 4, 0.7), ("route", 0, 0.7), ("route", 4, -1)],
)
def test_retrieve_documents_rejects_invalid_arguments(
    query: str, top_k: int, max_distance: float
) -> None:
    with pytest.raises(ValueError):
        retrieve_documents(
            FakeCollection({}),
            query,
            top_k=top_k,
            max_distance=max_distance,
            embedding_model=FakeEmbeddingModel(),
        )
