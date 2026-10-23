import pytest

from rag_pipeline import chunk_text


def test_chunk_text_creates_overlapping_chunks() -> None:
    chunks = chunk_text("abcdefghijk", chunk_size=5, overlap=2)

    assert chunks == ["abcde", "defgh", "ghijk"]
    assert all(chunk for chunk in chunks)
    assert chunks[0][-2:] == chunks[1][:2]
    assert chunks[1][-2:] == chunks[2][:2]


def test_chunk_text_returns_no_chunks_for_empty_text() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t") == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (5, -1), (5, 5)],
)
def test_chunk_text_rejects_invalid_sizes(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("example", chunk_size=chunk_size, overlap=overlap)
