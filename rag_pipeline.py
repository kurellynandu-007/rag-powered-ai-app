"""Local document extraction, embedding, ChromaDB indexing, and retrieval."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from config import CHROMA_DB_PATH, EMBEDDING_MODEL, KNOWLEDGE_BASE_PATH


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_MAX_COSINE_DISTANCE = 0.7
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentError(ValueError):
    """Raised when a document cannot be read or contains no usable text."""


class UnsupportedFileTypeError(DocumentError):
    """Raised when a document is not a supported PDF or TXT file."""


class VectorStoreError(RuntimeError):
    """Raised when vector indexing or retrieval fails."""


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from a document page; TXT files use page_number=None."""

    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class TextChunk:
    """A text fragment with the source metadata persisted alongside it."""

    text: str
    metadata: dict[str, str | int]
    chunk_id: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved passage and its source and vector-distance information."""

    text: str
    source_filename: str
    page_number: int | None
    distance: float | None
    similarity: float | None


@dataclass(frozen=True)
class UploadedDocument:
    """An uploaded document represented independently of any UI framework."""

    filename: str
    content: bytes


class EmbeddingModel(Protocol):
    """The sentence-transformers functionality needed by this module."""

    def encode(
        self, sentences: Sequence[str], *, convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> Any: ...


class ChromaCollection(Protocol):
    """The ChromaDB collection operations used by the pipeline."""

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str | int]],
    ) -> Any: ...

    def get(
        self, *, where: dict[str, str], include: list[str]
    ) -> dict[str, Any]: ...

    def delete(
        self,
        *,
        where: dict[str, str] | None = None,
        ids: list[str] | None = None,
    ) -> Any: ...

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, Any]: ...


def _validate_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type for {filename!r}. Supported types: {supported}."
        )
    return suffix


def _extract_txt(filename: str, content: bytes) -> list[ExtractedPage]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DocumentError(
            f"Could not decode {filename!r} as UTF-8. Save the text file using UTF-8."
        ) from error
    if not text.strip():
        raise DocumentError(f"{filename!r} is empty or contains only whitespace.")
    return [ExtractedPage(text=text)]


def _extract_pdf(filename: str, content: bytes) -> list[ExtractedPage]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise DocumentError(
                f"{filename!r} is encrypted and cannot be read without a password."
            )

        extracted_pages = [
            ExtractedPage(text=page.extract_text() or "", page_number=page_number)
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except DocumentError:
        raise
    except (PdfReadError, OSError, ValueError) as error:
        raise DocumentError(f"Could not read PDF {filename!r}: {error}") from error

    if not any(page.text.strip() for page in extracted_pages):
        raise DocumentError(
            f"No extractable text was found in {filename!r}. It may be empty or "
            "scanned; this pipeline does not provide OCR."
        )
    return [page for page in extracted_pages if page.text.strip()]


def extract_document_text(file_path: str | Path) -> list[ExtractedPage]:
    """Extract text from a PDF or UTF-8 TXT file on disk."""
    path = Path(file_path)
    suffix = _validate_file_type(path.name)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DocumentError(f"Could not read {path.name!r}: {error}") from error

    if suffix == ".pdf":
        return _extract_pdf(path.name, content)
    return _extract_txt(path.name, content)


def extract_uploaded_document(filename: str, content: bytes) -> list[ExtractedPage]:
    """Extract text from uploaded PDF/TXT bytes without depending on Streamlit."""
    suffix = _validate_file_type(filename)
    if not content:
        raise DocumentError(f"{filename!r} is empty.")
    if suffix == ".pdf":
        return _extract_pdf(filename, content)
    return _extract_txt(filename, content)


def _split_page_text(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            preferred_break = text.rfind(" ", start + chunk_size // 2, end)
            if preferred_break > start:
                end = preferred_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        whitespace = text.find(" ", next_start, min(next_start + 40, end))
        start = whitespace + 1 if whitespace >= next_start else next_start
    return chunks


def chunk_text(
    text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """Split plain text into overlapping, approximately sized chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if not text:
        return []
    return _split_page_text(text, chunk_size, overlap)


def create_document_chunks(
    pages: Sequence[ExtractedPage],
    source_filename: str,
    *,
    source_id: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Chunk extracted pages and attach deterministic IDs and source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    stable_source_id = source_id or hashlib.sha256(
        source_filename.casefold().encode("utf-8")
    ).hexdigest()
    result: list[TextChunk] = []
    for page in pages:
        for page_chunk_index, text in enumerate(
            _split_page_text(page.text, chunk_size, overlap)
        ):
            metadata: dict[str, str | int] = {
                "source_id": stable_source_id,
                "source_filename": Path(source_filename).name,
                "chunk_index": page_chunk_index,
            }
            page_key = str(page.page_number) if page.page_number is not None else "txt"
            if page.page_number is not None:
                metadata["page_number"] = page.page_number
            chunk_hash = hashlib.sha256(
                f"{stable_source_id}:{page_key}:{page_chunk_index}".encode("utf-8")
            ).hexdigest()
            result.append(TextChunk(text=text, metadata=metadata, chunk_id=chunk_hash))
    return result


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> EmbeddingModel:
    """Load the configured local sentence-transformers model on demand."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str], model: EmbeddingModel | None = None
) -> list[list[float]]:
    """Generate normalized local embeddings; Groq is not used for embeddings."""
    if not texts:
        return []
    embedding_model = model or load_embedding_model()
    vectors = embedding_model.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True
    )
    return [[float(value) for value in vector] for vector in vectors]


def create_vector_collection(
    collection_name: str = "milemate_knowledge",
    chroma_db_path: str | Path = CHROMA_DB_PATH,
) -> ChromaCollection:
    """Open or create a persistent cosine-distance ChromaDB collection."""
    import chromadb
    from chromadb.errors import ChromaError

    try:
        client = chromadb.PersistentClient(path=str(chroma_db_path))
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    except (ChromaError, OSError, ValueError) as error:
        raise VectorStoreError(
            f"Could not open ChromaDB at {str(chroma_db_path)!r}: {error}"
        ) from error


def index_document(
    pages: Sequence[ExtractedPage],
    source_filename: str,
    collection: ChromaCollection,
    *,
    embedding_model: EmbeddingModel | None = None,
    source_id: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Embed and index one document, replacing any older chunks for its source."""
    chunks = create_document_chunks(
        pages,
        source_filename,
        source_id=source_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        raise DocumentError(f"No usable text was extracted from {source_filename!r}.")

    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        model=embedding_model,
    )
    stable_source_id = str(chunks[0].metadata["source_id"])
    from chromadb.errors import ChromaError

    try:
        previous = collection.get(
            where={"source_id": stable_source_id},
            include=["metadatas"],
        )
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        current_ids = set(chunk_ids)
        collection.upsert(
            ids=chunk_ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks],
        )
        stale_ids = [
            chunk_id
            for chunk_id in previous.get("ids", [])
            if chunk_id not in current_ids
        ]
        if stale_ids:
            collection.delete(ids=stale_ids)
    except (ChromaError, OSError, ValueError) as error:
        raise VectorStoreError(
            f"Could not index {source_filename!r} in ChromaDB: {error}"
        ) from error
    return len(chunks)


def ingest_file(
    file_path: str | Path,
    collection: ChromaCollection,
    *,
    embedding_model: EmbeddingModel | None = None,
    source_id: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> int:
    """Extract, chunk, and index one PDF or TXT file from disk."""
    path = Path(file_path)
    pages = extract_document_text(path)
    return index_document(
        pages,
        path.name,
        collection,
        embedding_model=embedding_model,
        source_id=source_id,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def ingest_uploaded_documents(
    documents: Iterable[UploadedDocument],
    collection: ChromaCollection,
    *,
    embedding_model: EmbeddingModel | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, int]:
    """Extract and index a sequence of uploaded PDF/TXT documents."""
    indexed: dict[str, int] = {}
    for document in documents:
        pages = extract_uploaded_document(document.filename, document.content)
        indexed[document.filename] = index_document(
            pages,
            document.filename,
            collection,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    return indexed


def ingest_knowledge_base(
    collection: ChromaCollection,
    knowledge_base_path: str | Path = KNOWLEDGE_BASE_PATH,
    *,
    embedding_model: EmbeddingModel | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, int]:
    """Index all bundled PDF/TXT files, retaining relative paths as source IDs."""
    root = Path(knowledge_base_path)
    if not root.is_dir():
        raise DocumentError(f"Knowledge-base directory does not exist: {root}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise DocumentError(
            f"No supported PDF or TXT documents were found in {root}."
        )

    indexed: dict[str, int] = {}
    for path in paths:
        relative_source_id = path.relative_to(root).as_posix().casefold()
        indexed[path.name] = ingest_file(
            path,
            collection,
            embedding_model=embedding_model,
            source_id=relative_source_id,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    return indexed


def retrieve_documents(
    collection: ChromaCollection,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    embedding_model: EmbeddingModel | None = None,
    max_distance: float | None = DEFAULT_MAX_COSINE_DISTANCE,
) -> list[RetrievedChunk]:
    """Retrieve relevant passages with source metadata and cosine similarity."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be non-negative or None")

    query_embedding = embed_texts([query], model=embedding_model)
    from chromadb.errors import ChromaError

    try:
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except (ChromaError, OSError, ValueError) as error:
        raise VectorStoreError(
            f"ChromaDB similarity retrieval failed: {error}"
        ) from error

    documents = (result.get("documents") or [[]])[0] or []
    metadatas = (result.get("metadatas") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []
    retrieved: list[RetrievedChunk] = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        numeric_distance = float(distance) if distance is not None else None
        if (
            max_distance is not None
            and numeric_distance is not None
            and numeric_distance > max_distance
        ):
            continue

        page = metadata.get("page_number")
        retrieved.append(
            RetrievedChunk(
                text=text,
                source_filename=str(metadata.get("source_filename", "Unknown source")),
                page_number=int(page) if page is not None else None,
                distance=numeric_distance,
                similarity=(
                    1.0 - numeric_distance if numeric_distance is not None else None
                ),
            )
        )
    return retrieved
