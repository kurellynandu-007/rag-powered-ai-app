from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from rag_pipeline import (
    DocumentError,
    ExtractedPage,
    UnsupportedFileTypeError,
    UploadedDocument,
    chunk_text,
    create_document_chunks,
    create_vector_collection,
    extract_document_text,
    extract_uploaded_document,
    ingest_knowledge_base,
    index_document,
    ingest_uploaded_documents,
    retrieve_documents,
)


class FixedEmbeddingModel:
    def encode(self, sentences, *, convert_to_numpy=True, normalize_embeddings=True):
        assert convert_to_numpy
        assert normalize_embeddings
        return [[1.0, 0.0] for _ in sentences]


def _text_pdf(text: str) -> bytes:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def test_extract_document_text_reads_utf8_txt(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("MileMate café sample.", encoding="utf-8")

    assert extract_document_text(source) == [
        ExtractedPage(text="MileMate café sample.")
    ]


def test_extract_uploaded_document_reads_pdf_text_with_page_number() -> None:
    pages = extract_uploaded_document("sample.pdf", _text_pdf("PDF demo text"))

    assert len(pages) == 1
    assert "PDF demo text" in pages[0].text
    assert pages[0].page_number == 1


def test_extract_uploaded_document_rejects_unsupported_type() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="Supported types: .pdf, .txt"):
        extract_uploaded_document("sample.docx", b"not supported")


def test_extract_uploaded_document_reports_invalid_utf8() -> None:
    with pytest.raises(DocumentError, match="UTF-8"):
        extract_uploaded_document("sample.txt", b"\xff")


def test_extract_uploaded_document_explains_scanned_pdf_has_no_ocr() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    stream = BytesIO()
    writer.write(stream)

    with pytest.raises(DocumentError, match="does not provide OCR"):
        extract_uploaded_document("scanned.pdf", stream.getvalue())


def test_extract_document_text_reports_unreadable_file(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a PDF")

    with pytest.raises(DocumentError, match="Could not read PDF"):
        extract_document_text(source)


def test_chunk_text_uses_requested_approximate_size_and_overlap() -> None:
    text = "word " * 400
    chunks = chunk_text(text, chunk_size=800, overlap=150)

    assert len(chunks) > 1
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_create_document_chunks_preserves_source_and_page_metadata() -> None:
    chunks = create_document_chunks(
        [ExtractedPage("one two three", page_number=3)],
        "sample.pdf",
        source_id="kb/sample.pdf",
        chunk_size=8,
        overlap=2,
    )

    assert chunks
    assert chunks[0].metadata["source_filename"] == "sample.pdf"
    assert chunks[0].metadata["page_number"] == 3
    assert chunks[0].metadata["source_id"] == "kb/sample.pdf"
    assert create_document_chunks(
        [ExtractedPage("one two three", page_number=3)],
        "sample.pdf",
        source_id="kb/sample.pdf",
        chunk_size=8,
        overlap=2,
    )[0].chunk_id == chunks[0].chunk_id


def test_ingestion_reindexes_document_without_stale_chunks(tmp_path: Path) -> None:
    collection = create_vector_collection(
        collection_name="reindex-test",
        chroma_db_path=tmp_path / "chroma",
    )
    model = FixedEmbeddingModel()
    source_id = "uploaded:routes.txt"

    first_count = index_document(
        [ExtractedPage("old paragraph. " * 100)],
        "routes.txt",
        collection,
        embedding_model=model,
        source_id=source_id,
    )
    second_count = index_document(
        [ExtractedPage("updated route information.")],
        "routes.txt",
        collection,
        embedding_model=model,
        source_id=source_id,
    )

    stored = collection.get(
        where={"source_id": source_id},
        include=["documents", "metadatas"],
    )
    assert first_count > second_count
    assert len(stored["ids"]) == second_count == 1
    assert stored["documents"] == ["updated route information."]
    assert stored["metadatas"][0]["source_filename"] == "routes.txt"


def test_ingest_uploaded_documents_indexes_text_and_retrieves_it(tmp_path: Path) -> None:
    collection = create_vector_collection(
        collection_name="upload-test",
        chroma_db_path=tmp_path / "upload-chroma",
    )
    model = FixedEmbeddingModel()
    indexed = ingest_uploaded_documents(
        [UploadedDocument("upload.txt", b"Fictional route A1 is a demo route.")],
        collection,
        embedding_model=model,
    )

    result = retrieve_documents(
        collection,
        "Which route is fictional?",
        embedding_model=model,
    )

    assert indexed == {"upload.txt": 1}
    assert result[0].text == "Fictional route A1 is a demo route."
    assert result[0].source_filename == "upload.txt"
    assert result[0].page_number is None
    assert result[0].distance == pytest.approx(0.0)
    assert result[0].similarity == pytest.approx(1.0)


def test_ingest_knowledge_base_indexes_supported_files(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge"
    knowledge_base.mkdir()
    (knowledge_base / "demo.txt").write_text(
        "This is fictional demo knowledge.", encoding="utf-8"
    )
    (knowledge_base / "ignored.md").write_text("not indexed", encoding="utf-8")
    collection = create_vector_collection(
        collection_name="knowledge-test",
        chroma_db_path=tmp_path / "knowledge-chroma",
    )

    indexed = ingest_knowledge_base(
        collection,
        knowledge_base,
        embedding_model=FixedEmbeddingModel(),
    )

    assert indexed == {"demo.txt": 1}
    stored = collection.get(
        where={"source_id": "demo.txt"},
        include=["documents", "metadatas"],
    )
    assert stored["documents"] == ["This is fictional demo knowledge."]
