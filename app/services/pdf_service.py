
import re
from uuid import uuid4

import pymupdf

from app.schemas import Paragraph, StoredDocument


def extract_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> StoredDocument:
    """Extract text blocks and metadata from a PDF.

    Args:
        pdf_bytes (bytes): Binary content of the uploaded PDF.
        filename (str): Original name of the PDF file.

    Returns:
        StoredDocument: Document metadata and extracted text.

    Raises:
        ValueError: If the PDF is invalid, encrypted,
            empty, or contains no extractable text.
    """
    paragraphs: list[Paragraph] = []

    try:
        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as document:
            if document.needs_pass:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            page_count = len(document)

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                blocks = page.get_text(
                    "blocks",
                    sort=True,
                )

                for block in blocks:
                    block_type = block[6]

                    if block_type != 0:
                        continue

                    raw_text = str(block[4])

                    sections = re.split(
                        r"\n\s*\n",
                        raw_text,
                    )

                    for section in sections:
                        text = " ".join(section.split())

                        if not text:
                            continue

                        paragraphs.append(
                            Paragraph(
                                page_number=page_number,
                                text=text,
                            )
                        )

    except pymupdf.FileDataError as exc:
        raise ValueError(
            "The uploaded PDF is invalid or damaged."
        ) from exc

    if not paragraphs:
        raise ValueError(
            "No extractable text was found in the PDF."
        )

    return StoredDocument(
        document_id=uuid4(),
        filename=filename,
        page_count=page_count,
        paragraphs=paragraphs,
    )
