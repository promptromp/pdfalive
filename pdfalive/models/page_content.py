"""Page content data model."""

from pydantic import BaseModel


class PageContent(BaseModel):
    page_num: int
    text: str
    has_native_text: bool
    image: bytes | None  # For OCR fallback
    font_spans: list[dict]  # Font metadata for each text span
    layout_bbox: list[tuple]  # Bounding boxes
