import fitz  # PyMuPDF
import docx
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ParsedBlock:
    text: str
    locator: str  # "page 1", "paragraph 5" и т.д.

def parse_document(path: Path) -> list[ParsedBlock]:
    """Извлекает текст из файла и разбивает его на логические блоки."""
    ext = path.suffix.lower()
    blocks = []

    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        # Делим по пустым строкам, как было в вашем чанкере
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, p in enumerate(paragraphs):
            blocks.append(ParsedBlock(text=p, locator=f"text block {i+1}"))

    elif ext == ".pdf":
        try:
            doc = fitz.open(path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if text:
                    blocks.append(ParsedBlock(text=text, locator=f"page {page_num + 1}"))
            if not blocks:
                raise ValueError("PDF не содержит текстового слоя (возможно, это скан).")
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения PDF {path.name}: {e}")

    elif ext == ".docx":
        try:
            doc = docx.Document(path)
            for i, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                if text:
                    blocks.append(ParsedBlock(text=text, locator=f"paragraph {i+1}"))
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения DOCX {path.name}: {e}")
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    return blocks