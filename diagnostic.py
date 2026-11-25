import pymupdf as fitz
from pathlib import Path
for p in sorted(Path("docs(origin)").glob("*.pdf")):
    doc = fitz.open(p)
    chars = sum(len(pg.get_text("text").strip()) for pg in doc)
    print(f"{p.name}: страниц {len(doc)}, символов текста {chars}")