from pathlib import Path

path = Path("docs/reglament.txt")
text = path.read_text(encoding="utf-8")

lines = text.splitlines()
print(f"Строк в файле: {len(lines)}")

paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
print(f"Абзацев: {len(paragraphs)}")
for p in paragraphs[:5]:
    print("-", p[:150], "...")

def chunk_text(text: str, size: int = 300, overlap: int = 100) -> list[str]:
    """Режет текст на куски по size символов с перекрытием overlap."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]

chunks = chunk_text(text)
print(f"Чанков: {len(chunks)}, размер первого: {len(chunks[0])}")