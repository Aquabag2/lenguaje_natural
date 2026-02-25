from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import spacy


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "libro.txt"
OUTPUT_DIR = ROOT / "outputs"

SPACY_MODEL = "es_core_news_sm"


def load_spanish_model():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        from spacy.cli import download

        download(SPACY_MODEL)
        return spacy.load(SPACY_MODEL)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def lemmatize(
    doc,
    *,
    keep_pos: Iterable[str] | None = None,
) -> list[str]:
    lemmas: list[str] = []

    for token in doc:
        if token.is_space or token.is_punct or token.like_num:
            continue
        if token.is_stop:
            continue
        if not token.is_alpha:
            continue
        if keep_pos is not None and token.pos_ not in keep_pos:
            continue

        lemma = token.lemma_.lower().strip()
        if not lemma:
            continue
        lemmas.append(lemma)

    return lemmas


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = DATA_PATH.read_text(encoding="utf-8")
    raw = normalize_whitespace(raw)

    nlp = load_spanish_model()
    doc = nlp(raw)

    lemmas = lemmatize(doc)
    normalized_text = " ".join(lemmas)

    (OUTPUT_DIR / "libro_lemmas.txt").write_text("\n".join(lemmas) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "libro_normalizado.txt").write_text(normalized_text + "\n", encoding="utf-8")

    freq = Counter(lemmas)
    top_30 = freq.most_common(30)
    (OUTPUT_DIR / "top_30_frecuencias.txt").write_text(
        "\n".join([f"{w}\t{c}" for w, c in top_30]) + "\n",
        encoding="utf-8",
    )

    print("=== Limpieza de texto (normalización + lematización) ===")
    print(f"Entrada:   {DATA_PATH}")
    print(f"Chars:     {len(raw):,}")
    print(f"Tokens:    {len(doc):,}")
    print(f"Lemas:     {len(lemmas):,}")
    print(f"Únicos:    {len(freq):,}")
    print(f"Outputs:   {OUTPUT_DIR}")
    print("Top 10:", top_30[:10])


if __name__ == "__main__":
    main()

