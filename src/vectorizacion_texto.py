from __future__ import annotations

import argparse
import json
from pathlib import Path

import spacy
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "principito.txt"
FALLBACK_INPUT = ROOT / "data" / "libro.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

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


def build_lemmatized_sentence_corpus(doc) -> list[str]:
    corpus: list[str] = []

    for sent in doc.sents:
        lemmas: list[str] = []
        for token in sent:
            if token.is_space or token.is_punct or token.like_num:
                continue
            if token.is_stop:
                continue
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower().strip()
            if lemma:
                lemmas.append(lemma)
        if lemmas:
            corpus.append(" ".join(lemmas))

    return corpus


def resolve_input_path(p: Path | None) -> Path:
    if p is not None:
        return p
    return DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT


def save_vectorization_artifacts(
    *,
    output_dir: Path,
    bow: sparse.spmatrix,
    tfidf: sparse.spmatrix,
    bow_vectorizer: CountVectorizer,
    tfidf_vectorizer: TfidfVectorizer,
    input_path: Path,
    n_sentences: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(output_dir / "X_bow.npz", bow)
    sparse.save_npz(output_dir / "X_tfidf.npz", tfidf)

    vocab_bow = bow_vectorizer.get_feature_names_out().tolist()
    vocab_tfidf = tfidf_vectorizer.get_feature_names_out().tolist()
    (output_dir / "vocab_bow.txt").write_text("\n".join(vocab_bow) + "\n", encoding="utf-8")
    (output_dir / "vocab_tfidf.txt").write_text("\n".join(vocab_tfidf) + "\n", encoding="utf-8")

    meta = {
        "input_path": str(input_path),
        "n_sentences": n_sentences,
        "bow_shape": list(bow.shape),
        "tfidf_shape": list(tfidf.shape),
        "bow_vocab_size": len(vocab_bow),
        "tfidf_vocab_size": len(vocab_tfidf),
        "spacy_model": SPACY_MODEL,
    }
    (output_dir / "vectorizacion_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorización de texto (BoW y TF-IDF) para PLN.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Ruta a tu libro en texto plano (UTF-8). Por defecto usa data/principito.txt si existe; si no, data/libro.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio de salida (default: outputs/).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Máximo n-grama para vectorización (default: 2 => unigramas + bigramas).",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    raw = normalize_whitespace(input_path.read_text(encoding="utf-8"))
    nlp = load_spanish_model()
    if not any(p in nlp.pipe_names for p in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    doc = nlp(raw)

    corpus = build_lemmatized_sentence_corpus(doc)
    if not corpus:
        raise ValueError("No se pudo construir el corpus por oraciones (quedó vacío).")

    bow_vectorizer = CountVectorizer(ngram_range=(1, max(1, args.ngram_max)))
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, max(1, args.ngram_max)))

    X_bow = bow_vectorizer.fit_transform(corpus)
    X_tfidf = tfidf_vectorizer.fit_transform(corpus)

    save_vectorization_artifacts(
        output_dir=args.output_dir,
        bow=X_bow,
        tfidf=X_tfidf,
        bow_vectorizer=bow_vectorizer,
        tfidf_vectorizer=tfidf_vectorizer,
        input_path=input_path,
        n_sentences=len(corpus),
    )

    print("=== Vectorización de texto (BoW / TF-IDF) ===")
    print(f"Entrada:      {input_path}")
    print(f"Oraciones:    {len(corpus):,}")
    print(f"BoW shape:    {tuple(X_bow.shape)}")
    print(f"TF-IDF shape: {tuple(X_tfidf.shape)}")
    print(f"Salida:       {args.output_dir}")


if __name__ == "__main__":
    main()

