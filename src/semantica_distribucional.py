from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "principito.txt"
FALLBACK_INPUT = ROOT / "data" / "libro.txt"

DEFAULT_FIGURES_DIR = ROOT / "figures"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_PROCESSED_TEXT = ROOT / "data" / "texto_procesado.txt"

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


def resolve_input_path(p: Path | None) -> Path:
    if p is not None:
        return p
    return DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT


def lemmatized_sentences(doc) -> list[list[str]]:
    sents = list(doc.sents)
    if not sents:
        sents = [doc]

    corpus: list[list[str]] = []
    for sent in sents:
        toks: list[str] = []
        for token in sent:
            if token.is_space or token.is_punct or token.like_num:
                continue
            if token.is_stop:
                continue
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower().strip()
            if lemma:
                toks.append(lemma)
        if toks:
            corpus.append(toks)
    return corpus


def pick_top_words(freq: Counter[str], allowed: set[str], k: int) -> list[str]:
    out: list[str] = []
    for w, _c in freq.most_common():
        if w in allowed:
            out.append(w)
        if len(out) >= k:
            break
    return out


def plot_embedding_2d(coords: np.ndarray, words: list[str], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))
    plt.scatter(coords[:, 0], coords[:, 1], s=35, alpha=0.85, edgecolors="k", linewidths=0.3)
    for i, w in enumerate(words):
        plt.text(coords[i, 0], coords[i, 1], w, fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semántica distribucional: entrenar Word2Vec y graficar el espacio vectorial."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Ruta al libro en texto plano (UTF-8). Default: data/principito.txt si existe; si no, data/libro.txt.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Carpeta para imágenes PNG (versionable). Default: figures/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta para artefactos (no versionada por defecto). Default: outputs/",
    )
    parser.add_argument(
        "--processed-text",
        type=Path,
        default=DEFAULT_PROCESSED_TEXT,
        help="Archivo TXT con el texto procesado (versionable). Default: data/texto_procesado.txt",
    )
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=60, help="Cantidad de palabras a graficar.")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    raw = normalize_whitespace(input_path.read_text(encoding="utf-8"))

    nlp = load_spanish_model()
    if not any(p in nlp.pipe_names for p in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    doc = nlp(raw)

    corpus = lemmatized_sentences(doc)
    if not corpus:
        raise ValueError("El corpus quedó vacío después de limpiar/lematizar.")

    # Guardar texto procesado (una oración por línea)
    args.processed_text.parent.mkdir(parents=True, exist_ok=True)
    args.processed_text.write_text("\n".join(" ".join(s) for s in corpus) + "\n", encoding="utf-8")

    # Entrenar Word2Vec (Skip-gram)
    w2v = Word2Vec(
        sentences=corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sg=1,
        workers=4,
        seed=42,
        epochs=args.epochs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    w2v.save(str(args.output_dir / "word2vec.model"))
    w2v.wv.save(str(args.output_dir / "word2vec.kv"))

    # Seleccionar palabras frecuentes para graficar
    freq = Counter(w for sent in corpus for w in sent)
    vocab = set(w2v.wv.index_to_key)
    words = pick_top_words(freq, vocab, args.top_k)
    vectors = np.vstack([w2v.wv[w] for w in words])

    # 1) PCA 2D
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(vectors)
    plot_embedding_2d(
        coords_pca,
        words,
        "Word2Vec (Skip-gram) — PCA 2D",
        args.figures_dir / "word2vec_pca_2d.png",
    )

    # 2) t-SNE 2D
    perplexity = min(30, max(5, (len(words) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    coords_tsne = tsne.fit_transform(vectors)
    plot_embedding_2d(
        coords_tsne,
        words,
        f"Word2Vec (Skip-gram) — t-SNE 2D (perplexity={perplexity})",
        args.figures_dir / "word2vec_tsne_2d.png",
    )

    meta = {
        "input_path": str(input_path),
        "n_sentences": len(corpus),
        "tokens_total": sum(len(s) for s in corpus),
        "vocab_size_trained": len(vocab),
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "top_k_plotted": len(words),
        "spacy_model": SPACY_MODEL,
        "figures": [
            str(args.figures_dir / "word2vec_pca_2d.png"),
            str(args.figures_dir / "word2vec_tsne_2d.png"),
        ],
        "processed_text": str(args.processed_text),
    }
    (args.output_dir / "word2vec_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== Semántica distribucional (Word2Vec) ===")
    print(f"Entrada:          {input_path}")
    print(f"Oraciones:        {len(corpus):,}")
    print(f"Vocab entrenado:  {len(vocab):,}")
    print(f"Texto procesado:  {args.processed_text}")
    print(f"Figuras:          {args.figures_dir}")
    print(f"Artefactos:       {args.output_dir}")


if __name__ == "__main__":
    main()

