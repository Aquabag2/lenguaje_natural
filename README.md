## Haciendo leer tu libro favorito a tu computadora (PLN) — Pt. 1, 2 y 3

Proyecto de PLN para:
- **Carga** de un libro en texto plano
- **Limpieza** (normalización + lematización)
- **Vectorización**:
  - **BoW** (CountVectorizer) y **TF‑IDF** (TfidfVectorizer)
  - **Semántica distribucional** (Word2Vec)

## Contenido requerido (incluido)

- `data/libro.txt`: texto a procesar (placeholder).
- `data/principito.txt` (opcional): si lo agregas, se usa por defecto.
- `data/texto_procesado.txt`: texto procesado (lematizado por oraciones) generado en Pt. 3.
- `src/limpieza_libro.py`: limpieza (normalización + lematización).
- `src/vectorizacion_texto.py`: vectorización BoW/TF‑IDF (n‑gramas).
- `src/semantica_distribucional.py`: Word2Vec + generación de imágenes `.png`.
- `limpieza_libro.ipynb`: notebook con el flujo completo.
- `requirements.txt`: dependencias.
- `figures/`: imágenes `.png` de los espacios vectoriales (mínimo 2).

## Cómo ejecutar

Instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Limpieza:

```bash
python src/limpieza_libro.py
```

Vectorización (BoW / TF‑IDF):

```bash
python src/vectorizacion_texto.py
```

Semántica distribucional (Word2Vec) + figuras:

```bash
python src/semantica_distribucional.py
```

Genera (mínimo):
- `figures/word2vec_pca_2d.png`
- `figures/word2vec_tsne_2d.png`
- `data/texto_procesado.txt`

## Usar tu propio libro

Para entregar *El Principito*, pega el texto en **UTF‑8** en `data/principito.txt` (o reemplaza `data/libro.txt`).

