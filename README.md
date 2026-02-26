# Haciendo leer un libro a tu computadora (PLN) — Parte 1 y 2

Proyecto mínimo para **cargar un libro en `.txt`**, aplicar **limpieza** (normalización + lematización) en español y luego hacer **vectorización** (BoW y TF‑IDF).

## Contenido requerido (incluido)

- `data/libro.txt`: texto a procesar (fragmento de dominio público).
- `data/principito.txt` (opcional): si lo agregas, se usará como entrada por defecto para vectorización.
- `src/limpieza_libro.py`: desarrollo de limpieza (normalización + lematización).
- `src/vectorizacion_texto.py`: vectorización (BoW/TF‑IDF + n‑gramas) desde terminal.
- `limpieza_libro.ipynb`: notebook con el mismo flujo (para entregar como `.ipynb`).
- `requirements.txt`: dependencias congeladas.

## Cómo ejecutar

Crear entorno virtual e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ejecutar el script:

```bash
python src/limpieza_libro.py
```

Vectorización (BoW / TF‑IDF):

```bash
python src/vectorizacion_texto.py
```

Opcional: especificar un archivo de entrada:

```bash
python src/vectorizacion_texto.py --input data/principito.txt
```

Salidas:

- `outputs/libro_lemmas.txt`: 1 lema por línea
- `outputs/libro_normalizado.txt`: texto normalizado (lemmas en una sola línea)
- `outputs/top_30_frecuencias.txt`: conteos de frecuencia
- `outputs/X_bow.npz`: matriz BoW (sparse)
- `outputs/X_tfidf.npz`: matriz TF‑IDF (sparse)
- `outputs/vocab_bow.txt`: vocabulario BoW
- `outputs/vocab_tfidf.txt`: vocabulario TF‑IDF
- `outputs/vectorizacion_meta.json`: metadatos del proceso

## Usar tu propio libro

Reemplaza el contenido de `data/libro.txt` por tu libro en texto plano (UTF-8).

Si tu entrega requiere *El Principito*, agrega el archivo como `data/principito.txt` (UTF‑8) o reemplaza `data/libro.txt`.

## Subir a GitHub (comandos)

```bash
git add .
git commit -m "Proyecto PLN: limpieza y lematización de libro"
```

Luego crea un repo **público** en GitHub y enlázalo como remoto:

```bash
git branch -M main
git remote add origin <URL_DE_TU_REPO>
git push -u origin main
```

