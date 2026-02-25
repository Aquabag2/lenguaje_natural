# Haciendo leer un libro a tu computadora (PLN) — Parte 1

Proyecto mínimo para **cargar un libro en `.txt`** y aplicar **limpieza** (normalización + lematización) en español.

## Contenido requerido (incluido)

- `data/libro.txt`: texto a procesar (fragmento de dominio público).
- `src/limpieza_libro.py`: desarrollo de limpieza (normalización + lematización).
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

Salidas:

- `outputs/libro_lemmas.txt`: 1 lema por línea
- `outputs/libro_normalizado.txt`: texto normalizado (lemmas en una sola línea)
- `outputs/top_30_frecuencias.txt`: conteos de frecuencia

## Usar tu propio libro

Reemplaza el contenido de `data/libro.txt` por tu libro en texto plano (UTF-8).

## Subir a GitHub (comandos)

```bash
git init
git add .
git commit -m "Proyecto PLN: limpieza y lematización de libro"
```

Luego crea un repo **público** en GitHub y enlázalo como remoto:

```bash
git branch -M main
git remote add origin <URL_DE_TU_REPO>
git push -u origin main
```

