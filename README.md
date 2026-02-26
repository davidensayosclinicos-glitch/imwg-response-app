# IMWG Response Calculator (Streamlit)

Aplicación en Streamlit para evaluar respuesta IMWG en mieloma múltiple.

## Ejecutar en local

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecuta la app:

```bash
streamlit run imwg_response_app.py
```

## Publicar gratis en Streamlit Community Cloud

1. Sube estos archivos a un repositorio de GitHub:
   - `imwg_response_app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`

2. Entra a Streamlit Community Cloud:
   - https://share.streamlit.io

3. Crea una app nueva (`New app`) y selecciona:
   - Repo: tu repositorio
   - Branch: la rama principal
   - Main file path: `imwg_response_app.py`

4. Presiona `Deploy`.

## Importante sobre persistencia de datos

En esta versión, los datos de pacientes se guardan en carpeta local (`data_pacientes`).
En Streamlit Cloud el almacenamiento local no es persistente a largo plazo.

Si necesitas persistencia real, migra el guardado a una base de datos (por ejemplo: SQLite gestionado externamente, Supabase, Firebase o similar).
