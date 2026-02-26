# imwg_response_app.py
# Streamlit app to calculate IMWG response category over time
# Usage: streamlit run imwg_response_app.py

import re
import pandas as pd
import numpy as np
import streamlit as st
import json
import os
import webbrowser
import tempfile
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="IMWG Response Calculator", layout="wide")

STANDARD_COLUMNS = [
    "Fecha", "Timepoint", "Inmunoglobulinas", "Proteina_M_suero", "IFE_suero", "IFE_orina",
    "Cadenas_kappa", "Cadenas_lambda", "FLC_ratio", "Proteina_orina", "BMPC_pct",
    "Citometria_MO", "Cambio_tamano_pct", "Evaluacion_respuesta_IMWG", "Iniciales_investigador"
]

# Inicializar session state
if "mostrar_impresion" not in st.session_state:
    st.session_state.mostrar_impresion = False

st.title("📊 Calculadora de Respuesta IMWG (Mieloma Múltiple)")
st.caption("Calcula la categoría de respuesta según criterios IMWG adaptados a cada tipo de enfermedad")

# ----------------------------
# Data persistence functions
# ----------------------------
DATA_DIR = Path("data_pacientes")
DATA_DIR.mkdir(exist_ok=True)

def normalize_dataframe_schema(dataframe):
    """Asegura columnas estándar y compatibilidad con versiones antiguas."""
    df = dataframe.copy()

    # Compatibilidad con formato antiguo
    if "FLC_ratio" not in df.columns and "FLC" in df.columns:
        df["FLC_ratio"] = df["FLC"]

    # Eliminar columna legacy para unificar esquema
    if "FLC" in df.columns:
        df = df.drop(columns=["FLC"])

    # Agregar columnas faltantes con valor vacío
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Mantener columnas extra al final para no perder información
    extra_cols = [c for c in df.columns if c not in STANDARD_COLUMNS]
    ordered_cols = STANDARD_COLUMNS + extra_cols
    return df[ordered_cols]

def guardar_datos_paciente(codigo_paciente, dataframe, criterio):
    """Guarda los datos de un paciente en archivo JSON"""
    filepath = DATA_DIR / f"paciente_{codigo_paciente}.json"
    normalized_df = normalize_dataframe_schema(dataframe)
    data = normalized_df.to_dict(orient="records")
    metadata = {"criterio": criterio}
    
    save_data = {"metadata": metadata, "datos": data}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    return filepath

def cargar_datos_paciente(codigo_paciente):
    """Carga los datos de un paciente desde archivo JSON"""
    filepath = DATA_DIR / f"paciente_{codigo_paciente}.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            save_data = json.load(f)
        
        # Compatibilidad con formato antiguo
        if isinstance(save_data, list):
            return normalize_dataframe_schema(pd.DataFrame(save_data)), None
        
        metadata = save_data.get("metadata", {})
        criterio = metadata.get("criterio", "1")
        return normalize_dataframe_schema(pd.DataFrame(save_data.get("datos", []))), criterio
    return None, None

# ----------------------------
# Helper functions
# ----------------------------
def to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    if s == "" or s.lower() in {"na","nan","none","null","-"}:
        return np.nan
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def compute_dfLC(kappa, lambda_):
    k = to_float(kappa); l = to_float(lambda_)
    if np.isnan(k) or np.isnan(l):
        return np.nan
    return abs(k - l)

def is_negative(x):
    if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
        return False
    if isinstance(x, bool):
        return not x
    s = str(x).strip().lower()
    return s in {"neg", "negative", "no", "n", "0", "false", "(-)", "-", "not detected", "negativo"}

def is_missing_value(x):
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "na", "nan", "none", "null", "-", "nd"}

# ----------------------------
# Classification functions by criteria type
# ----------------------------
CRITERIOS_INFO = {
    "1": {
        "nombre": "🩸 Proteína M Sérica",
        "descripcion": "Enfermedad medible: proteína M ≥ 1 g/dL"
    },
    "2": {
        "nombre": "🚽 Proteína M Urinaria (Bence-Jones)",
        "descripcion": "Enfermedad medible: ≥ 200 mg/24h"
    },
    "3": {
        "nombre": "🔬 Cadenas Ligeras Libres (FLC)",
        "descripcion": "Enfermedad medible: dFLC ≥ 100 mg/L o Ratio κ/λ anormal"
    },
    "4": {
        "nombre": "🧬 Enfermedad No Secretora",
        "descripcion": "Seguimiento por infiltración de médula ósea + citometría MO"
    },
    "5": {
        "nombre": "🦴 Plasmocitoma / Enfermedad Extramedular",
        "descripcion": "Seguimiento por imagen (PET-CT/RM) usando % cambio de tamaño"
    }
}

CRITERIOS_IMWG_TEXTO = """
### 📚 Algoritmo de clasificación (implementación actual)

La app usa una **evaluación integrada** con los marcadores disponibles en cada fila.

**1) Prioridad absoluta: PD (Progresión)**
- Se evalúa primero contra **nadir** (mejor valor previo) y, si cumple, se clasifica como PD.
- Suero: incremento ≥ 25% + aumento absoluto ≥ 0.5 g/dL (o ≥ 1.0 g/dL si basal sérica ≥ 5 g/dL).
- Orina: incremento ≥ 25% + aumento absoluto ≥ 200 mg/24h.
- dFLC: incremento ≥ 25% + aumento absoluto ≥ 100 mg/L.
- Médula: PD si nadir BMPC < 5% y BMPC actual ≥ 10%.
- Plasmocitoma/EMD: `Cambio_tamano_pct` ≥ 50.

**2) RC / sCR**
- RC: `IFE_suero` negativo + `IFE_orina` negativo + `BMPC_pct` ≤ 5 + desaparición de plasmocitoma si existe.
- sCR: cumple RC + `FLC_ratio` normal (0.26–1.65) + `Citometria_MO` negativa.

**3) VGPR / RP (si no hay PD ni RC/sCR)**
- Si **suero y orina son medibles** (basal suero ≥ 1 y basal orina ≥ 200), la respuesta es combinada:
    - VGPR: suero ↓ ≥ 90% **y** orina < 100 mg/24h.
    - RP: suero ↓ ≥ 50% **y** (orina ↓ ≥ 90% **o** orina < 200 mg/24h).
- Si solo **suero** es medible: VGPR suero ↓ ≥ 90%; RP suero ↓ ≥ 50%.
- Si solo **orina** es medible: VGPR orina < 100; RP orina ↓ ≥ 90% o < 200.
- dFLC clasifica solo si suero/orina no son medibles y con enfermedad FLC medible (dFLC basal ≥ 100 + ratio basal anormal).
- Si no hay marcador sérico/urinario/FLC medible, se usa BMPC (VGPR ↓ ≥ 90%, RP ↓ ≥ 50%).

**4) EE (Enfermedad Estable)**
- Se asigna cuando no cumple PD, RC/sCR, VGPR o RP.

**Nota:** si faltan datos en una fila, la app clasifica con la mejor evidencia disponible.
"""

def classify_integrated_imwg(row, baseline_row, nadir_values):

    # --- Extraer valores ---
    serum = to_float(row.get("Proteina_M_suero", ""))
    urine = to_float(row.get("Proteina_orina", ""))
    bmpc = to_float(row.get("BMPC_pct", ""))
    flc_ratio = to_float(row.get("FLC_ratio", ""))
    dflc = compute_dfLC(row.get("Cadenas_kappa", ""), row.get("Cadenas_lambda", ""))
    size_pct = to_float(row.get("Cambio_tamano_pct", ""))

    base_serum = to_float(baseline_row.get("Proteina_M_suero", ""))
    base_urine = to_float(baseline_row.get("Proteina_orina", ""))
    base_dflc = compute_dfLC(baseline_row.get("Cadenas_kappa", ""), baseline_row.get("Cadenas_lambda", ""))
    base_bmpc = to_float(baseline_row.get("BMPC_pct", ""))
    base_flc_ratio = to_float(baseline_row.get("FLC_ratio", ""))

    nadir_serum = nadir_values.get("serum", np.nan)
    nadir_urine = nadir_values.get("urine", np.nan)
    nadir_dflc = nadir_values.get("dflc", np.nan)
    nadir_bmpc = nadir_values.get("bmpc", np.nan)

    ife_neg = is_negative(row.get("IFE_suero", "")) and is_negative(row.get("IFE_orina", ""))
    cit_neg = is_negative(row.get("Citometria_MO", ""))
    flc_ratio_normal = not np.isnan(flc_ratio) and 0.26 <= flc_ratio <= 1.65
    bmpc_rc = not np.isnan(bmpc) and bmpc <= 5

    serum_measurable = not np.isnan(base_serum) and base_serum >= 1
    urine_measurable = not np.isnan(base_urine) and base_urine >= 200
    base_flc_ratio_abnormal = not np.isnan(base_flc_ratio) and (base_flc_ratio < 0.26 or base_flc_ratio > 1.65)
    flc_measurable = not np.isnan(base_dflc) and base_dflc >= 100 and base_flc_ratio_abnormal

    lesions_known = not np.isnan(size_pct)
    lesions_resolved = (size_pct <= -100) if lesions_known else True

    # --------------------------------------------------
    # 1️⃣ PRIORIDAD ABSOLUTA: PROGRESIÓN (PD)
    # --------------------------------------------------

    # Suero
    if not np.isnan(serum) and not np.isnan(nadir_serum):
        pd_abs_min = 1.0 if (not np.isnan(base_serum) and base_serum >= 5) else 0.5
        if serum >= nadir_serum * 1.25 and (serum - nadir_serum) >= pd_abs_min:
            return "PD"

    # Orina
    if not np.isnan(urine) and not np.isnan(nadir_urine):
        if urine >= nadir_urine * 1.25 and (urine - nadir_urine) >= 200:
            return "PD"

    # dFLC
    if not np.isnan(dflc) and not np.isnan(nadir_dflc):
        if dflc >= nadir_dflc * 1.25 and (dflc - nadir_dflc) >= 100:
            return "PD"

    # Médula
    if not np.isnan(bmpc) and not np.isnan(nadir_bmpc):
        if nadir_bmpc < 5 and bmpc >= 10:
            return "PD"

    # Plasmocitoma / enfermedad extramedular
    if not np.isnan(size_pct) and size_pct >= 50:
        return "PD"

    # --------------------------------------------------
    # 2️⃣ REMISIÓN COMPLETA
    # --------------------------------------------------

    if ife_neg and bmpc_rc and lesions_resolved:
        if flc_ratio_normal and cit_neg:
            return "sCR"
        return "RC"

    # --------------------------------------------------
    # 3️⃣ VGPR / RP SEGÚN TIPO DE ENFERMEDAD
    # --------------------------------------------------

    serum_pct_red = np.nan
    urine_pct_red = np.nan
    dflc_pct_red = np.nan
    bmpc_pct_red = np.nan

    if serum_measurable and not np.isnan(serum):
        serum_pct_red = ((base_serum - serum) / base_serum) * 100

    if urine_measurable and not np.isnan(urine):
        urine_pct_red = ((base_urine - urine) / base_urine) * 100

    if flc_measurable and not np.isnan(dflc) and not np.isnan(base_dflc) and base_dflc > 0:
        dflc_pct_red = ((base_dflc - dflc) / base_dflc) * 100

    if not np.isnan(bmpc) and not np.isnan(base_bmpc) and base_bmpc > 0:
        bmpc_pct_red = ((base_bmpc - bmpc) / base_bmpc) * 100

    # Suero + orina medibles: evaluación combinada
    if serum_measurable and urine_measurable:
        if not np.isnan(serum_pct_red) and not np.isnan(urine):
            if serum_pct_red >= 90 and urine < 100:
                return "VGPR"
        if not np.isnan(serum_pct_red) and (not np.isnan(urine_pct_red) or not np.isnan(urine)):
            urine_rp_ok = (not np.isnan(urine_pct_red) and urine_pct_red >= 90) or (not np.isnan(urine) and urine < 200)
            if serum_pct_red >= 50 and urine_rp_ok:
                return "RP"

    # Solo suero medible
    elif serum_measurable:
        if not np.isnan(serum_pct_red):
            if serum_pct_red >= 90:
                return "VGPR"
            if serum_pct_red >= 50:
                return "RP"

    # Solo orina medible
    elif urine_measurable:
        if not np.isnan(urine) and urine < 100:
            return "VGPR"
        if (not np.isnan(urine_pct_red) and urine_pct_red >= 90) or (not np.isnan(urine) and urine < 200):
            return "RP"

    # FLC solo si suero/orina no medibles
    elif flc_measurable:
        if not np.isnan(dflc_pct_red):
            if dflc_pct_red >= 90:
                return "VGPR"
            if dflc_pct_red >= 50:
                return "RP"

    # No secretora: usar médula cuando no hay marcadores medibles arriba
    else:
        if not np.isnan(bmpc_pct_red):
            if bmpc_pct_red >= 90:
                return "VGPR"
            if bmpc_pct_red >= 50:
                return "RP"

    return "EE"

# ----------------------------
# Template data
# ----------------------------
default_rows = [
    {"Fecha": "", "Timepoint": "Screening", "Inmunoglobulinas": "", 
    "Proteina_M_suero": "", "IFE_suero": "", "IFE_orina": "", "Cadenas_kappa": "", "Cadenas_lambda": "", 
     "FLC_ratio": "", "Proteina_orina": "", "BMPC_pct": "", "Citometria_MO": "", 
     "Cambio_tamano_pct": "", "Evaluacion_respuesta_IMWG": "", "Iniciales_investigador": ""},
    {"Fecha": "", "Timepoint": "C1D1", "Inmunoglobulinas": "", 
    "Proteina_M_suero": "", "IFE_suero": "", "IFE_orina": "", "Cadenas_kappa": "", "Cadenas_lambda": "", 
     "FLC_ratio": "", "Proteina_orina": "", "BMPC_pct": "", "Citometria_MO": "", 
     "Cambio_tamano_pct": "", "Evaluacion_respuesta_IMWG": "", "Iniciales_investigador": ""},
    {"Fecha": "", "Timepoint": "Week 5", "Inmunoglobulinas": "", 
    "Proteina_M_suero": "", "IFE_suero": "", "IFE_orina": "", "Cadenas_kappa": "", "Cadenas_lambda": "", 
     "FLC_ratio": "", "Proteina_orina": "", "BMPC_pct": "", "Citometria_MO": "", 
     "Cambio_tamano_pct": "", "Evaluacion_respuesta_IMWG": "", "Iniciales_investigador": ""},
]

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Configuración")

# Selector de código de paciente
st.sidebar.markdown("**👤 Código de Paciente**")
codigos_disponibles = [f"{i:06d}" for i in range(601, 609)]
codigo_paciente = st.sidebar.selectbox("Seleccionar paciente:", options=codigos_disponibles, key="codigo_paciente")

st.sidebar.divider()

# Cargar datos previos
datos_cargados, criterio_cargado = cargar_datos_paciente(codigo_paciente)

# Selector de criterio de seguimiento
st.sidebar.markdown("**📋 Criterio Basal (informativo + validador)**")
criterio_default = criterio_cargado if criterio_cargado else "1"
criterio_seleccionado = st.sidebar.radio(
    "Selecciona el criterio reportado para este paciente:",
    options=list(CRITERIOS_INFO.keys()),
    format_func=lambda x: f"{CRITERIOS_INFO[x]['nombre']}",
    key=f"criterio_selector_{codigo_paciente}",
    index=int(criterio_default) - 1
)

st.sidebar.divider()

# Nombre del IP (Investigador Principal)
st.sidebar.markdown("**👨‍⚕️ Investigador Principal**")
nombre_ip = st.sidebar.text_input("Nombre del IP:", key="nombre_ip", placeholder="Dr. Juan Pérez")

st.sidebar.divider()
uploaded = st.sidebar.file_uploader(
    "Subir CSV (opcional)",
    type=["csv"],
    key=f"uploaded_csv_{codigo_paciente}",
    help="Este archivo se aplica solo al paciente seleccionado."
)

st.sidebar.divider()
st.sidebar.markdown("**❓ ¿Paciente con EMD?**")
tiene_emd = st.sidebar.radio("", options=["No", "Sí"], horizontal=True, key="emd_radio")
if tiene_emd == "Sí":
    st.sidebar.info("⚠️ Si presenta EMD: Q12W from C1D1 until PD using the same modality as used at screening")

# ----------------------------
# Main
# ----------------------------
st.info(f"📋 **Paciente: {codigo_paciente}** | Criterio: {CRITERIOS_INFO[criterio_seleccionado]['nombre']}")
st.caption(CRITERIOS_INFO[criterio_seleccionado]['descripcion'])

with st.expander("📖 Ver criterios IMWG de respuesta y progresión", expanded=False):
    st.markdown(CRITERIOS_IMWG_TEXTO)

# Cargar datos
if uploaded is not None:
    df = normalize_dataframe_schema(pd.read_csv(uploaded))
elif datos_cargados is not None:
    df = normalize_dataframe_schema(datos_cargados)
    st.success(f"✅ Datos cargados para paciente {codigo_paciente}")
else:
    df = normalize_dataframe_schema(pd.DataFrame(default_rows))

# Calculate responses before displaying
work = df.copy()

# Find baseline (C1D1)
baseline_mask = work["Timepoint"].astype(str).str.contains("C1D1", case=False, na=False)
if baseline_mask.sum() > 0:
    baseline_idx = work.index[baseline_mask].tolist()[0]
    baseline_row = work.loc[baseline_idx]
    baseline_found = True
else:
    baseline_row = work.iloc[0] if len(work) > 0 else pd.Series()
    baseline_found = False

# Show baseline info
if baseline_found:
    baseline_value = baseline_row.get("Proteina_M_suero", "N/A")
    st.info(f"📊 **Baseline (C1D1)**: Toda evaluación se compara contra este punto")

    # Validación de coherencia entre criterio seleccionado y perfil basal
    base_serum = to_float(baseline_row.get("Proteina_M_suero", ""))
    base_urine = to_float(baseline_row.get("Proteina_orina", ""))
    base_dflc = compute_dfLC(baseline_row.get("Cadenas_kappa", ""), baseline_row.get("Cadenas_lambda", ""))
    base_flc_ratio = to_float(baseline_row.get("FLC_ratio", ""))
    base_bmpc = to_float(baseline_row.get("BMPC_pct", ""))
    base_emd = to_float(baseline_row.get("Cambio_tamano_pct", ""))

    serum_measurable = not np.isnan(base_serum) and base_serum >= 1
    urine_measurable = not np.isnan(base_urine) and base_urine >= 200
    flc_ratio_abnormal = not np.isnan(base_flc_ratio) and (base_flc_ratio < 0.26 or base_flc_ratio > 1.65)
    flc_measurable = not np.isnan(base_dflc) and base_dflc >= 100 and flc_ratio_abnormal
    emd_trackable = not np.isnan(base_emd)

    detected_types = []
    if serum_measurable:
        detected_types.append("Proteína M sérica medible")
    if urine_measurable:
        detected_types.append("Proteína M urinaria medible")
    if flc_measurable:
        detected_types.append("Enfermedad medible por dFLC")
    if emd_trackable:
        detected_types.append("Enfermedad extramedular valorable")
    if not (serum_measurable or urine_measurable or flc_measurable):
        detected_types.append("Perfil compatible con no secretora")

    st.caption("🔎 Tipo de enfermedad detectado automáticamente en basal: " + ", ".join(detected_types))

    warnings_criterio = []
    if criterio_seleccionado == "1" and not serum_measurable:
        warnings_criterio.append("Seleccionaste Proteína M sérica, pero en basal no es medible (<1 g/dL o faltante).")
    if criterio_seleccionado == "2" and not urine_measurable:
        warnings_criterio.append("Seleccionaste Proteína M urinaria, pero en basal no es medible (<200 mg/24h o faltante).")
    if criterio_seleccionado == "3" and not flc_measurable:
        warnings_criterio.append("Seleccionaste FLC, pero en basal no cumple medibilidad por dFLC (≥100 mg/L) con ratio anormal.")
    if criterio_seleccionado == "4" and (serum_measurable or urine_measurable or flc_measurable):
        warnings_criterio.append("Seleccionaste No secretora, pero en basal hay al menos un marcador medible (suero/orina/FLC).")
    if criterio_seleccionado == "5" and not emd_trackable:
        warnings_criterio.append("Seleccionaste enfermedad extramedular, pero no hay dato basal de tamaño para seguimiento por imagen.")

    if warnings_criterio:
        st.warning("⚠️ Inconsistencia entre criterio seleccionado y perfil basal detectado:")
        for warning_text in warnings_criterio:
            st.write(f"- {warning_text}")
else:
    st.warning("⚠️ No se encontró C1D1. Las respuestas se calcularán cuando lo ingreses.")

# Calculate response for each row
nadir_values = {
    "serum": to_float(baseline_row.get("Proteina_M_suero", "")) if baseline_found else np.nan,
    "urine": to_float(baseline_row.get("Proteina_orina", "")) if baseline_found else np.nan,
    "dflc": compute_dfLC(baseline_row.get("Cadenas_kappa", ""), baseline_row.get("Cadenas_lambda", "")) if baseline_found else np.nan,
    "bmpc": to_float(baseline_row.get("BMPC_pct", "")) if baseline_found else np.nan,
}

responses = []
for i, row in work.iterrows():
    timepoint = str(row["Timepoint"]).strip().lower()
    
    # No calcular respuesta para Screening o C1D1
    if "screening" in timepoint or "c1d1" in timepoint:
        responses.append("")
    elif not baseline_found:
        responses.append("")
    else:
        resp = classify_integrated_imwg(row, baseline_row, nadir_values)
        responses.append(resp)

        serum_value = to_float(row.get("Proteina_M_suero", ""))
        if not np.isnan(serum_value):
            if np.isnan(nadir_values["serum"]):
                nadir_values["serum"] = serum_value
            else:
                nadir_values["serum"] = min(nadir_values["serum"], serum_value)

        urine_value = to_float(row.get("Proteina_orina", ""))
        if not np.isnan(urine_value):
            if np.isnan(nadir_values["urine"]):
                nadir_values["urine"] = urine_value
            else:
                nadir_values["urine"] = min(nadir_values["urine"], urine_value)

        dflc_value = compute_dfLC(row.get("Cadenas_kappa", ""), row.get("Cadenas_lambda", ""))
        if not np.isnan(dflc_value):
            if np.isnan(nadir_values["dflc"]):
                nadir_values["dflc"] = dflc_value
            else:
                nadir_values["dflc"] = min(nadir_values["dflc"], dflc_value)

        bmpc_value = to_float(row.get("BMPC_pct", ""))
        if not np.isnan(bmpc_value):
            if np.isnan(nadir_values["bmpc"]):
                nadir_values["bmpc"] = bmpc_value
            else:
                nadir_values["bmpc"] = min(nadir_values["bmpc"], bmpc_value)

work["Evaluacion_respuesta_IMWG"] = responses

# Validaciones automáticas para confirmación RC/sCR
faltantes_rc_scr = []
for i, row in work.iterrows():
    timepoint = str(row.get("Timepoint", "")).strip().lower()
    if "screening" in timepoint or "c1d1" in timepoint:
        continue

    campos_faltantes = []
    if is_missing_value(row.get("IFE_suero", "")):
        campos_faltantes.append("IFE_suero")
    if is_missing_value(row.get("IFE_orina", "")):
        campos_faltantes.append("IFE_orina")
    if np.isnan(to_float(row.get("BMPC_pct", ""))):
        campos_faltantes.append("BMPC_pct")
    if np.isnan(to_float(row.get("FLC_ratio", ""))):
        campos_faltantes.append("FLC_ratio")
    if is_missing_value(row.get("Citometria_MO", "")):
        campos_faltantes.append("Citometria_MO")

    if campos_faltantes:
        faltantes_rc_scr.append({
            "Fila": i + 1,
            "Timepoint": row.get("Timepoint", ""),
            "Datos faltantes para RC/sCR": ", ".join(campos_faltantes)
        })

if faltantes_rc_scr:
    st.warning("⚠️ Faltan datos clave para confirmar RC/sCR en una o más evaluaciones.")
    st.dataframe(pd.DataFrame(faltantes_rc_scr), width="stretch", hide_index=True)
    st.caption("Para confirmar RC/sCR, completa al menos: IFE_suero, IFE_orina, BMPC_pct, FLC_ratio y Citometria_MO.")

# Color mapping for response types
response_colors = {
    "sCR": "#6EE7B7",     # Verde menta
    "RC": "#90EE90",      # Verde claro
    "VGPR": "#87CEEB",    # Azul cielo
    "RP": "#FFD700",      # Oro
    "EE": "#D3D3D3",      # Gris claro
    "PD": "#FF6B6B",      # Rojo
    "": ""
}

def highlight_response(row):
    val = row["Evaluacion_respuesta_IMWG"]
    color = response_colors.get(val, "")
    return ['background-color: {}'.format(color) if color else '' for _ in row]

st.subheader("📝 Tabla de Evaluación IMWG")
st.caption("✏️ Edita los datos y presiona el botón para recalcular automáticamente.")

st.markdown("### 🧭 Leyenda de carga de datos")
st.markdown(
        """
- **Proteina_M_suero**: numérico en g/dL. Ej: `3.2`
- **IFE_suero**: texto categórico. Usar preferentemente: `Negativo` / `Positivo`
- **IFE_orina**: texto categórico. Usar preferentemente: `Negativo` / `Positivo`
- **Cadenas_kappa / Cadenas_lambda**: numérico en mg/L. Ej: `245`, `32.5`
- **FLC_ratio**: numérico decimal. Ej: `0.85`, `2.1`
- **Proteina_orina**: numérico en mg/24h. Ej: `120`
- **BMPC_pct**: porcentaje de infiltración medular (solo número, sin `%`). Ej: `18`, `4.5`
- **Citometria_MO**: texto categórico. Usar preferentemente: `Negativo` / `Positivo`
- **Cambio_tamano_pct**: porcentaje de cambio por imagen (solo número, sin `%`):
    - reducción: valor negativo (ej: `-60`)
    - crecimiento: valor positivo (ej: `+30`)
    - desaparición completa: `-100`
"""
)

timepoint_options = [
    "Screening", "C1D1", "Week 5", "Week 9", "Week 13", "Week 17", "Week 21",
    "Week 25", "Week 29", "Week 33", "Week 37", "Week 41", "Week 45", "Week 49"
]

inmunoglobulina_options = ["IgG", "IgA", "IgM", "Bence-Jones", "No secretora", "Otro"]
estado_options = ["Negativo", "Positivo", "ND"]

# Display single editable table
edited = st.data_editor(
    work, 
    num_rows="dynamic", 
    width="stretch", 
    height=500,
    column_config={
        "Timepoint": st.column_config.SelectboxColumn(
            "Timepoint",
            options=timepoint_options,
            required=False,
            help="Selecciona el punto temporal de evaluación"
        ),
        "Inmunoglobulinas": st.column_config.SelectboxColumn(
            "Inmunoglobulinas",
            options=inmunoglobulina_options,
            required=False,
            help="Tipo de inmunoglobulina monoclonal"
        ),
        "IFE_suero": st.column_config.SelectboxColumn(
            "IFE_suero",
            options=estado_options,
            required=False,
            help="Resultado de inmunofijación en suero"
        ),
        "IFE_orina": st.column_config.SelectboxColumn(
            "IFE_orina",
            options=estado_options,
            required=False,
            help="Resultado de inmunofijación en orina"
        ),
        "Citometria_MO": st.column_config.SelectboxColumn(
            "Citometria_MO",
            options=estado_options,
            required=False,
            help="Resultado de citometría de médula ósea"
        ),
        "Evaluacion_respuesta_IMWG": st.column_config.TextColumn(
            "Evaluacion_respuesta_IMWG",
            disabled=True,
            help="Campo calculado automáticamente"
        )
    },
    key=f"data_editor_{codigo_paciente}"
)

# Action buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Recalcular respuestas", type="primary"):
        st.rerun()

with col2:
    if st.button("💾 Guardar datos del paciente", type="secondary"):
        try:
            guardar_datos_paciente(codigo_paciente, edited, criterio_seleccionado)
            st.success(f"✅ Datos guardados para paciente {codigo_paciente} (Criterio: {CRITERIOS_INFO[criterio_seleccionado]['nombre']})")
        except Exception as e:
            st.error(f"❌ Error al guardar: {e}")

with col3:
    csv_data = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV",
        data=csv_data,
        file_name=f"paciente_{codigo_paciente}_imwg.csv",
        mime="text/csv"
    )

with col4:
    if st.button("  Generar Informe PDF", type="secondary"):
        st.session_state.mostrar_impresion = True

# Mostrar ventana de impresión si está activa
if st.session_state.get("mostrar_impresion", False):
    with st.container():
        st.markdown("---")
        st.markdown("### 📄 Informe de Evaluación IMWG")
        
        # Generar HTML profesional para el informe
        html_content = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="utf-8">
            <title>Informe IMWG - Paciente {codigo_paciente}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                html, body {{
                    font-family: 'Arial', 'Segoe UI', sans-serif;
                    line-height: 1.6;
                    color: #1f2937;
                    background: white;
                }}
                body {{
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #059669;
                    padding-bottom: 20px;
                    margin-bottom: 25px;
                }}
                .header h1 {{
                    font-size: 24px;
                    color: #059669;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .header p {{
                    font-size: 12px;
                    color: #666;
                    margin: 3px 0;
                }}
                .info-section {{
                    margin-bottom: 20px;
                    padding: 12px 15px;
                    background: #ecfdf5;
                    border-left: 4px solid #059669;
                    border-radius: 3px;
                }}
                .info-row {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 15px;
                    margin-bottom: 8px;
                }}
                .info-item {{
                    font-size: 11px;
                }}
                .info-label {{
                    color: #059669;
                    font-weight: bold;
                    display: inline-block;
                    width: 140px;
                }}
                .info-value {{
                    color: #1f2937;
                    font-weight: normal;
                }}
                .criteria-box {{
                    background: #fffaf0;
                    border-left: 4px solid #f59e0b;
                    padding: 10px 12px;
                    margin-bottom: 20px;
                    border-radius: 3px;
                    font-size: 11px;
                }}
                .criteria-box strong {{
                    color: #d97706;
                }}
                h2 {{
                    font-size: 13px;
                    color: #059669;
                    margin-top: 20px;
                    margin-bottom: 10px;
                    padding-bottom: 5px;
                    border-bottom: 2px solid #dbeafe;
                    font-weight: bold;
                }}
                .table-wrapper {{
                    overflow-x: auto;
                    margin: 10px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 10px;
                    background: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                thead {{
                    background-color: #059669;
                    color: white;
                }}
                th {{
                    padding: 8px 6px;
                    text-align: left;
                    font-weight: bold;
                    border: 1px solid #059669;
                    white-space: nowrap;
                }}
                td {{
                    padding: 6px;
                    border: 1px solid #d1d5db;
                    background: white;
                    word-break: break-word;
                }}
                tbody tr:nth-child(odd) {{
                    background-color: #f9fafb;
                }}
                tbody tr:nth-child(even) {{
                    background-color: #ffffff;
                }}
                .legend {{
                    background: #f3f4f6;
                    padding: 10px 12px;
                    margin: 15px 0;
                    font-size: 10px;
                    border-left: 4px solid #059669;
                    border-radius: 3px;
                    line-height: 1.6;
                }}
                .legend strong {{
                    color: #059669;
                }}
                .footer {{
                    margin-top: 35px;
                    padding-top: 20px;
                    border-top: 2px solid #e5e7eb;
                }}
                .signature-title {{
                    font-weight: bold;
                    font-size: 11px;
                    color: #1f2937;
                    margin-bottom: 15px;
                }}
                .signature-container {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 40px;
                    margin-top: 20px;
                }}
                .signature-box {{
                    padding-top: 15px;
                }}
                .signature-line {{
                    border-bottom: 1px solid #1f2937;
                    height: 35px;
                    margin-bottom: 3px;
                }}
                .signature-label {{
                    font-size: 9px;
                    font-weight: bold;
                    color: #1f2937;
                    display: block;
                    margin-bottom: 2px;
                }}
                .signature-name {{
                    font-size: 9px;
                    color: #666;
                    margin-top: 1px;
                }}
                .confidential {{
                    margin-top: 20px;
                    font-size: 8px;
                    color: #999;
                    text-align: center;
                    font-style: italic;
                }}
                @media print {{
                    body {{
                        margin: 0;
                        padding: 10px;
                        background: white;
                    }}
                    .container {{
                        max-width: 100%;
                    }}
                    table {{
                        page-break-inside: avoid;
                        box-shadow: none;
                    }}
                    tr {{
                        page-break-inside: avoid;
                    }}
                    .footer {{
                        page-break-inside: avoid;
                    }}
                    .table-wrapper {{
                        overflow: visible;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <!-- ENCABEZADO -->
                <div class="header">
                    <h1>📊 INFORME DE EVALUACIÓN IMWG</h1>
                    <p>Evaluación de Respuesta en Mieloma Múltiple</p>
                    <p style="margin-top: 10px;">Generado: {datetime.now().strftime('%d de %B de %Y - %H:%M')}</p>
                </div>
                
                <!-- INFORMACIÓN DEL PACIENTE -->
                <div class="info-section">
                    <h2 style="margin-top: 0; border: none; padding: 0 0 10px 0; background: transparent;">Datos del Paciente</h2>
                    <div class="info-row">
                        <div class="info-item">
                            <span class="info-label">Código Paciente:</span>
                            <span class="info-value" style="color: #dc2626; font-size: 13px;">{codigo_paciente}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Fecha del Reporte:</span>
                            <span class="info-value">{datetime.now().strftime('%d/%m/%Y')}</span>
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-item">
                            <span class="info-label">Investigador Principal:</span>
                            <span class="info-value">{nombre_ip if nombre_ip else 'No especificado'}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Criterio de Seguimiento:</span>
                            <span class="info-value" style="color: #d97706; font-weight: 600;">{CRITERIOS_INFO[criterio_seleccionado]['nombre']}</span>
                        </div>
                    </div>
                </div>
                
                <!-- INFORMACIÓN DEL CRITERIO -->
                <div class="criteria-box">
                    <strong>📋 Criterio Aplicado:</strong><br>
                    {CRITERIOS_INFO[criterio_seleccionado]['descripcion']}
                </div>
                
                <!-- TABLA DE DATOS -->
                <h2>Datos Clínicos</h2>
                <div class="table-wrapper">
                <table border="1" cellspacing="0" cellpadding="6">
                    <thead>
                        <tr>
        """
        
        # Agregar encabezados
        for col in edited.columns:
            html_content += f"<th>{col}</th>"
        
        html_content += """
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Agregar datos de la tabla
        for _, row in edited.iterrows():
            html_content += "<tr>"
            for col in edited.columns:
                value = str(row[col]) if pd.notna(row[col]) else ""
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += f"""
                    </tbody>
                </table>
                </div>
                
                <!-- LEYENDA DE CRITERIOS -->
                <div class="legend">
                    <strong>Leyenda de Criterios de Respuesta:</strong><br>
                    <strong>RC:</strong> Remisión Completa &nbsp; | &nbsp;
                    <strong>VGPR:</strong> Muy Buena Respuesta Parcial &nbsp; | &nbsp;
                    <strong>RP:</strong> Respuesta Parcial &nbsp; | &nbsp;
                    <strong>EE:</strong> Enfermedad Estable &nbsp; | &nbsp;
                    <strong>PD:</strong> Enfermedad Progresiva
                </div>
                
                <!-- SECCIÓN DE FIRMAS -->
                <div class="footer">
                    <p class="signature-title">✍️ Firmas Autorizadas:</p>
                    
                    <div class="signature-container">
                        <div class="signature-box">
                            <div class="signature-line"></div>
                            <span class="signature-label">Investigador Principal / Médico Responsable</span>
                            <span class="signature-name">{nombre_ip if nombre_ip else '___________________________'}</span>
                        </div>
                        
                        <div class="signature-box">
                            <div class="signature-line"></div>
                            <span class="signature-label">Fecha de Aprobación</span>
                            <span class="signature-name">{datetime.now().strftime('%d / %m / %Y')}</span>
                        </div>
                    </div>
                    
                    <p class="confidential">
                        Este documento es confidencial y contiene información médica protegida bajo HIPAA/RGPD.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Guardar el HTML en un archivo temporal
        html_file = DATA_DIR / f"informe_{codigo_paciente}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Mostrar opciones
        st.markdown("---")
        st.success("✅ Informe generado exitosamente")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Botón para abrir en navegador
            if st.button("🌐 Abrir en Navegador", key="abrir_informe"):
                webbrowser.open(f"file:///{html_file.absolute()}")
                st.info("✓ Informe abierto en tu navegador. Usa Ctrl+P (o Cmd+P en Mac) para imprimir como PDF.")
        
        with col2:
            # Botón para descargar
            with open(html_file, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📥 Descargar HTML",
                    data=f.read(),
                    file_name=f"informe_{codigo_paciente}.html",
                    mime="text/html"
                )
        
        with col3:
            # Botón para cerrar
            if st.button("❌ Cerrar", key="cerrar_informe"):
                st.session_state.mostrar_impresion = False
                st.rerun()
        
        st.markdown("---")
        st.info("""
        **📖 Instrucciones de impresión:**
        1. Haz clic en "Abrir en Navegador" para ver el informe en una ventana limpia
        2. Presiona **Ctrl+P** (Windows/Linux) o **Cmd+P** (Mac)
        3. Selecciona tu impresora o "Guardar como PDF"
        4. ¡Listo! Tendrás un PDF profesional
        """)

st.caption("Desarrollado para evaluación de respuesta en mieloma múltiple según IMWG guidelines")
