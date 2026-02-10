import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from src.config import PROCESSED_DATA_DIR

API_BASE_URL = "https://home-credit-project.onrender.com"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
SHAP_ENDPOINT = f"{API_BASE_URL}/shap"
THRESHOLD_FALLBACK = 0.54

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# WCAG 1.4.3 compliant palette (contrast ratio >= 4.5:1 on white background)
# ---------------------------------------------------------------------------
COLOR_POSITIVE_SHAP = "#B91C1C"   # dark red  – increases risk
COLOR_NEGATIVE_SHAP = "#15803D"   # dark green – decreases risk
COLOR_ACCEPTED = "#15803D"
COLOR_REFUSED = "#B91C1C"
COLOR_CLIENT_MARKER = "#1D4ED8"   # dark blue – client position

# ---------------------------------------------------------------------------
# WCAG 2.4.2 – page title
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Home Credit Scoring Dashboard",
    layout="wide",
)

# ---------------------------------------------------------------------------
# WCAG 1.4.4 – allow text resizing via custom CSS base font
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="st-"] { font-size: 1rem; }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.25rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_clients_sample() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "clients_sample.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {path}\n"
            f"Lance d'abord: python scripts/make_clients_sample.py"
        )
    df = pd.read_parquet(path)
    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Le parquet doit contenir la colonne SK_ID_CURR.")
    return df



# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def to_jsonable(val):
    if pd.isna(val):
        return None
    if isinstance(val, (np.floating, float)) and np.isinf(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


_INTERNAL_COLS = {"PROBA_DEFAULT", "DECISION"}


def build_payload_from_row(row: pd.Series) -> dict:
    return {
        col: to_jsonable(row[col])
        for col in row.index
        if col not in _INTERNAL_COLS
    }


@st.cache_data(show_spinner=False)
def call_predict_api_cached(payload_str: str) -> dict:
    payload = json.loads(payload_str)
    r = requests.post(PREDICT_ENDPOINT, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def call_shap_api_cached(payload_str: str) -> dict:
    payload = json.loads(payload_str)
    r = requests.post(SHAP_ENDPOINT, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def compute_age_years(row: pd.Series):
    if "DAYS_BIRTH" not in row.index:
        return None
    try:
        return round(abs(float(row["DAYS_BIRTH"])) / 365.25, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Speedometer gauge (WCAG: text labels + high contrast – 1.4.1, 1.4.3)
# ---------------------------------------------------------------------------

def make_speedometer_gauge(prob: float, threshold: float, title: str):
    """Half-circle speedometer gauge using Plotly Indicator."""
    prob = float(min(max(prob, 0.0), 1.0))
    threshold = float(min(max(threshold, 0.0), 1.0))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"valueformat": ".3f", "font": {"size": 32, "color": "#111827"}},
            title={"text": title, "font": {"size": 18, "color": "#111827"}},
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickvals": [0, threshold, 0.5, 1],
                    "ticktext": ["0", f"seuil {threshold:.2f}", "0.5", "1"],
                    "tickfont": {"size": 12, "color": "#111827"},
                },
                "bar": {"color": "#111827", "thickness": 0.25},
                "steps": [
                    {"range": [0, threshold], "color": "rgba(21, 128, 61, 0.30)"},
                    {"range": [threshold, 1], "color": "rgba(185, 28, 28, 0.30)"},
                ],
                "threshold": {
                    "line": {"color": "#374151", "width": 4},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
        )
    )

    # WCAG 1.4.1 – text labels so zones are not identified by color alone
    fig.add_annotation(
        text="ACCEPTE", x=0.20, y=-0.10, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color=COLOR_ACCEPTED),
    )
    fig.add_annotation(
        text="REFUSE", x=0.80, y=-0.10, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color=COLOR_REFUSED),
    )

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor="white",
    )

    return fig


# ---------------------------------------------------------------------------
# SHAP waterfall chart (WCAG: color + text labels – 1.4.1, 1.1.1 alt via title)
# ---------------------------------------------------------------------------

def make_shap_waterfall(shap_data: dict):
    """Build a horizontal bar chart of top SHAP values."""
    top_shap = shap_data["top_shap"]
    base_value = shap_data["base_value"]

    features = [item["feature"] for item in reversed(top_shap)]
    values = [item["shap_value"] for item in reversed(top_shap)]

    colors = [COLOR_POSITIVE_SHAP if v > 0 else COLOR_NEGATIVE_SHAP for v in values]

    # WCAG 1.4.1: use hatching pattern text in addition to color
    text_labels = [
        f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values, y=features,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="outside",
            textfont=dict(size=12, color="#111827"),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Importance locale SHAP (base = {base_value:.3f}). "
                "Rouge : augmente le risque. Vert : diminue le risque."
            ),
            font=dict(size=13, color="#111827"),
        ),
        height=max(350, len(top_shap) * 38 + 80),
        margin=dict(l=10, r=80, t=50, b=30),
        xaxis=dict(title="Valeur SHAP", zeroline=True, zerolinecolor="#9CA3AF"),
        yaxis=dict(automargin=True),
        plot_bgcolor="white",
    )

    return fig


# ---------------------------------------------------------------------------
# Comparison distribution chart (WCAG compliant)
# ---------------------------------------------------------------------------

def make_comparison_histogram(
    df: pd.DataFrame,
    column: str,
    client_value: float,
    label: str,
):
    """Histogram split by model decision (accepted vs refused) with client marker."""
    fig = go.Figure()

    accepted = df.loc[df["DECISION"] == "ACCEPTED", column].dropna()
    fig.add_trace(
        go.Histogram(
            x=accepted,
            nbinsx=50,
            marker_color=COLOR_ACCEPTED,
            opacity=0.55,
            name="Acceptes",
            hovertemplate=f"{label}: " + "%{x}<br>Effectif: %{y}<extra>Acceptes</extra>",
        )
    )

    refused = df.loc[df["DECISION"] == "REFUSED", column].dropna()
    fig.add_trace(
        go.Histogram(
            x=refused,
            nbinsx=50,
            marker_color=COLOR_REFUSED,
            opacity=0.55,
            name="Refuses",
            hovertemplate=f"{label}: " + "%{x}<br>Effectif: %{y}<extra>Refuses</extra>",
        )
    )

    # Client position – vertical line + annotation (WCAG 1.4.1: shape + text)
    fig.add_vline(
        x=client_value,
        line_width=3,
        line_dash="solid",
        line_color=COLOR_CLIENT_MARKER,
        annotation_text=f"Client: {client_value:,.1f}",
        annotation_position="top right",
        annotation_font=dict(size=13, color=COLOR_CLIENT_MARKER),
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(
            text=(
                f"Distribution de {label} – Acceptes (vert) vs Refuses (rouge). "
                "Trait bleu = client selectionne."
            ),
            font=dict(size=13, color="#111827"),
        ),
        xaxis_title=label,
        yaxis_title="Nombre de clients",
        height=370,
        margin=dict(l=50, r=30, t=60, b=40),
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12, color="#111827"),
        ),
    )

    return fig


# ===========================================================================
# MAIN APP
# ===========================================================================

st.title("Home Credit — Scoring")
st.write(
    "Sélectionnez un client pour obtenir la décision du modèle, "
    "comprendre les facteurs explicatifs (SHAP) et comparer son profil "
    "à l'ensemble des clients."
)

with st.spinner("Chargement des clients..."):
    df_clients = load_clients_sample()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Client")
st.sidebar.write(f"{len(df_clients):,} clients (sample)")

default_id = int(df_clients["SK_ID_CURR"].iloc[0])
sk_id = st.sidebar.number_input("SK_ID_CURR", min_value=0, value=default_id, step=1)
run = st.sidebar.button("Predire")

# ---------------------------------------------------------------------------
# Sidebar – filter for comparison charts
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Comparaison")
filter_gender = st.sidebar.selectbox(
    "Filtrer par genre",
    options=["Tous", "F", "M"],
    index=0,
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if run:
    matches = df_clients[df_clients["SK_ID_CURR"] == sk_id]
    if matches.empty:
        st.error("Client introuvable dans clients_sample.parquet.")
        st.stop()

    row = matches.iloc[0]
    payload = build_payload_from_row(row)
    payload_str = json.dumps(payload, sort_keys=True)

    # --- API calls (predict + SHAP) ---
    try:
        with st.spinner("Appel API /predict..."):
            pred = call_predict_api_cached(payload_str)
    except requests.HTTPError as e:
        st.error(f"Erreur HTTP: {e}")
        st.code(getattr(e.response, "text", ""))
        st.stop()
    except Exception as e:
        st.error(f"Erreur: {e}")
        st.stop()

    if pred.get("status") != "ok":
        st.error(f"Reponse API inattendue: {pred}")
        st.stop()

    proba = float(pred["probability_default"])
    decision = pred["decision"]
    threshold_used = float(pred.get("threshold_used", THRESHOLD_FALLBACK))
    predicted_label = int(pred.get("predicted_label", 1 if proba >= threshold_used else 0))

    api_id = pred.get("sk_id_curr")
    if api_id is not None and int(api_id) != int(sk_id):
        st.warning(f"Attention: l'API renvoie sk_id_curr={api_id} (different de {sk_id}).")

    age = compute_age_years(row)
    amt_credit = row.get("AMT_CREDIT", None)

    # ===================================================================
    # SECTION 1 – Decision & Score
    # ===================================================================
    st.subheader("Resultat de la prediction")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Client", f"{sk_id}")
    c2.metric("Proba defaut", f"{proba:.4f}")
    c3.metric("Seuil", f"{threshold_used:.2f}")
    c4.metric("Label", f"{predicted_label}")

    if decision == "REFUSED":
        st.error("Decision : REFUSE — Le modele recommande de refuser le pret.")
    elif decision == "ACCEPTED":
        st.success("Decision : ACCEPTE — Le modele recommande d'accorder le pret.")
    else:
        st.info(f"Decision : {decision}")

    gauge_title = "ACCEPTE" if decision == "ACCEPTED" else "REFUSE"
    fig_gauge = make_speedometer_gauge(proba, threshold_used, title=gauge_title)
    st.plotly_chart(fig_gauge, use_container_width=True)

    if proba < threshold_used:
        st.write(
            f"Le score ({proba:.3f}) est inferieur au seuil ({threshold_used:.2f}). "
            "Le risque est juge acceptable."
        )
    else:
        st.write(
            f"Le score ({proba:.3f}) est superieur ou egal au seuil ({threshold_used:.2f}). "
            "Le risque est juge trop eleve."
        )

    # Client info
    st.subheader("Informations client")
    i1, i2 = st.columns(2)
    if age is not None:
        i1.metric("Age (approx.)", f"{age} ans")
    else:
        i1.write("Age : indisponible")
    if amt_credit is not None and not pd.isna(amt_credit):
        i2.metric("AMT_CREDIT", f"{float(amt_credit):,.0f}")
    else:
        i2.write("AMT_CREDIT : indisponible")

    # ===================================================================
    # SECTION 2 – SHAP local explanation
    # ===================================================================
    st.markdown("---")
    st.subheader("Explication locale (SHAP)")
    st.write(
        "Le graphique ci-dessous montre les variables qui ont le plus influence "
        "la decision du modele pour ce client. "
        "Les barres **rouges** augmentent le risque de defaut, "
        "les barres **vertes** le diminuent."
    )

    try:
        with st.spinner("Calcul SHAP en cours..."):
            shap_data = call_shap_api_cached(payload_str)

        if shap_data.get("status") == "ok":
            fig_shap = make_shap_waterfall(shap_data)
            st.plotly_chart(fig_shap, use_container_width=True)

            # Textual explanation for the analyst (WCAG 1.1.1 – non-text alternative)
            top3 = shap_data["top_shap"][:3]
            explanation_parts = []
            for item in top3:
                direction = "augmente" if item["shap_value"] > 0 else "diminue"
                explanation_parts.append(
                    f"**{item['feature']}** (SHAP = {item['shap_value']:+.3f}) "
                    f"{direction} le risque"
                )
            st.write(
                "**Resume pour le charge d'etude :** Les 3 facteurs les plus importants "
                "pour ce client sont : " + " ; ".join(explanation_parts) + "."
            )
        else:
            st.warning("L'API SHAP n'a pas renvoye de resultat exploitable.")

    except Exception as e:
        st.warning(f"Impossible de recuperer les valeurs SHAP : {e}")

    # ===================================================================
    # SECTION 3 – Comparison charts (2 variables)
    # ===================================================================
    st.markdown("---")
    st.subheader("Comparaison avec les autres clients")
    st.write(
        "Position du client par rapport aux clients acceptes (vert) et refuses (rouge) "
        f"(filtre : genre = {filter_gender}). "
        "Le trait bleu vertical indique la valeur du client selectionne."
    )

    # Apply filter
    if filter_gender == "Tous":
        df_filtered = df_clients
    else:
        df_filtered = df_clients[df_clients["CODE_GENDER"] == filter_gender]

    # Variable 1: AMT_CREDIT (montant du credit)
    # Variable 2: Age (derived from DAYS_BIRTH)
    col_left, col_right = st.columns(2)

    with col_left:
        client_credit = row.get("AMT_CREDIT", None)
        if client_credit is not None and not pd.isna(client_credit):
            fig_credit = make_comparison_histogram(
                df_filtered, "AMT_CREDIT", float(client_credit),
                label="Montant du credit (AMT_CREDIT)",
            )
            st.plotly_chart(fig_credit, use_container_width=True)
            # WCAG 1.1.1 – textual alternative
            med_acc = df_filtered.loc[
                df_filtered["DECISION"] == "ACCEPTED", "AMT_CREDIT"
            ].median()
            med_ref = df_filtered.loc[
                df_filtered["DECISION"] == "REFUSED", "AMT_CREDIT"
            ].median()
            st.write(
                f"Credit du client : **{float(client_credit):,.0f}**. "
                f"Mediane acceptes : **{med_acc:,.0f}** ; "
                f"mediane refuses : **{med_ref:,.0f}**."
            )
        else:
            st.write("AMT_CREDIT indisponible pour ce client.")

    with col_right:
        # Compute age column for distribution
        df_age = df_filtered.copy()
        df_age["AGE_YEARS"] = (df_age["DAYS_BIRTH"].abs() / 365.25).round(1)
        client_age_val = age if age is not None else np.nan
        if not pd.isna(client_age_val):
            fig_age = make_comparison_histogram(
                df_age, "AGE_YEARS", float(client_age_val),
                label="Age (annees)",
            )
            st.plotly_chart(fig_age, use_container_width=True)
            med_age_acc = df_age.loc[
                df_age["DECISION"] == "ACCEPTED", "AGE_YEARS"
            ].median()
            med_age_ref = df_age.loc[
                df_age["DECISION"] == "REFUSED", "AGE_YEARS"
            ].median()
            st.write(
                f"Age du client : **{client_age_val} ans**. "
                f"Mediane acceptes : **{med_age_acc:.1f} ans** ; "
                f"mediane refuses : **{med_age_ref:.1f} ans**."
            )
        else:
            st.write("Age indisponible pour ce client.")

    # ===================================================================
    # Expanders (raw data)
    # ===================================================================
    with st.expander("Reponse brute API"):
        st.code(json.dumps(pred, indent=2, ensure_ascii=False))

    with st.expander("Payload envoye"):
        st.code(json.dumps(payload, indent=2, ensure_ascii=False))

else:
    st.info("Entrez le SK_ID_CURR a gauche puis cliquez sur Predire.")
