# page_anomalies_sechoir.py
# ============================================================
# Page Anomalies Séchoir — Isolation Forest
# CJO Poulina — PFE Maintenance Prédictive
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR.parent / "cjo-maintenance-data-cleaned"
RESULTS_FILE  = DATA_DIR / "sechoir_anomalies_results.csv"
CAPTEURS_FILE = DATA_DIR / "sechoir_capteurs_IF.csv"

DARK  = '#0f172a'
CARD  = '#1e293b'
BORD  = '#334155'
TEXT  = '#94a3b8'
WHITE = '#f1f5f9'

# ── CSS ──────────────────────────────────────────────────────
def _css():
    st.markdown("""
    <style>
    .kpi-card {
        background:#1e293b; border:1px solid #334155;
        border-radius:12px; padding:1rem 1.2rem;
        text-align:center; margin-bottom:0.5rem;
    }
    .kpi-label { color:#94a3b8; font-size:0.75rem;
        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem; }
    .kpi-value { font-size:1.8rem; font-weight:700; }
    .kpi-sub   { color:#64748b; font-size:0.72rem; margin-top:0.2rem; }
    </style>
    """, unsafe_allow_html=True)

def kpi(label, value, sub="", color=WHITE):
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)

# ── LOADERS ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_results():
    df = pd.read_csv(RESULTS_FILE, parse_dates=['timestamp'])
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_capteurs():
    df = pd.read_csv(CAPTEURS_FILE, parse_dates=['timestamp'])
    return df

# ── LAYOUT COMMUN ────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor=DARK, paper_bgcolor=DARK, font_color=TEXT,
    xaxis=dict(gridcolor=CARD), yaxis=dict(gridcolor=CARD),
    legend=dict(bgcolor=CARD, bordercolor=BORD),
    hovermode='x unified', height=400,
)

# ── PAGE PRINCIPALE ──────────────────────────────────────────
def show_anomalies_sechoir():
    _css()

    st.markdown("## 🔍 Détection d'Anomalies — Isolation Forest")
    st.markdown(
        f"<div style='color:{TEXT};font-size:0.85rem;margin-bottom:1.5rem;'>"
        "Isolation Forest &nbsp;|&nbsp; Capteurs synthétiques calibrés SCADA &nbsp;|&nbsp;"
        " Mars → Mai 2026 &nbsp;|&nbsp; 1 mesure/min</div>",
        unsafe_allow_html=True
    )

    try:
        df_res = load_results()
        df_cap = load_capteurs()
    except FileNotFoundError as e:
        st.error(f"Fichier manquant : {e.filename}  —  Lance d'abord `train_isolation_forest_sechoir.py`")
        return

    # Colonnes attendues
    y_true = df_res['true_label'].values
    y_pred = df_res['anomaly_pred'].values
    scores = df_res['anomaly_score'].values

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float('nan')

    # ── KPIs MODÈLE ──────────────────────────────────────────
    st.markdown("### Performances du modèle")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Accuracy",   f"{acc*100:.1f}%", "Test",          "#22C55E")
    with c2: kpi("F1 Score",   f"{f1:.3f}",        "Weighted",      "#22C55E")
    with c3: kpi("AUC-ROC",    f"{auc:.3f}",        "Binaire",       "#3B82F6")
    with c4: kpi("Algorithme", "IF",                "IsolationForest","#8B5CF6")
    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 Anomaly Score",
        "🌡️ Températures",
        "📋 Détail anomalies",
    ])

    # ── TAB 1 : ANOMALY SCORE ────────────────────────────────
    with tab1:
        n_total   = len(df_res)
        n_anom    = int(y_pred.sum())
        n_real    = int(y_true.sum())
        n_correct = int((y_pred == y_true).sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Total mesures",       f"{n_total:,}",   "Test")
        with c2: kpi("Anomalies détectées", f"{n_anom:,}",    f"{n_anom/n_total*100:.1f}%", "#EF4444")
        with c3: kpi("Anomalies réelles",   f"{n_real:,}",    f"{n_real/n_total*100:.1f}%", "#F97316")
        with c4: kpi("Correct",             f"{n_correct:,}", f"{n_correct/n_total*100:.1f}%", "#22C55E")
        st.write("")

        threshold = np.percentile(scores, (1 - 0.16) * 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_res['timestamp'], y=scores,
            mode='lines', name='Anomaly Score',
            line=dict(color='#3B82F6', width=0.8), opacity=0.8,
        ))
        real_idx = df_res[y_true == 1]
        fig.add_trace(go.Scatter(
            x=real_idx['timestamp'], y=real_idx['anomaly_score'],
            mode='markers', name='Anomalie réelle',
            marker=dict(color='#EF4444', size=3, opacity=0.5),
        ))
        fig.add_hline(
            y=threshold, line_dash="dash",
            line_color="#F97316", line_width=1.5,
            annotation_text=f"Seuil ({threshold:.3f})",
            annotation_font_color="#F97316",
        )
        fig.update_layout(title="Anomaly Score — Isolation Forest", **_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Matrice de confusion
        st.markdown("#### Matrice de Confusion")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm,
            x=['Normal', 'Anomalie'], y=['Normal', 'Anomalie'],
            labels=dict(x="Prédit", y="Réel", color="Count"),
            color_continuous_scale='Blues', text_auto=True,
            title=f"Confusion — Acc={acc*100:.1f}% | F1={f1:.3f} | AUC={auc:.3f}",
        )
        fig_cm.update_layout(plot_bgcolor=DARK, paper_bgcolor=DARK, font_color=TEXT)
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── TAB 2 : TEMPÉRATURES ─────────────────────────────────
    with tab2:
        st.markdown("#### Températures capteurs")

        sensors_avail = [s for s in
            ['Temp_Module1','Temp_Module2','Temp_Module3','Temp_Bruleur3']
            if s in df_cap.columns]

        sensors = st.multiselect(
            "Capteurs", options=sensors_avail,
            default=sensors_avail[-2:] if len(sensors_avail) >= 2 else sensors_avail,
        )
        if not sensors:
            st.info("Sélectionnez au moins un capteur.")
        else:
            colors_map = {
                'Temp_Module1':  '#3B82F6',
                'Temp_Module2':  '#8B5CF6',
                'Temp_Module3':  '#F97316',
                'Temp_Bruleur3': '#EF4444',
            }
            sample = df_cap.iloc[::5]

            fig2 = go.Figure()
            for s in sensors:
                fig2.add_trace(go.Scatter(
                    x=sample['timestamp'], y=sample[s],
                    mode='lines', name=s,
                    line=dict(color=colors_map.get(s, TEXT), width=0.8),
                ))

            if 'alarm_active' in df_cap.columns:
                anom_cap = df_cap[df_cap['alarm_active'] == 1].iloc[::5]
                fig2.add_trace(go.Scatter(
                    x=anom_cap['timestamp'], y=anom_cap[sensors[0]],
                    mode='markers', name='Anomalie active',
                    marker=dict(color='#EF4444', size=3, opacity=0.4),
                ))

            fig2.update_layout(title="Températures avec anomalies", **_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Statistiques descriptives")
            st.dataframe(df_cap[sensors].describe().round(2), use_container_width=True)

            if 'failure_type' in df_cap.columns and 'alarm_active' in df_cap.columns:
                st.markdown("#### Taux d'activité par type de panne")
                det = (df_cap[df_cap['failure_type'] != 'Normal']
                       .groupby('failure_type')['alarm_active']
                       .mean()
                       .mul(100)
                       .reset_index())
                det.columns = ['Type', 'Taux (%)']
                if not det.empty:
                    fig3 = px.bar(
                        det, x='Type', y='Taux (%)', color='Type',
                        color_discrete_map={
                            'Burner_Failure':   '#EF4444',
                            'Temp_Deviation':   '#F97316',
                            'Gradient_Anomaly': '#8B5CF6',
                        },
                        title="Taux d'activité alarme par type de panne",
                        text='Taux (%)',
                    )
                    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig3.update_layout(
                        plot_bgcolor=DARK, paper_bgcolor=DARK,
                        font_color=TEXT, showlegend=False,
                        xaxis=dict(gridcolor=CARD),
                        yaxis=dict(gridcolor=CARD, range=[0, 115]),
                    )
                    st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3 : DÉTAIL ANOMALIES ─────────────────────────────
    with tab3:
        st.markdown("#### Anomalies détectées par le modèle")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_tp = st.checkbox("Vrais positifs uniquement", value=False)
        with col_f2:
            show_fn = st.checkbox("Faux négatifs (manqués)", value=False)

        if show_fn:
            df_detail = df_res[(df_res['anomaly_pred'] == 0) & (df_res['true_label'] == 1)].copy()
            st.warning(f"{len(df_detail)} anomalies manquées par le modèle")
        else:
            df_detail = df_res[df_res['anomaly_pred'] == 1].copy()
            if show_tp:
                df_detail = df_detail[df_detail['true_label'] == 1]

        cols_show = ['timestamp', 'failure_type', 'anomaly_score',
                     'anomaly_pred', 'true_label', 'correct']
        sensor_cols = [c for c in ['Temp_Module3', 'Temp_Bruleur3',
                                   'Temp_Module1', 'Temp_Module2'] if c in df_detail.columns]
        cols_show = cols_show + sensor_cols

        st.dataframe(
            df_detail[cols_show].sort_values('anomaly_score', ascending=False)
                                .reset_index(drop=True),
            use_container_width=True,
        )

        st.markdown("---")
        st.info(
            "**Note méthodologique** : Les données capteurs sont synthétiques, "
            "calibrées sur les données réelles SCADA CJO "
            "(codes alarmes H196/H198/H200/H204/H220, fréquences et durées réelles). "
            "En production, ce modèle sera connecté directement au flux SCADA temps réel."
        )
