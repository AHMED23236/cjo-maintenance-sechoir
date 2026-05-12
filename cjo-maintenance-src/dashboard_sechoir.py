# dashboard_sechoir.py
# ============================================================
# Dashboard Séchoir Céramique — CJO Poulina
# Pages : Vue générale / Alarmes / Pressostats / Anomalies IF / Prédiction IA
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from page_anomalies_sechoir   import show_anomalies_sechoir
from page_prediction_sechoir import show_prediction_sechoir

# ── CHEMINS ──────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR.parent / "cjo-maintenance-data-cleaned"
MODEL_DIR    = BASE_DIR

ALARM_FILE   = DATA_DIR / "Alarme_sechoir_ML.csv"
PRESSOS_FILE = DATA_DIR / "modif_Pressostats_sechoir.csv"

# ── PAGES ────────────────────────────────────────────────────
PAGES_SECHOIR = [
    "📊 Vue Générale",
    "🚨 Alarmes",
    "💧 Pressostats",
    "🔍 Anomalies IF",
    "🔮 Prédiction Temps Réel",
    "🤖 Prédiction IA",
]

# ── FEATURES XGBOOST ─────────────────────────────────────────
FEATURES = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'shift_enc',
    'is_night_shift', 'is_morning_shift',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'time_since_last_any', 'time_since_last_thermal',
    'time_since_last_mechanical', 'mean_duration_last_20',
    'max_duration_last_20', 'std_duration_last_20', 'alarm_acceleration',
]

# ── LOADERS ──────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_alarms():
    df = pd.read_csv(ALARM_FILE)
    df['window_start'] = pd.to_datetime(df['window_start'])
    df['date']  = df['window_start'].dt.date
    df['week']  = df['window_start'].dt.isocalendar().week.astype(int)
    df['month'] = df['window_start'].dt.month
    return df

@st.cache_data(ttl=60, show_spinner=False)
def load_pressostats():
    try:
        return pd.read_csv(PRESSOS_FILE, parse_dates=['timestamp'])
    except FileNotFoundError:
        return None

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        import joblib
        model    = joblib.load(MODEL_DIR / "xgboost_sechoir_binary.pkl")
        features = joblib.load(MODEL_DIR / "feature_cols_sechoir.pkl")
        return model, features
    except Exception:
        return None, None

# ── CSS ──────────────────────────────────────────────────────
def _css():
    st.markdown("""
    <style>
    .kpi-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 1rem 1.2rem;
        text-align: center; margin-bottom: 0.5rem;
    }
    .kpi-label { color: #94a3b8; font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
    .kpi-value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; }
    .kpi-sub   { color: #64748b; font-size: 0.72rem; margin-top: 0.2rem; }
    .badge { display: inline-block; padding: 2px 10px;
        border-radius: 999px; font-size: 0.72rem; font-weight: 600; margin-right: 4px; }
    .badge-high   { background: #ef444422; color: #ef4444; }
    .badge-medium { background: #f9731622; color: #f97316; }
    .badge-low    { background: #22c55e22; color: #22c55e; }
    </style>
    """, unsafe_allow_html=True)

def kpi(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class='kpi-sub'>"+sub+"</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌀 Séchoir Céramique")
        st.markdown("---")
        page = st.radio("Navigation", options=PAGES_SECHOIR,
            key="nav_sechoir", label_visibility="collapsed")
        st.markdown("---")
        if st.button("🏠 Accueil", use_container_width=True):
            st.session_state.machine = None
            st.rerun()
        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.machine = None
            st.rerun()
    return page

# ── PAGE 1 : VUE GÉNÉRALE ────────────────────────────────────
def show_vue_generale(df):
    st.markdown("## 📊 Vue Générale — Séchoir Céramique")
    st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1.5rem;'>"
        "Période : Mars → Mai 2026 &nbsp;|&nbsp; Fenêtres 5 min &nbsp;|&nbsp;"
        " Données SCADA réelles</div>", unsafe_allow_html=True)

    df_p   = df[df['failure_type'] != 'No_Failure']
    total  = len(df)
    n_p    = len(df_p)
    n_th   = (df['failure_type'] == 'Thermal_Anomaly').sum()
    n_mech = (df['failure_type'] == 'Mechanical_Stop').sum()
    n_high = (df['severity'] == 'HIGH').sum()
    n_med  = (df['severity'] == 'MEDIUM').sum()
    n_low  = (df['severity'] == 'LOW').sum()
    dur    = df_p['alarm_duration_min'].sum() if 'alarm_duration_min' in df_p.columns else 0
    dur_m  = df_p['alarm_duration_min'].mean() if 'alarm_duration_min' in df_p.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Total Fenêtres",       f"{total:,}",     "5 min chacune")
    with c2: kpi("Fenêtres Panne",       f"{n_p:,}",       f"{n_p/max(total,1)*100:.1f}%")
    with c3: kpi("Durée Totale",         f"{dur/60:.0f}h", f"moy {dur_m:.1f} min")
    with c4: kpi("Anomalies Thermiques", f"{n_th:,}",      "H200/H202/H204")
    with c5: kpi("Arrêts Mécaniques",    f"{n_mech:,}",    "H386.*")

    st.markdown(
        f"<div style='margin:0.5rem 0 1.5rem;'>"
        f"<span class='badge badge-high'>HIGH : {n_high}</span>"
        f"<span class='badge badge-medium'>MEDIUM : {n_med}</span>"
        f"<span class='badge badge-low'>LOW : {n_low}</span></div>",
        unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        ft_c = df['failure_type'].value_counts().reset_index()
        ft_c.columns = ['Type', 'Count']
        fig = px.pie(ft_c, values='Count', names='Type', color='Type',
            color_discrete_map={'No_Failure':'#22C55E','Mechanical_Stop':'#3B82F6','Thermal_Anomaly':'#EF4444'},
            title="Répartition Failure Type", hole=0.45)
        fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='#94a3b8')
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        sv_c = df[df['severity'] != 'NONE']['severity'].value_counts().reset_index()
        sv_c.columns = ['Severity', 'Count']
        sv_c = sv_c.set_index('Severity').reindex(['HIGH','MEDIUM','LOW']).reset_index().dropna()
        fig2 = px.bar(sv_c, x='Severity', y='Count', color='Severity',
            color_discrete_map={'HIGH':'#EF4444','MEDIUM':'#F97316','LOW':'#22C55E'},
            title="Distribution Severity", text='Count')
        fig2.update_traces(textposition='outside')
        fig2.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
            font_color='#94a3b8', showlegend=False,
            xaxis=dict(gridcolor='#1e293b'), yaxis=dict(gridcolor='#1e293b'))
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE 2 : ALARMES ─────────────────────────────────────────
def show_alarmes(df):
    st.markdown("## 🚨 Alarmes — Séchoir Céramique")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        months_map = {3:"Mars 2026", 4:"Avril 2026", 5:"Mai 2026"}
        sel_months = st.multiselect("Mois", options=list(months_map.keys()),
            default=list(months_map.keys()), format_func=lambda x: months_map[x])
    with col_f2:
        sel_ft = st.multiselect("Type de panne",
            options=df['failure_type'].unique().tolist(),
            default=df['failure_type'].unique().tolist())

    mask = df['month'].isin(sel_months) & df['failure_type'].isin(sel_ft)
    df_f = df[mask].copy()
    df_p = df_f[df_f['failure_type'] != 'No_Failure']
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total fenêtres", f"{len(df_f):,}")
    with c2: kpi("Fenêtres panne", f"{len(df_p):,}", f"{len(df_p)/max(len(df_f),1)*100:.1f}%")
    with c3: kpi("Thermiques", f"{(df_f['failure_type']=='Thermal_Anomaly').sum():,}")
    with c4: kpi("Mécaniques", f"{(df_f['failure_type']=='Mechanical_Stop').sum():,}")
    st.write("")

    tab1, tab2 = st.tabs(["📅 Par Jour", "📆 Par Semaine"])
    with tab1:
        daily = df_p.groupby(['date','failure_type']).size().reset_index(name='count')
        if not daily.empty:
            fig = px.bar(daily, x='date', y='count', color='failure_type',
                color_discrete_map={'Mechanical_Stop':'#3B82F6','Thermal_Anomaly':'#EF4444'},
                labels={'count':'Fenêtres','date':'Date','failure_type':'Type'},
                title="Alarmes par Jour")
            fig.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
                font_color='#94a3b8', barmode='stack',
                xaxis=dict(gridcolor='#1e293b'), yaxis=dict(gridcolor='#1e293b'))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        weekly = df_p.groupby(['week','failure_type']).size().reset_index(name='count')
        weekly['week_label'] = "Sem. " + weekly['week'].astype(str)
        if not weekly.empty:
            fig2 = px.bar(weekly, x='week_label', y='count', color='failure_type',
                color_discrete_map={'Mechanical_Stop':'#3B82F6','Thermal_Anomaly':'#EF4444'},
                labels={'count':'Fenêtres','week_label':'Semaine'},
                title="Alarmes par Semaine")
            fig2.update_layout(plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
                font_color='#94a3b8', barmode='stack',
                xaxis=dict(gridcolor='#1e293b'), yaxis=dict(gridcolor='#1e293b'))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Détail des alarmes")
    cols_show = ['window_start','failure_type','severity','alarm_duration_min']
    cols_ok   = [c for c in cols_show if c in df_p.columns]
    st.dataframe(
        df_p[cols_ok].sort_values('window_start', ascending=False).reset_index(drop=True),
        use_container_width=True)

# ── PAGE 3 : PRESSOSTATS ─────────────────────────────────────
def show_pressostats(df_press):
    st.markdown("## 💧 Pressostats — EAU11 / EAU12 / EAU21 / EAU22")
    st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:1.5rem;'>"
        "Données réelles SCADA — 08-09 Nov 2016 — 1 mesure/min</div>",
        unsafe_allow_html=True)

    if df_press is None:
        st.warning("Fichier `modif_Pressostats_sechoir.csv` non trouvé.")
        return

    all_caps = ['EAU11','EAU12','EAU21','EAU22']
    available_caps = [c for c in all_caps if c in df_press.columns]
    if not available_caps:
        st.error("Aucune colonne EAU11/EAU12/EAU21/EAU22 trouvée dans le fichier.")
        return

    caps = st.multiselect("Capteurs", options=available_caps, default=available_caps)

    if not caps:
        st.info("Sélectionnez au moins un capteur.")
        return

    colors_eau = {'EAU11':'#3B82F6','EAU12':'#8B5CF6','EAU21':'#F97316','EAU22':'#EF4444'}
    fig = go.Figure()
    for c in caps:
        fig.add_trace(go.Scatter(x=df_press['timestamp'], y=df_press[c],
            name=c, line=dict(color=colors_eau[c], width=1.5), mode='lines'))
    fig.update_layout(
        title="Pressostats — Données réelles SCADA",
        plot_bgcolor='#0f172a', paper_bgcolor='#0f172a', font_color='#94a3b8',
        xaxis=dict(gridcolor='#1e293b', title='Timestamp'),
        yaxis=dict(gridcolor='#1e293b', title='Pression (Pa)'),
        legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
        hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Statistiques descriptives")
    st.dataframe(df_press[caps].describe().round(4), use_container_width=True)

# ── PAGE 4 : PRÉDICTION IA ───────────────────────────────────
def show_prediction():
    st.markdown("## 🤖 Prédiction IA — XGBoost Binaire Séchoir")
    st.markdown(
        f"<div style='color:#64748b;font-size:0.85rem;margin-bottom:1.5rem;'>"
        "Modèle binaire &nbsp;|&nbsp; 0 = Normal &nbsp;|&nbsp; 1 = ANORMAL "
        "(panne dans les 2h) &nbsp;|&nbsp; Données : Mars → Mai 2026</div>",
        unsafe_allow_html=True,
    )

    # ── KPIs ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Accuracy", "51.7%",  "Données complètes")
    with c2: kpi("F1",       "0.521",  "Weighted")
    with c3: kpi("AUC-ROC",  "0.801",  "Bonne discrimination")
    with c4: kpi("Seuil",    "50%",    "Prob. ANORMAL")

    st.markdown("---")

    # ── Matrice de confusion ──────────────────────────────────
    cm = np.array([[72, 11], [128, 77]])
    labels = ['Normal (0)', 'ANORMAL (1)']

    fig = px.imshow(
        cm, x=labels, y=labels,
        labels=dict(x="Prédit", y="Réel", color="Count"),
        color_continuous_scale='Blues', text_auto=True,
        title="Matrice de Confusion — XGBoost Binaire",
    )
    fig.update_layout(
        plot_bgcolor='#0f172a', paper_bgcolor='#0f172a',
        font_color='#94a3b8', height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Métriques détaillées ──────────────────────────────────
    st.markdown("### Métriques par classe")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Classe 0 — Normal**")
        tn, fp = 72, 11
        precision0 = tn / (tn + 128) if (tn + 128) > 0 else 0
        recall0    = tn / (tn + fp)  if (tn + fp)  > 0 else 0
        kpi("Précision", f"{precision0*100:.1f}%", "Normal prédit correct")
        kpi("Rappel",    f"{recall0*100:.1f}%",    "Normal réel détecté")
    with c2:
        st.markdown("**Classe 1 — ANORMAL**")
        tp, fn = 77, 128
        precision1 = tp / (tp + fp)  if (tp + fp)  > 0 else 0
        recall1    = tp / (tp + fn)  if (tp + fn)  > 0 else 0
        kpi("Précision", f"{precision1*100:.1f}%", "ANORMAL prédit correct")
        kpi("Rappel",    f"{recall1*100:.1f}%",    "ANORMAL réel détecté")

    st.info("💡 AUC-ROC = 0.801 : bonne capacité de discrimination. "
            "L'accuracy faible s'explique par le déséquilibre des classes (71% ANORMAL).")

# ── ENTRY POINT ──────────────────────────────────────────────
def show_dashboard_sechoir():
    _css()
    df       = load_alarms()
    df_press = load_pressostats()
    page     = render_sidebar()

    if page == "📊 Vue Générale":
        show_vue_generale(df)
    elif page == "🚨 Alarmes":
        show_alarmes(df)
    elif page == "💧 Pressostats":
        show_pressostats(df_press)
    elif page == "🔍 Anomalies IF":
        show_anomalies_sechoir()
    elif page == "🔮 Prédiction Temps Réel":
        show_prediction_sechoir()
    elif page == "🤖 Prédiction IA":
        show_prediction()