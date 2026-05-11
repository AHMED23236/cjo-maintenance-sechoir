# page_prediction_sechoir.py
# ============================================================
# Prédiction Temps Réel — Séchoir Céramique
# XGBoost Binaire (0=Normal / 1=ANORMAL HIGH dans 2h)
# + Rule Mapping type de panne + Alerte mail
# CJO Poulina — PFE Maintenance Prédictive
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path

# ── CHEMINS ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "cjo-maintenance-data-cleaned"

MODEL_FILE    = BASE_DIR / "xgboost_sechoir_binary.pkl"
FEATURES_FILE = BASE_DIR / "feature_cols_sechoir.pkl"
RULES_FILE    = BASE_DIR / "rule_mapping_sechoir.pkl"
DATA_FILE     = DATA_DIR / "Alarme_sechoir_ML.csv"
HIST_FILE     = DATA_DIR / "prediction_history_sechoir.csv"

# ── CONFIG MAIL (à adapter) ───────────────────────────────────
MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "cjoadmin@gmail.com",
    "password":    "cxhlmuudblbtxozj",
    "recipient":   "ahmed.naffeti10@gmail.com",
}

# ── FONCTION REQUISE PAR rule_mapping_sechoir.pkl ────────────
# joblib cherche predict_failure_type dans __main__ lors du chargement
def predict_failure_type(row):
    """Rule mapping : déduit le type de panne à partir des features."""
    try:
        thermal = row.get('past_thermal_6h',   0) if hasattr(row, 'get') else float(row['past_thermal_6h'])
        mech    = row.get('past_mechanical_6h', 0) if hasattr(row, 'get') else float(row['past_mechanical_6h'])
        eau     = row.get('EAU_max_all',        0) if hasattr(row, 'get') else float(row['EAU_max_all'])
    except Exception:
        thermal, mech, eau = 0, 0, 0
    if thermal > mech or eau > 1.5:
        return "Thermal_Anomaly"
    return "Mechanical_Stop"

REFRESH_INTERVAL = 30          # minutes entre deux prédictions auto
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
        border-radius:12px; padding:1.1rem 1.4rem;
        text-align:center; margin-bottom:0.5rem;
    }
    .kpi-label { color:#94a3b8; font-size:0.72rem; text-transform:uppercase;
        letter-spacing:0.08em; margin-bottom:0.3rem; }
    .kpi-value { font-size:1.8rem; font-weight:700; }
    .kpi-sub   { color:#64748b; font-size:0.70rem; margin-top:0.25rem; }

    .status-box {
        border-radius:16px; padding:1.6rem 2rem;
        text-align:center; margin-bottom:1rem;
    }
    .status-normal  { background:#052e1622; border:2px solid #22c55e; }
    .status-anormal { background:#45031722; border:2px solid #ef4444; }

    .status-icon  { font-size:3.5rem; line-height:1; margin-bottom:0.4rem; }
    .status-label { font-size:2rem; font-weight:800; letter-spacing:0.05em; }
    .status-sub   { color:#94a3b8; font-size:0.82rem; margin-top:0.4rem; }
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
@st.cache_resource(show_spinner=False)
def load_model_assets():
    try:
        model    = joblib.load(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
    except Exception:
        return None, None, None
    try:
        rules = joblib.load(RULES_FILE)
    except Exception:
        rules = predict_failure_type   # fallback : fonction locale
    return model, features, rules

@st.cache_data(ttl=60, show_spinner=False)
def load_live_data():
    df = pd.read_csv(DATA_FILE)
    df['window_start'] = pd.to_datetime(df['window_start'])
    return df

def load_history():
    if HIST_FILE.exists():
        return pd.read_csv(HIST_FILE, parse_dates=['timestamp'])
    return pd.DataFrame(columns=[
        'timestamp', 'prediction', 'probability',
        'failure_type_pred', 'window_start', 'mail_sent',
    ])

def save_history(hist_df):
    hist_df.tail(500).to_csv(HIST_FILE, index=False)

# ── RÈGLES TYPE DE PANNE ─────────────────────────────────────
def apply_rule_mapping(row: pd.Series, rules) -> str:
    """Applique le rule_mapping pour déterminer le type de panne prédit."""
    if rules is None:
        # Règle de secours basée sur les features disponibles
        if row.get('past_thermal_6h', 0) > row.get('past_mechanical_6h', 0):
            return "Thermal_Anomaly"
        return "Mechanical_Stop"

    # Si rules est un dict de conditions
    if isinstance(rules, dict):
        for panne_type, conditions in rules.items():
            match = all(
                row.get(feat, 0) >= thresh
                for feat, thresh in conditions.items()
            )
            if match:
                return panne_type
        return "Mechanical_Stop"

    # Si rules est callable
    if callable(rules):
        try:
            return rules(row)
        except Exception:
            pass

    return "Inconnu"

# ── ENVOI MAIL ───────────────────────────────────────────────
def send_alert_mail(prob: float, failure_type: str, window_start: str) -> bool:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[CJO ALERTE] Séchoir — Anomalie HIGH prédite ({prob*100:.0f}%)"
        msg["From"]    = MAIL_CONFIG["sender"]
        msg["To"]      = MAIL_CONFIG["recipient"]

        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0f172a;color:#f1f5f9;padding:20px;">
        <div style="max-width:600px;margin:auto;background:#1e293b;border-radius:12px;
                    padding:24px;border:2px solid #ef4444;">
            <h2 style="color:#ef4444;margin-top:0;">🔴 ALERTE MAINTENANCE — Séchoir Céramique</h2>
            <p style="color:#94a3b8;">Une anomalie de niveau <strong style="color:#ef4444">HIGH</strong>
            a été prédite par le système de maintenance prédictive CJO.</p>
            <table style="width:100%;border-collapse:collapse;margin:16px 0;">
                <tr style="background:#0f172a;">
                    <td style="padding:10px;color:#94a3b8;">Fenêtre temporelle</td>
                    <td style="padding:10px;color:#f1f5f9;font-weight:bold;">{window_start}</td>
                </tr>
                <tr>
                    <td style="padding:10px;color:#94a3b8;">Probabilité d'anomalie</td>
                    <td style="padding:10px;color:#ef4444;font-weight:bold;">{prob*100:.1f}%</td>
                </tr>
                <tr style="background:#0f172a;">
                    <td style="padding:10px;color:#94a3b8;">Type de panne prédit</td>
                    <td style="padding:10px;color:#f97316;font-weight:bold;">{failure_type}</td>
                </tr>
                <tr>
                    <td style="padding:10px;color:#94a3b8;">Heure d'alerte</td>
                    <td style="padding:10px;color:#f1f5f9;">{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</td>
                </tr>
            </table>
            <p style="color:#64748b;font-size:12px;margin-bottom:0;">
            CJO Poulina — Système de maintenance prédictive · Message automatique</p>
        </div></body></html>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(MAIL_CONFIG["smtp_server"], MAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(MAIL_CONFIG["sender"], MAIL_CONFIG["password"])
            server.sendmail(MAIL_CONFIG["sender"], MAIL_CONFIG["recipient"], msg.as_string())
        return True
    except Exception:
        return False

# ── PRÉDICTION ───────────────────────────────────────────────
def run_prediction(model, features, rules, df_live):
    """Prédit sur la fenêtre la plus proche de l'heure actuelle."""
    now = pd.Timestamp.now().floor('5min')
    # Trouver la fenêtre la plus proche de maintenant
    diff = (df_live['window_start'] - now).abs()
    idx  = diff.idxmin()
    last_row = df_live.loc[idx]
    window_start = str(last_row['window_start'])

    # Construire le vecteur de features
    available = [f for f in features if f in df_live.columns]
    missing   = [f for f in features if f not in df_live.columns]

    X = pd.DataFrame([last_row[available].values], columns=available)
    for col in missing:
        X[col] = 0.0
    X = X[features]

    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    failure_type = "No_Failure"
    if pred == 1:
        failure_type = apply_rule_mapping(last_row, rules)

    return pred, proba, failure_type, window_start

# ── GRAPHIQUE HISTORIQUE ─────────────────────────────────────
def plot_history(hist_df):
    if len(hist_df) < 2:
        st.info("Historique insuffisant — lancez au moins 2 prédictions.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_df['timestamp'], y=hist_df['probability'],
        mode='lines+markers', name='Probabilité anomalie',
        line=dict(color='#3B82F6', width=2),
        marker=dict(
            color=np.where(hist_df['prediction'] == 1, '#EF4444', '#22C55E'),
            size=8, line=dict(color=DARK, width=1),
        ),
        hovertemplate='%{x|%H:%M}<br>Prob: %{y:.1%}<extra></extra>',
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="#F97316",
                  line_width=1.5, annotation_text="Seuil 50%",
                  annotation_font_color="#F97316")

    anom = hist_df[hist_df['prediction'] == 1]
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom['timestamp'], y=anom['probability'],
            mode='markers', name='ANORMAL détecté',
            marker=dict(color='#EF4444', size=12, symbol='x',
                        line=dict(color='#EF4444', width=2)),
            hovertemplate='%{x|%H:%M} — ANORMAL<br>Prob: %{y:.1%}<extra></extra>',
        ))

    fig.update_layout(
        title="Historique des probabilités d'anomalie",
        plot_bgcolor=DARK, paper_bgcolor=DARK, font_color=TEXT,
        xaxis=dict(gridcolor=CARD, title='Heure'),
        yaxis=dict(gridcolor=CARD, title='Probabilité', range=[0, 1],
                   tickformat='.0%'),
        legend=dict(bgcolor=CARD, bordercolor=BORD),
        hovermode='x unified', height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PAGE PRINCIPALE ──────────────────────────────────────────
def show_prediction_sechoir():
    _css()

    st.markdown("## 🤖 Prédiction Temps Réel — Séchoir Céramique")
    st.markdown(
        f"<div style='color:{TEXT};font-size:0.85rem;margin-bottom:1.5rem;'>"
        "XGBoost Binaire &nbsp;|&nbsp; Fenêtres 5 min &nbsp;|&nbsp;"
        f" Actualisation auto toutes les {REFRESH_INTERVAL} min</div>",
        unsafe_allow_html=True,
    )

    # ── Chargement modèle ────────────────────────────────────
    model, features, rules = load_model_assets()
    if model is None:
        st.error("Modèle non trouvé. Placez `xgboost_sechoir_binary.pkl`, "
                 "`feature_cols_sechoir.pkl` et `rule_mapping_sechoir.pkl` dans `src/`.")
        return

    # ── Session state ────────────────────────────────────────
    if 'pred_history'        not in st.session_state:
        st.session_state.pred_history        = load_history()
    if 'last_pred_time'      not in st.session_state:
        st.session_state.last_pred_time      = 0.0
    if 'last_result'         not in st.session_state:
        st.session_state.last_result         = None
    if 'mail_enabled'        not in st.session_state:
        st.session_state.mail_enabled        = True
    if 'last_window_start'   not in st.session_state:
        st.session_state.last_window_start   = None

    # ── Barre de contrôle ────────────────────────────────────
    col_btn, col_mail, col_next = st.columns([2, 2, 3])
    with col_btn:
        force_pred = st.button("🔄 Lancer la prédiction", use_container_width=True, type="primary")
    with col_mail:
        st.session_state.mail_enabled = st.toggle(
            "📧 Alertes mail", value=st.session_state.mail_enabled)
    with col_next:
        elapsed = time.time() - st.session_state.last_pred_time
        remaining = max(0, REFRESH_INTERVAL * 60 - elapsed)
        mins, secs = divmod(int(remaining), 60)
        st.markdown(
            f"<div style='color:{TEXT};font-size:0.8rem;padding-top:0.6rem;'>"
            f"Prochaine auto : <strong style='color:{WHITE};'>{mins:02d}:{secs:02d}</strong>"
            "</div>", unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Déclenchement prédiction ─────────────────────────────
    # auto_trigger : seulement si 30 min écoulées ET bouton non pressé
    elapsed      = time.time() - st.session_state.last_pred_time
    auto_trigger = elapsed >= (REFRESH_INTERVAL * 60)
    # Première fois uniquement (pas de boucle)
    first_load   = st.session_state.last_result is None
    should_predict = force_pred or (auto_trigger and not first_load) or first_load

    if should_predict:
        try:
            df_live = load_live_data()
            pred, proba, failure_type, window_start = run_prediction(
                model, features, rules, df_live)

            # Ne sauvegarder dans l'historique que si c'est une nouvelle fenêtre
            is_new_window = window_start != st.session_state.last_window_start

            mail_sent = False
            if pred == 1 and st.session_state.mail_enabled and is_new_window:
                mail_sent = send_alert_mail(proba, failure_type, window_start)

            result = {
                'prediction':        pred,
                'probability':       proba,
                'failure_type_pred': failure_type,
                'window_start':      window_start,
                'mail_sent':         mail_sent,
                'timestamp':         datetime.now(),
            }
            st.session_state.last_result      = result
            st.session_state.last_pred_time   = time.time()

            if is_new_window:
                st.session_state.last_window_start = window_start
                new_row = pd.DataFrame([{
                    'timestamp':         result['timestamp'],
                    'prediction':        pred,
                    'probability':       proba,
                    'failure_type_pred': failure_type,
                    'window_start':      window_start,
                    'mail_sent':         mail_sent,
                }])
                st.session_state.pred_history = pd.concat(
                    [st.session_state.pred_history, new_row], ignore_index=True)
                save_history(st.session_state.pred_history)

        except FileNotFoundError:
            st.error(f"Fichier de données introuvable : `{DATA_FILE.name}`. "
                     "Lancez d'abord `generate_sechoir_daily.py`.")
            return
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            return

    res = st.session_state.last_result
    if res is None:
        return

    # ── AFFICHAGE ÉTAT ───────────────────────────────────────
    pred     = res['prediction']
    proba    = res['probability']
    ft       = res['failure_type_pred']
    ts       = res['timestamp'].strftime('%H:%M:%S') if hasattr(res['timestamp'], 'strftime') else str(res['timestamp'])

    if pred == 0:
        st.markdown(f"""
        <div class="status-box status-normal">
            <div class="status-icon">🟢</div>
            <div class="status-label" style="color:#22c55e;">NORMAL</div>
            <div class="status-sub">Aucune anomalie prévue dans les 2 prochaines heures</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-box status-anormal">
            <div class="status-icon">🔴</div>
            <div class="status-label" style="color:#ef4444;">ANORMAL — HIGH</div>
            <div class="status-sub">Anomalie probable dans les 2 prochaines heures · Type : {ft}</div>
        </div>""", unsafe_allow_html=True)
        if res.get('mail_sent'):
            st.success("📧 Alerte mail envoyée au responsable maintenance.")
        elif st.session_state.mail_enabled:
            st.warning("📧 Mail non envoyé (vérifiez la configuration SMTP).")

    # ── KPIs ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Probabilité anomalie", f"{proba*100:.1f}%",
            "HIGH dans 2h", "#EF4444" if proba >= 0.5 else "#22C55E")
    with c2:
        kpi("Type de panne prédit", ft.replace("_"," "),
            "", "#F97316" if pred == 1 else TEXT)
    with c3:
        hist = st.session_state.pred_history
        n_anom = int((hist['prediction'] == 1).sum()) if len(hist) > 0 else 0
        kpi("Anomalies détectées", str(n_anom),
            f"sur {len(hist)} prédictions", "#EF4444" if n_anom > 0 else "#22C55E")
    with c4:
        kpi("Dernière mise à jour", ts, res['window_start'][:16] if res['window_start'] else "")

    st.markdown("---")

    # ── DÉTAIL FEATURES ──────────────────────────────────────
    with st.expander("🔎 Détail des features utilisées", expanded=False):
        try:
            df_live = load_live_data()
            last_row = df_live.iloc[-1]
            available = [f for f in features if f in df_live.columns]
            feat_vals = last_row[available].to_frame(name='Valeur').reset_index()
            feat_vals.columns = ['Feature', 'Valeur']
            col_l, col_r = st.columns(2)
            mid = len(feat_vals) // 2
            with col_l:
                st.dataframe(feat_vals.iloc[:mid], use_container_width=True, hide_index=True)
            with col_r:
                st.dataframe(feat_vals.iloc[mid:], use_container_width=True, hide_index=True)
        except Exception:
            st.info("Données live non disponibles.")

    # ── GRAPHIQUE HISTORIQUE ──────────────────────────────────
    st.markdown("### 📈 Historique des prédictions")
    hist_df = st.session_state.pred_history.copy()
    plot_history(hist_df)

    # ── TABLE HISTORIQUE ──────────────────────────────────────
    if not hist_df.empty:
        st.markdown("### 📋 Journal des prédictions")
        disp = hist_df.copy().sort_values('timestamp', ascending=False).head(50)
        disp['état'] = disp['prediction'].map({0: '🟢 Normal', 1: '🔴 Anormal'})
        disp['prob'] = (disp['probability'] * 100).round(1).astype(str) + '%'
        disp['mail'] = disp['mail_sent'].map({True: '✅', False: '—'})
        st.dataframe(
            disp[['timestamp', 'état', 'prob', 'failure_type_pred',
                  'window_start', 'mail']].reset_index(drop=True),
            use_container_width=True,
            column_config={
                'timestamp':        st.column_config.DatetimeColumn("Horodatage", format="DD/MM HH:mm:ss"),
                'état':             st.column_config.TextColumn("État"),
                'prob':             st.column_config.TextColumn("Probabilité"),
                'failure_type_pred':st.column_config.TextColumn("Type prédit"),
                'window_start':     st.column_config.TextColumn("Fenêtre"),
                'mail':             st.column_config.TextColumn("Mail"),
            },
        )

    # ── CONFIG MAIL ──────────────────────────────────────────
    with st.expander("⚙️ Configuration des alertes mail", expanded=False):
        st.markdown("Renseignez vos paramètres SMTP directement dans `page_prediction_sechoir.py` → `MAIL_CONFIG`.")
        st.code("""MAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port":   587,
    "sender":      "votre.email@gmail.com",
    "password":    "votre_app_password",   # mot de passe d'application Gmail
    "recipient":   "responsable@cjo.com.tn",
}""", language="python")
        st.info("Pour Gmail : activez l'authentification à 2 facteurs → "
                "Sécurité → Mots de passe des applications → générer un mot de passe.")

    # ── INFO prochaine actualisation ─────────────────────────
    elapsed_now = time.time() - st.session_state.last_pred_time
    remaining   = max(0, REFRESH_INTERVAL * 60 - elapsed_now)
    if remaining > 0:
        st.caption(f"Prochaine actualisation automatique dans "
                   f"{int(remaining//60):02d}:{int(remaining%60):02d} min "
                   f"— ou cliquez sur 🔄 pour forcer.")
