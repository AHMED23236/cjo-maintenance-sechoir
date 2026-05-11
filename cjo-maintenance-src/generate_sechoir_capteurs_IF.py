# ============================================================
# Génération dataset capteurs synthétiques — Séchoir
# Calibré sur données réelles Excel (14/05/2024)
# CJO Poulina — PFE Maintenance Prédictive
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

print("=" * 60)
print("  Génération capteurs synthétiques séchoir")
print("  Calibré sur données réelles SCADA CJO")
print("=" * 60)

# ── PARAMÈTRES CALIBRÉS SUR DONNÉES RÉELLES ──────────────────
# Extraits du fichier Excel modif.xlsx — Alarme_sechoir + évenement séchoir

# Températures normales séchoir (estimées d'après setpoints et codes alarmes)
TEMP_PARAMS = {
    'Temp_Module1': {'normal': 85,  'std': 5,  'setpoint': 90},
    'Temp_Module2': {'normal': 105, 'std': 6,  'setpoint': 110},
    'Temp_Module3': {'normal': 130, 'std': 8,  'setpoint': 135},  # Module critique H196/H198/H204
    'Temp_Bruleur3':{'normal': 145, 'std': 10, 'setpoint': 150},  # Brûleur 3 H220.02
    'Pression_EAU': {'normal': 0.3, 'std': 0.1,'setpoint': 0.5},
    'Vitesse_Tapis':{'normal': 1.2, 'std': 0.1,'setpoint': 1.3},
}

# Fréquences réelles des alarmes (observées dans le fichier)
# Sur 1 journée (14/05/2024) on a observé :
# H220.02 (Brûleur 3) : 5 fois → ~5/jour
# H200/H204 (Temp Module) : 3 fois → ~3/jour  
# H196/H198 (Gradient) : 4 fois → ~4/jour
# H386 (Mécanique) : ~30 fois → bruit opérationnel

ALARM_RATES = {
    'burner_failure':     0.10 / (24*60),   # par minute
    'temp_deviation':     0.07 / (24*60),
    'gradient_anomaly':   0.08 / (24*60),
}

# Durées réelles observées (minutes)
ALARM_DURATIONS = {
    'burner_failure':     {'mean': 6,   'std': 8},    # H220.02 : réduit pour équilibrer classes
    'temp_deviation':     {'mean': 25,  'std': 20},   # H200/H204
    'gradient_anomaly':   {'mean': 20,  'std': 25},   # H196/H198
}

# Événements brûleur (mise en manuel) : précèdent les pannes brûleur
# Observé : K3 mis en manuel 10:32 → H220.02 à 10:24 (déjà en panne)
# Donc brûleur en manuel = signal de pré-panne

# ── GÉNÉRATION TIMELINE 65 JOURS ─────────────────────────────
START_DATE = datetime(2026, 3, 1, 0, 0, 0)
END_DATE   = datetime(2026, 5, 5, 0, 0, 0)
FREQ_SEC   = 60  # 1 mesure par minute

print(f"\nPériode : {START_DATE.date()} → {END_DATE.date()}")

timestamps = []
t = START_DATE
while t < END_DATE:
    timestamps.append(t)
    t += timedelta(seconds=FREQ_SEC)

N = len(timestamps)
print(f"Nombre de points : {N} ({N//(60*24)} jours)")

df = pd.DataFrame({'timestamp': timestamps})
df['hour']        = df['timestamp'].apply(lambda x: x.hour)
df['day_of_week'] = df['timestamp'].apply(lambda x: x.weekday())
df['month']       = df['timestamp'].apply(lambda x: x.month)
df['minute_of_day'] = df['timestamp'].apply(lambda x: x.hour*60 + x.minute)

# ── GÉNÉRATION DES CAPTEURS DE BASE ──────────────────────────
print("\n[1] Génération des capteurs de base...")

for name, p in TEMP_PARAMS.items():
    # Signal de base avec variation jour/nuit
    base = p['normal']
    night_effect = np.where(
        (df['hour'] < 6) | (df['hour'] >= 22),
        -3, np.where(df['hour'] < 14, 2, 0)
    )
    # Bruit gaussien
    noise = np.random.normal(0, p['std'], N)
    # Drift lente (variation sur plusieurs jours)
    drift = 2 * np.sin(2 * np.pi * np.arange(N) / (7 * 24 * 60))
    
    df[name] = base + night_effect + noise + drift
    df[name] = df[name].clip(0, p['setpoint'] * 1.5)

# ── GÉNÉRATION DES ÉVÉNEMENTS DE PANNES ──────────────────────
print("[2] Génération des événements de pannes...")

# Initialiser les labels
df['failure_type']    = 'Normal'
df['alarm_active']    = 0
df['burner_manuel']   = 0  # Brûleur en mode manuel (précurseur)

# Générer les pannes aléatoires calibrées sur les vraies fréquences
def inject_failure(df, failure_type, rate, dur_mean, dur_std, sensor_affected, amplitude):
    """Injecte des pannes dans le dataset avec dérive progressive des capteurs."""
    i = 0
    while i < N:
        # Probabilité d'une panne à cette minute
        if np.random.random() < rate * 60:  # rate par minute
            # Durée de la panne
            dur = max(1, int(np.random.normal(dur_mean, dur_std)))
            dur = min(dur, 300)
            
            # Précurseur : dérive 30 min avant
            pre_start = max(0, i - 30)
            for pre_i in range(pre_start, i):
                progress = (pre_i - pre_start) / 30
                for sensor in sensor_affected:
                    if sensor in df.columns:
                        df.loc[pre_i, sensor] += amplitude * progress * 0.5
            
            # Panne active
            end_i = min(i + dur, N)
            df.loc[i:end_i, 'failure_type'] = failure_type
            df.loc[i:end_i, 'alarm_active'] = 1
            
            # Élévation des capteurs pendant la panne
            for j in range(i, end_i):
                for sensor in sensor_affected:
                    if sensor in df.columns:
                        df.loc[j, sensor] += amplitude * np.random.normal(1, 0.2)
            
            # Brûleur en manuel avant panne brûleur
            if failure_type == 'Burner_Failure':
                manuel_start = max(0, i - 15)
                df.loc[manuel_start:i, 'burner_manuel'] = 1
            
            i = end_i + np.random.randint(30, 120)  # pause entre pannes
        else:
            i += 1
    return df

# Injecter chaque type de panne
df = inject_failure(df, 'Burner_Failure',
                    rate=ALARM_RATES['burner_failure'],
                    dur_mean=ALARM_DURATIONS['burner_failure']['mean'],
                    dur_std=ALARM_DURATIONS['burner_failure']['std'],
                    sensor_affected=['Temp_Bruleur3', 'Temp_Module3'],
                    amplitude=20)

df = inject_failure(df, 'Temp_Deviation',
                    rate=ALARM_RATES['temp_deviation'],
                    dur_mean=ALARM_DURATIONS['temp_deviation']['mean'],
                    dur_std=ALARM_DURATIONS['temp_deviation']['std'],
                    sensor_affected=['Temp_Module1', 'Temp_Module2', 'Temp_Module3'],
                    amplitude=15)

df = inject_failure(df, 'Gradient_Anomaly',
                    rate=ALARM_RATES['gradient_anomaly'],
                    dur_mean=ALARM_DURATIONS['gradient_anomaly']['mean'],
                    dur_std=ALARM_DURATIONS['gradient_anomaly']['std'],
                    sensor_affected=['Temp_Module3', 'Temp_Bruleur3'],
                    amplitude=25)

# ── FEATURE ENGINEERING ──────────────────────────────────────
print("[3] Feature engineering...")

temp_sensors = ['Temp_Module1','Temp_Module2','Temp_Module3','Temp_Bruleur3']

# Rolling features
for sensor in temp_sensors:
    df[f'{sensor}_mean10'] = df[sensor].rolling(10, min_periods=1).mean()
    df[f'{sensor}_std10']  = df[sensor].rolling(10, min_periods=1).std().fillna(0)
    df[f'{sensor}_diff1']  = df[sensor].diff(1).fillna(0)
    df[f'{sensor}_diff5']  = df[sensor].diff(5).fillna(0)

# Déviation par rapport au setpoint
for name, p in TEMP_PARAMS.items():
    if name in df.columns:
        df[f'{name}_dev'] = df[name] - p['setpoint']

# Features globales
df['temp_mean_all'] = df[temp_sensors].mean(axis=1)
df['temp_max_all']  = df[temp_sensors].max(axis=1)
df['temp_std_all']  = df[temp_sensors].std(axis=1).fillna(0)
df['temp_range']    = df[temp_sensors].max(axis=1) - df[temp_sensors].min(axis=1)

# Features temporelles
df['is_night']    = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
df['is_morning']  = ((df['hour'] >= 6) & (df['hour'] < 14)).astype(int)
df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)

# Pression et vitesse rolling
df['Pression_EAU_mean10'] = df['Pression_EAU'].rolling(10, min_periods=1).mean()
df['Vitesse_Tapis_mean10'] = df['Vitesse_Tapis'].rolling(10, min_periods=1).mean()

# Label anomalie : panne dans les 30 prochaines minutes
df['anomaly_label'] = df['alarm_active'].rolling(5, min_periods=1).max().shift(-5).fillna(0).astype(int)

# ── STATISTIQUES ─────────────────────────────────────────────
print("\n[4] Vérification...")
print(f"  Total lignes     : {len(df)}")
print(f"  Nulls            : {df.isna().sum().sum()}")
print(f"\n  failure_type :")
for ft, n in df['failure_type'].value_counts().items():
    print(f"    {ft:25s} : {n:>6} ({n/len(df)*100:.1f}%)")
print(f"\n  anomaly_label :")
print(f"    Normal   : {(df['anomaly_label']==0).sum()} ({(df['anomaly_label']==0).mean()*100:.1f}%)")
print(f"    Anomalie : {(df['anomaly_label']==1).sum()} ({(df['anomaly_label']==1).mean()*100:.1f}%)")

# ── SAUVEGARDE ───────────────────────────────────────────────
OUT_PATH = Path(__file__).parent.parent / "cjo-maintenance-data-cleaned" / "sechoir_capteurs_IF.csv"

COLS = ['timestamp', 'hour', 'day_of_week', 'month', 'is_night', 'is_morning',
        'hour_sin', 'hour_cos',
        'Temp_Module1', 'Temp_Module2', 'Temp_Module3', 'Temp_Bruleur3',
        'Pression_EAU', 'Vitesse_Tapis',
        'Temp_Module1_mean10', 'Temp_Module1_std10', 'Temp_Module1_diff1', 'Temp_Module1_diff5',
        'Temp_Module2_mean10', 'Temp_Module2_std10', 'Temp_Module2_diff1', 'Temp_Module2_diff5',
        'Temp_Module3_mean10', 'Temp_Module3_std10', 'Temp_Module3_diff1', 'Temp_Module3_diff5',
        'Temp_Bruleur3_mean10','Temp_Bruleur3_std10','Temp_Bruleur3_diff1','Temp_Bruleur3_diff5',
        'Temp_Module1_dev', 'Temp_Module2_dev', 'Temp_Module3_dev', 'Temp_Bruleur3_dev',
        'temp_mean_all', 'temp_max_all', 'temp_std_all', 'temp_range',
        'Pression_EAU_mean10', 'Vitesse_Tapis_mean10',
        'burner_manuel',
        'failure_type', 'alarm_active', 'anomaly_label']

df_out = df[COLS].fillna(0)
df_out.to_csv(OUT_PATH, index=False)

print(f"\n✅ Fichier sauvegardé : {OUT_PATH.name}")
print(f"   {len(df_out)} lignes | {len(df_out.columns)} colonnes")
print("=" * 60)