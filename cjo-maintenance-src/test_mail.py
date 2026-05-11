# test_mail.py — Diagnostic mail SMTP
import smtplib, sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# ── Copier ici les mêmes valeurs que dans alerter_sechoir.py ──
SENDER    = input("Email expéditeur Gmail : ").strip()
PASSWORD  = input("Mot de passe d'application (16 car.) : ").strip()
RECIPIENT = input("Email destinataire : ").strip()

print("\n[1] Connexion SMTP Gmail...")
try:
    server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
    server.ehlo()
    server.starttls()
    print("    TLS OK")
    server.login(SENDER, PASSWORD)
    print("    Authentification OK")
except smtplib.SMTPAuthenticationError:
    print("\n❌ ERREUR AUTHENTIFICATION")
    print("   Gmail exige un mot de passe d'application (pas votre mot de passe normal).")
    print("   → myaccount.google.com > Sécurité > Mots de passe des applications")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Erreur connexion : {e}")
    sys.exit(1)

print("\n[2] Envoi mail de test...")
msg = MIMEMultipart("alternative")
msg["Subject"] = "[CJO TEST] Mail de test — Alerter Séchoir"
msg["From"]    = SENDER
msg["To"]      = RECIPIENT
msg.attach(MIMEText(
    "<h2 style='color:#22c55e'>✅ Mail de test CJO reçu !</h2>"
    "<p>Le système d'alerte est correctement configuré.</p>",
    "html"
))

try:
    server.sendmail(SENDER, RECIPIENT, msg.as_string())
    server.quit()
    print(f"    ✅ Mail envoyé à {RECIPIENT}")
    print("\n    Vérifiez votre boîte (et les spams).")
except Exception as e:
    print(f"\n❌ Erreur envoi : {e}")
    sys.exit(1)
