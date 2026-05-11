@echo off
:: ============================================================
:: Planifie alerter_sechoir.py toutes les 30 minutes
:: Exécuter EN TANT QU'ADMINISTRATEUR une seule fois
:: ============================================================

set PYTHON="%~dp0.venv\Scripts\python.exe"
set SCRIPT="%~dp0cjo-maintenance-src\alerter_sechoir.py"
set TASK_NAME=CJO_Alerter_Sechoir

echo.
echo  [1] Suppression de l'ancienne tache (si existante)...
schtasks /delete /tn "%TASK_NAME%" /f 2>nul

echo  [2] Creation de la tache planifiee...
schtasks /create ^
  /tn "%TASK_NAME%" ^
  /tr "%PYTHON% %SCRIPT%" ^
  /sc MINUTE ^
  /mo 30 ^
  /st 00:00 ^
  /ru "%USERNAME%" ^
  /rl HIGHEST ^
  /f

echo.
echo  [3] Verification...
schtasks /query /tn "%TASK_NAME%" /fo LIST

echo.
echo  ============================================================
echo   Tache planifiee : alerter_sechoir.py toutes les 30 min
echo   Pour la supprimer : schtasks /delete /tn %TASK_NAME% /f
echo  ============================================================
pause
