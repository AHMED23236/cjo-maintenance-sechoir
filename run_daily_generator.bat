@echo off
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\msi\Desktop\cjo maintenance\cjo-maintenance-src"
python generate_sechoir_daily.py >> "C:\Users\msi\Desktop\cjo maintenance\cjo-maintenance-data-cleaned\generator_log.txt" 2>&1
