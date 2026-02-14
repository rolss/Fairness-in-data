@echo off
REM Batch script to run multiple combinations of the fairness pipeline

REM Change to the script's directory
cd /d "%~dp0"

echo Starting batch run of fairness pipeline...

REM Create results directory if it doesn't exist
if not exist "results" mkdir results
if not exist "results/COMPAS" mkdir results/COMPAS

echo.
echo Running sex-race combination...
papermill "Reweight_COMPAS_BINARY Model XGB NO THRESHOLD.ipynb" "results/COMPAS/output_sex_race_XGB_COMPAS_nt.ipynb" -p sensible_attribute "sex-race"

echo.
echo Running age-race combination...
papermill "Reweight_COMPAS_BINARY Model XGB NO THRESHOLD.ipynb" "results/COMPAS/output_age_race_XGB_COMPAS_nt.ipynb" -p sensible_attribute "age-race"

echo.
echo Running age-sex combination...
papermill "Reweight_COMPAS_BINARY Model XGB NO THRESHOLD.ipynb" "results/COMPAS/output_age_sex_XGB_COMPAS_nt.ipynb" -p sensible_attribute "age-sex"

echo.
echo All combinations completed! Results are in the results/COMPAS/ folder.
exit 0