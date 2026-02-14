@echo off
REM Batch script to run multiple combinations of the fairness pipeline

REM Change to the script's directory
cd /d "%~dp0"

echo Starting batch run of fairness pipeline...

REM Create results directory if it doesn't exist
if not exist "results" mkdir results

echo.
echo Running sex-race combination...
papermill "Reweight_Census_BINARY Model XGB.ipynb" "results/output_sex_race.ipynb" -p sensible_attribute "sex-race"

echo.
echo All combinations completed! Results are in the results/ folder.
exit 0