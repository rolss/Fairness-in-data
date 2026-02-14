@echo off
REM Batch script to run multiple combinations of the fairness pipeline

REM Change to the script's directory
cd /d "%~dp0"

echo Starting batch run of fairness pipeline...

REM Create results directory if it doesn't exist
if not exist "results" mkdir results
if not exist "results/Census" mkdir results/Census

echo.
echo Running sex-race combination...
papermill "Reweight_Census_BINARY Model XGB.ipynb" "results/Census/output_sex_race_XGB_Census.ipynb" -p sensible_attribute "sex-race"

echo.
echo Running age-race combination...
papermill "Reweight_Census_BINARY Model XGB.ipynb" "results/Census/output_age_race_XGB_Census.ipynb" -p sensible_attribute "age-race"

echo.
echo Running age-sex combination...
papermill "Reweight_Census_BINARY Model XGB.ipynb" "results/Census/output_age_sex_XGB_Census.ipynb" -p sensible_attribute "age-sex"

echo.
echo Running sex-edu combination...
papermill "Reweight_Census_BINARY Model XGB.ipynb" "results/Census/output_sex_edu_XGB_Census.ipynb" -p sensible_attribute "sex-edu"

echo.
echo Running age-edu combination...
papermill "Reweight_Census_BINARY Model XGB.ipynb" "results/Census/output_age_edu_XGB_Census.ipynb" -p sensible_attribute "age-edu"

echo.
echo Running race-edu combination...
papermill "Reweight_Census_BINARY Model XGB.ipynb" "results/Census/output_race_edu_XGB_Census.ipynb" -p sensible_attribute "race-edu"

echo.
echo All combinations completed! Results are in the results/Census/ folder.
exit 0