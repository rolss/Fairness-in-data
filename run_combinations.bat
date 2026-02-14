@echo off
REM Batch script to run multiple combinations of the fairness pipeline
REM Make sure you have papermill installed: pip install papermill

echo Starting batch run of fairness pipeline...

REM Create results directory if it doesn't exist
if not exist "results" mkdir results

echo.
echo Running sex-race combination...
papermill "Reweight_COMPAS_BINARY XGB kfold.ipynb" "results/output_sex_race.ipynb" -p sensible_attribute "sex-race"

echo.
echo Running age-sex combination...
papermill "Reweight_COMPAS_BINARY XGB kfold.ipynb" "results/output_age_sex.ipynb" -p sensible_attribute "age-sex"

echo.
echo Running age-race combination...
papermill "Reweight_COMPAS_BINARY XGB kfold.ipynb" "results/output_age_race.ipynb" -p sensible_attribute "age-race"

echo.
echo Running age-sex-race combination...
papermill "Reweight_COMPAS_BINARY XGB kfold.ipynb" "results/output_age_sex_race.ipynb" -p sensible_attribute "age-sex-race"

echo.
echo All combinations completed! Results are in the results/ folder.
pause