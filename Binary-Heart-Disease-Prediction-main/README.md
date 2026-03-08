# Binary Heart Disease Prediction (ECS 171)

## Quick start

### 1) Train / regenerate Model C artifacts (Model 3)
From the repo root:
py ecs171_project/ecs171model3.py
generate:
- `heart_disease_processed.csv`
- `ecs171_project/outputs_model3/model.pkl`
- `ecs171_project/outputs_model3/preprocess.pkl`
- other metric files

> No need to delete old files: rerunning will overwrite outputs.

### 2) Run the demo web app

```bash
cd ecs171_project/web
python app.py
```
Open: `http://127.0.0.1:5000/`

## Notes
- The Flask app loads `model.pkl` + `preprocess.pkl` from `ecs171_project/outputs_model3/`.
- If you move folders, keep relative structure the same.

