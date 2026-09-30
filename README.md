# AI4I Predictive Maintenance

A Flask-based predictive-maintenance dashboard for the UCI AI4I 2020 dataset. The application supports single-machine inference, CSV batch prediction, model inspection, research metrics, model-predicted failure modes, and ranked maintenance guidance.

The interface runs locally or as a Python serverless application on Vercel.

## Features

- Single-machine prediction from five sensor readings
- Cost-aware and default failure decisions
- Model probability ring and risk bands
- Model-based failure-mode probabilities:
  - TWF — Tool Wear Failure
  - HDF — Heat Dissipation Failure
  - PWF — Power Failure
  - OSF — Overstrain Failure
  - RNF — Random Failure
- Ranked maintenance recommendations with:
  - Priority
  - Explanation
  - Evidence
  - Suggested next step
- Composite Machine Health Score (CMHS)
- Batch CSV prediction and downloadable results
- Original AI4I dataset column-name compatibility
- Research findings loaded from `research_summary.pkl`
- Vercel deployment configuration

## Project structure

```text
vehicle_health_advance/
├── app.py                         # Flask application and prediction API
├── api/
│   └── index.py                   # Vercel Flask entrypoint
├── templates/
│   └── index.html                 # Dashboard UI
├── dataset/
│   └── ai4i2020.csv               # Local training/reference dataset
├── results/                       # Research charts used by the project
├── best_model.pkl                 # Main binary failure model
├── failure_modes_model.pkl        # Five failure-mode models
├── scaler.pkl                     # Feature standardization artifact
├── feature_names.pkl              # Ordered model feature list
├── research_summary.pkl           # Research metrics and findings
├── train_deployment_model.py      # Retrains lightweight deployment artifacts
├── train_failure_modes.py         # Failure-mode training utility
├── requirements.txt               # Runtime dependencies
├── vercel.json                    # Vercel build and routing configuration
└── .vercelignore                  # Files excluded from Vercel deployment
```

## Requirements

- Python 3.10 or newer
- `pip`
- Node.js and npm only if deploying through the Vercel CLI
- Approximately 1 GB of temporary space for local scientific Python packages and model training

## Local setup

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Start the application:

```powershell
python app.py
```

Open:

```text
http://localhost:5000
```

The application listens on `0.0.0.0:5000` so it can also be accessed from another device on the same network when firewall rules permit it.

## Model pipeline

The application uses five raw measurements:

| Input | Unit | Description |
|---|---:|---|
| `Air_temperature_K` | K | Ambient air temperature |
| `Process_temperature_K` | K | Machine process temperature |
| `Rotational_speed_rpm` | rpm | Spindle or rotational speed |
| `Torque_Nm` | Nm | Applied torque |
| `Tool_wear_min` | min | Accumulated tool wear |

The application adds:

```text
Type_enc
Temp_Differential
Power_W
Wear_Torque
Mechanical_Stress
Thermal_Efficiency
Wear_per_Torque
Speed_Torque_Ratio
```

The final feature order is stored in `feature_names.pkl`. The scaler stored in `scaler.pkl` must be used with the same order.

### Main failure model

The main model returns the probability of `Machine failure = 1`.

The application exposes two decisions:

- **Cost-optimal decision:** probability is compared with `OPTIMAL_THR`, loaded from `research_summary.pkl`
- **Default decision:** probability is compared with `0.50`

The displayed risk level is derived from the model probability:

| Probability | Risk level |
|---:|---|
| `0%` to `<30%` | Low |
| `30%` to `<50%` | Medium |
| `50%` to `<70%` | High |
| `70%` to `100%` | Critical |

The probability and failure decisions are model outputs. Risk labels are probability bands applied after inference.

### Failure-mode models

`failure_modes_model.pkl` contains five independent binary classifiers. Each returns its own probability; the probabilities do not need to sum to 100%.

A mode is displayed as a predicted mode when its probability is at least 50%. The UI still displays all mode probabilities so lower-confidence signals remain visible.

## Recommended actions

Recommendations are model-informed action guidance rather than additional model predictions. They combine:

- Main failure probability
- Failure-mode probabilities
- Raw sensor readings
- Calculated power
- CMHS score

Each recommendation contains:

```text
priority
title
detail
reason
next_step
source
```

The recommendation engine can suggest actions such as:

- Controlled machine hold for critical overall risk
- Mechanical-load and work-zone inspection for OSF evidence
- Drive and power-delivery inspection for PWF evidence
- Tool inspection or replacement planning for TWF evidence
- Coolant, ventilation, and thermal-sensor checks for HDF evidence
- Sensor-quality and controller-log checks for RNF evidence
- Routine monitoring for low-risk conditions

Critical recommendations should always be confirmed by a qualified operator or maintenance engineer.

## API reference

### `GET /`

Returns the dashboard HTML.

### `POST /api/predict`

Accepts JSON:

```json
{
  "Type": "M",
  "Air_temperature_K": 300.0,
  "Process_temperature_K": 310.0,
  "Rotational_speed_rpm": 1500,
  "Torque_Nm": 40.0,
  "Tool_wear_min": 100
}
```

The response includes:

```json
{
  "failure_probability": 12.345,
  "prediction_optimal": 1,
  "prediction_default": 0,
  "optimal_threshold": 0.01,
  "risk_level": "Low",
  "cmhs": 0.8123,
  "cmhs_tier": "Healthy",
  "likely_modes": [],
  "failure_mode_scores": {
    "TWF": 1.2,
    "HDF": 0.8,
    "PWF": 0.5,
    "OSF": 2.1,
    "RNF": 3.4
  },
  "recommended_actions": [],
  "power_w": 6283.1
}
```

Invalid or non-finite sensor values return HTTP `400`.

### `POST /api/batch`

Accepts a multipart form upload:

```text
file=<CSV file>
```

The response includes aggregate statistics and the first 300 output rows.

### `POST /api/batch/download`

Accepts the same CSV upload and returns a downloadable CSV containing:

- Original input columns
- `Failure_Probability_%`
- `Prediction`
- `TWF_Probability_%`
- `HDF_Probability_%`
- `PWF_Probability_%`
- `OSF_Probability_%`
- `RNF_Probability_%`
- `Predicted_Failure_Modes`
- `CMHS`
- `CMHS_Tier`
- `Risk_Level`

### `GET /api/model_info`

Returns model type, feature names, threshold, scaler type, failure-mode model names, training-row count, and research metadata.

### `GET /api/sample/<kind>`

Available samples:

```text
healthy
tool_wear
heat
overstrain
power
```

## CSV format

The preferred application column names are:

```text
Type
Air_temperature_K
Process_temperature_K
Rotational_speed_rpm
Torque_Nm
Tool_wear_min
```

The original AI4I dataset column names are also accepted:

```text
Type
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

Additional columns are preserved in batch output. For example, the original columns `UDI`, `Product ID`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` remain available for comparison.

## Retraining deployment artifacts

The Vercel runtime intentionally uses lightweight scikit-learn gradient-boosting models instead of XGBoost. This keeps the serverless bundle below Vercel’s function-size limit.

To retrain the deployment artifacts using the local dataset:

```powershell
python train_deployment_model.py
```

This regenerates:

```text
best_model.pkl
failure_modes_model.pkl
scaler.pkl
```

The training script uses the existing `feature_names.pkl` feature order. Do not change the feature engineering logic or feature order without regenerating all dependent artifacts together.

`train_failure_modes.py` is retained as a separate failure-mode training utility. The deployment workflow uses `train_deployment_model.py`.

## Deploying to Vercel

The project already includes:

- `api/index.py`
- `vercel.json`
- `requirements.txt`
- `.vercelignore`

Install the Vercel CLI:

```powershell
npm install -g vercel
```

Log in:

```powershell
vercel login
```

Deploy a preview:

```powershell
cd "C:\Users\uncan\OneDrive\Desktop\vehicle_health_advance"
vercel
```

Deploy to production:

```powershell
vercel --prod
```

The Vercel configuration routes all paths to the Flask entrypoint:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

After deployment, test:

```text
https://your-project.vercel.app/
https://your-project.vercel.app/api/model_info
https://your-project.vercel.app/api/sample/healthy
```

### Vercel bundle-size note

Do not add XGBoost to `requirements.txt` for the deployment runtime. The XGBoost package significantly increases the serverless bundle and can exceed Vercel’s 500 MB function limit. The current deployment models use scikit-learn and retain the same prediction API.

If a deployment still reports an old bundle size, create a fresh deployment after confirming that:

```powershell
Get-Content requirements.txt
Get-Content .vercelignore
```

does not include XGBoost or excluded training assets.

## Troubleshooting

### `Not Found` after deployment

Confirm that `vercel.json` is in the project root and contains the `api/index.py` build and route. Then redeploy:

```powershell
vercel --prod
```

Use the latest deployment URL rather than an older preview URL.

### Missing artifact error

The application requires these files beside `app.py`:

```text
best_model.pkl
failure_modes_model.pkl
scaler.pkl
feature_names.pkl
research_summary.pkl
```

Do not exclude them in `.vercelignore`.

### Batch upload says columns are missing

Use either the normalized names or the original AI4I names documented in the CSV section. The backend automatically normalizes the original dataset names.

### Model import warning

Serialized machine-learning artifacts are generally most reliable when loaded with the same major package versions used to create them. If a model-loading warning appears after changing dependencies, regenerate the artifacts with:

```powershell
python train_deployment_model.py
```

### Local Windows console encoding error

The application uses ASCII startup messages to avoid code-page issues in Windows PowerShell. If other console output is added, prefer ASCII or run Python with UTF-8 mode:

```powershell
python -X utf8 app.py
```

## Validation checklist

Before deployment:

```powershell
python -B -c "import app; c=app.app.test_client(); print(c.get('/').status_code); print(c.get('/api/model_info').status_code)"
python -m py_compile app.py train_deployment_model.py
```

Expected output includes HTTP `200` for the page and model-info endpoint.

After deployment, verify:

1. The root page loads.
2. A sample scenario returns a prediction.
3. The model information tab loads.
4. A small custom CSV completes batch inference.
5. The original `dataset\ai4i2020.csv` completes batch inference.
6. The download action returns a CSV.

## Limitations

- This is a research and demonstration system, not a certified safety controller.
- Failure-mode classifiers are independent models; their probabilities are not mutually exclusive.
- CMHS is a domain diagnostic and not a learned probability.
- Recommended actions are model-informed guidance and require human confirmation.
- Vercel serverless functions are suitable for demonstrations and light workloads. High-volume inference may require a dedicated service.
- The trained models depend on the AI4I 2020 dataset and may not generalize to other machines without domain validation.

## License and data

The project uses the UCI AI4I 2020 Predictive Maintenance Dataset. Review the dataset’s original license and attribution requirements before redistributing the dataset or publishing a hosted copy.
