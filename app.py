"""
AI4I 2020 Predictive Maintenance — Local Testing System
Flask backend: loads trained model, serves predictions via REST API
Run: python app.py
Then open: http://localhost:5000
"""

import os, pickle, io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS  (place pkl files in the same folder as this app, or ../outputs)
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [BASE, os.path.join(BASE, '..', 'outputs'), '/mnt/user-data/outputs']

def _load(name):
    for d in SEARCH_DIRS:
        path = os.path.join(d, name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Missing artifact: {name} — copy it next to app.py or into ../outputs")

MODEL    = _load('best_model.pkl')
SCALER   = _load('scaler.pkl')
FEATURES = _load('feature_names.pkl')          # 13-element list
MODELS   = _load('failure_modes_model.pkl')
try:
    RS = _load('research_summary.pkl')
    OPTIMAL_THR = float(RS['gap4_optimal_thr'])  # 0.010
except Exception:
    RS = {}
    OPTIMAL_THR = 0.01

print(f"[OK] Model loaded  ({type(MODEL).__name__})")
print(f"[OK] Scaler loaded ({type(SCALER).__name__})")
print(f"[OK] Features: {len(FEATURES)} columns")
print(f"[OK] Failure-mode models: {', '.join(MODELS['labels'])}")
print(f"[OK] Optimal threshold: {OPTIMAL_THR}")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE (identical to ai4i_research.py)
# ─────────────────────────────────────────────────────────────────────────────
RAW_NUMERIC = ['Air_temperature_K','Process_temperature_K',
               'Rotational_speed_rpm','Torque_Nm','Tool_wear_min']
CSV_COLUMN_ALIASES = {
    'Air temperature [K]': 'Air_temperature_K',
    'Process temperature [K]': 'Process_temperature_K',
    'Rotational speed [rpm]': 'Rotational_speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_wear_min',
}

TYPE_MAP = {'L': 0, 'M': 1, 'H': 2}   # matches LabelEncoder alphabetical order

SAFE_RANGES = {
    'Air_temperature_K':      (295.0, 304.5),
    'Process_temperature_K':  (305.7, 313.8),
    'Rotational_speed_rpm':   (1168,  2886),
    'Torque_Nm':              (3.8,   76.6),
    'Tool_wear_min':          (0,     200),
}

CMHS_WEIGHTS = {
    'Tool_wear_min':          0.30,
    'Torque_Nm':              0.25,
    'Air_temperature_K':      0.15,
    'Process_temperature_K':  0.15,
    'Rotational_speed_rpm':   0.15,
}

ACTION_MAP = {
    'Tool_wear_min':         {'high':(200, "CRITICAL: Tool wear exceeded safe limit — replace cutting tool immediately")},
    'Torque_Nm':              {'high':(65, "High torque detected — check workpiece material, feed rate, and tool condition"),
                               'low': (8,  "Abnormally low torque — inspect spindle and workpiece clamping")},
    'Rotational_speed_rpm':   {'high':(2500,"Overspeed detected — reduce rotational speed (OSF risk)"),
                               'low': (1200,"Below minimum speed — check drive motor and control system")},
    'Air_temperature_K':      {'high':(303, "Elevated ambient temperature — improve workshop ventilation (HDF risk)"),
                               'low': (296, "Low ambient temperature — allow warm-up period before operation")},
    'Process_temperature_K':  {'high':(312, "Process temperature high — check coolant flow and heat dissipation (HDF risk)"),
                               'low': (307, "Process temperature low — inspect heating system and thermal sensors")},
}

FAILURE_MODE_DESCRIPTIONS = {
    'TWF': 'Tool Wear Failure',
    'HDF': 'Heat Dissipation Failure',
    'PWF': 'Power Failure',
    'OSF': 'Overstrain Failure',
    'RNF': 'Random Failure',
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d['Temp_Differential']  = d['Process_temperature_K'] - d['Air_temperature_K']
    d['Power_W']            = d['Torque_Nm'] * (d['Rotational_speed_rpm'] * 2 * np.pi / 60)
    d['Wear_Torque']        = d['Tool_wear_min'] * d['Torque_Nm']
    d['Mechanical_Stress']  = d['Rotational_speed_rpm'] * d['Torque_Nm']
    d['Thermal_Efficiency'] = d['Process_temperature_K'] / d['Air_temperature_K']
    d['Wear_per_Torque']    = d['Tool_wear_min'] / (d['Torque_Nm'] + 0.001)
    d['Speed_Torque_Ratio'] = d['Rotational_speed_rpm'] / (d['Torque_Nm'] + 0.001)
    return d


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def compute_cmhs(row: dict) -> float:
    total_w = sum(CMHS_WEIGHTS.values())
    score = 0.0
    for feat, (lo, hi) in SAFE_RANGES.items():
        val  = float(row.get(feat, (lo+hi)/2))
        mid  = (lo+hi)/2.0
        half = (hi-lo)/2.0
        health = float(np.clip(1.0 - abs(val-mid)/half, 0.0, 1.0))
        score += health * CMHS_WEIGHTS[feat]
    return round(score/total_w, 4)


def cmhs_tier(cmhs: float) -> str:
    if cmhs >= 0.80: return 'Healthy'
    if cmhs >= 0.60: return 'Monitor'
    if cmhs >= 0.40: return 'At Risk'
    return 'Critical'


def predict_failure_modes(X_sc: np.ndarray) -> tuple[list, dict]:
    predictions = []
    scores = {}
    for label in MODELS['labels']:
        probability = float(MODELS['models'][label].predict_proba(X_sc)[0, 1])
        scores[label] = round(probability * 100, 3)
        if probability >= 0.50:
            predictions.append({
                'code': label,
                'name': FAILURE_MODE_DESCRIPTIONS[label],
                'probability': scores[label],
            })
    return sorted(predictions, key=lambda item: item['probability'], reverse=True), scores


def get_actions(row: dict, mode_scores: dict, failure_probability: float, cmhs: float) -> list:
    """Create ranked actions using model scores plus sensor evidence."""
    actions = []

    def add(priority, title, detail, reason, next_step, source='Model + sensor evidence'):
        actions.append({
            'priority': priority,
            'title': title,
            'detail': detail,
            'reason': reason,
            'next_step': next_step,
            'source': source,
        })

    failure_pct = failure_probability * 100
    if failure_pct >= 70:
        add('Immediate', 'Place the machine in a controlled hold',
            'Pause the current run or move it to a supervised operating state before the next cycle.',
            f'The primary failure model estimates {failure_pct:.1f}% failure probability.',
            'Confirm the condition with an operator, then inspect the highest-scoring mode first.')
    elif failure_pct >= 50:
        add('High', 'Schedule a maintenance inspection this shift',
            'Do not defer the inspection to the next planned service window.',
            f'The primary failure model estimates {failure_pct:.1f}% failure probability.',
            'Review the mode probabilities below and assign the matching inspection checklist.')
    elif failure_pct >= 10:
        add('Monitor', 'Increase observation frequency',
            'Keep the machine running only with closer observation and shorter review intervals.',
            f'The primary failure model estimates {failure_pct:.1f}% failure probability.',
            'Recheck the sensor readings after the next production cycle.')

    mode = lambda code: float(mode_scores.get(code, 0))
    if mode('TWF') >= 20 or float(row['Tool_wear_min']) >= 160:
        twf_source = 'TWF model + Tool_wear_min' if mode('TWF') >= 20 else 'Sensor diagnostic'
        add('High' if mode('TWF') >= 50 else 'Monitor',
            'Inspect and plan the cutting-tool change',
            'Check insert edge condition, tool seating, chip evacuation, and wear measurement.',
            f'TWF model score is {mode("TWF"):.1f}% and tool wear is {row["Tool_wear_min"]:.1f} min.',
            'Replace the tool if edge damage or the wear limit is confirmed.',
            twf_source)
    if mode('HDF') >= 20 or float(row['Process_temperature_K']) > 312:
        hdf_source = 'HDF model + thermal sensors' if mode('HDF') >= 20 else 'Sensor diagnostic'
        add('High' if mode('HDF') >= 50 else 'Monitor',
            'Check heat dissipation before continuing at full rate',
            'Inspect coolant flow, filters, ventilation, thermal load, and process-temperature sensing.',
            f'HDF model score is {mode("HDF"):.1f}% and process temperature is {row["Process_temperature_K"]:.1f} K.',
            'Verify that process temperature returns toward the normal band under a controlled test.',
            hdf_source)
    if mode('PWF') >= 20:
        add('High' if mode('PWF') >= 50 else 'Monitor',
            'Inspect the drive and power-delivery path',
            'Check motor current, drive alarms, cable connections, supply stability, and spindle response.',
            f'PWF model score is {mode("PWF"):.1f}%; calculated power is {row.get("Power_W", 0):.0f} W.',
            'Run a supervised no-load and low-load test before returning to production.',
            'PWF model + power features')
    if mode('OSF') >= 20 or float(row['Torque_Nm']) > 65:
        osf_source = 'OSF model + Torque_Nm' if mode('OSF') >= 20 else 'Sensor diagnostic'
        add('Immediate' if mode('OSF') >= 70 else 'High',
            'Reduce mechanical load and inspect the work zone',
            'Check workpiece clamping, feed rate, tool condition, spindle alignment, and unexpected obstruction.',
            f'OSF model score is {mode("OSF"):.1f}% with torque at {row["Torque_Nm"]:.1f} Nm.',
            'Do not resume full-load operation until the cause of excess torque is cleared.',
            osf_source)
    if mode('RNF') >= 20:
        add('Monitor',
            'Verify sensor quality and operating context',
            'Look for intermittent readings, missing events, calibration drift, or an unusual operating sequence.',
            f'RNF model score is {mode("RNF"):.1f}%, indicating a less specific failure signature.',
            'Repeat the measurement and compare it with the controller event log.',
            'RNF model + input validation')

    if float(row['Air_temperature_K']) > 303:
        add('Monitor', 'Improve ambient cooling around the machine',
            'Increase local ventilation and check whether nearby heat sources are affecting the reading.',
            f'Air temperature is {row["Air_temperature_K"]:.1f} K, above the preferred operating band.',
            'Confirm the ambient reading with an independent thermometer.',
            'Sensor diagnostic')
    if float(row['Rotational_speed_rpm']) > 2500:
        add('High', 'Review spindle speed and overspeed protection',
            'Confirm the commanded speed, controller limits, and spindle feedback.',
            f'Rotational speed is {row["Rotational_speed_rpm"]:.0f} rpm.',
            'Reduce speed to the approved process setting before the next cycle.',
            'Sensor diagnostic')

    if not actions:
        if cmhs >= 0.80:
            add('Routine', 'Continue normal operation with routine checks',
                'The model shows low failure risk and the composite health score is healthy.',
                f'Failure probability is {failure_pct:.1f}% and CMHS is {cmhs:.4f}.',
                'Keep standard inspection and calibration intervals.')
        else:
            add('Monitor', 'Keep the machine under routine observation',
                'No specific failure mode crossed the action threshold, but the health score merits attention.',
                f'CMHS is {cmhs:.4f}; no mode probability exceeded the action threshold.',
                'Recheck the machine during the next scheduled inspection.')

    priority_order = {'Immediate': 0, 'High': 1, 'Monitor': 2, 'Routine': 3}
    return sorted(actions, key=lambda item: priority_order[item['priority']])[:8]


def predict_single(raw_row: dict, product_type: str) -> dict:
    df = pd.DataFrame([raw_row])[RAW_NUMERIC]
    df_eng = engineer_features(df)
    df_eng['Type_enc'] = TYPE_MAP.get(product_type, 0)

    X    = df_eng[FEATURES]
    X_sc = SCALER.transform(X)

    prob     = float(MODEL.predict_proba(X_sc)[0, 1])
    failure_modes, failure_mode_scores = predict_failure_modes(X_sc)
    pred_opt = int(prob >= OPTIMAL_THR)
    pred_def = int(prob >= 0.50)

    full_row = dict(raw_row)
    full_row['Power_W'] = float(df_eng['Power_W'].iloc[0])

    cmhs    = compute_cmhs(raw_row)
    tier    = cmhs_tier(cmhs)
    actions = get_actions(raw_row, failure_mode_scores, prob, cmhs)
    risk = ('Critical' if prob > 0.70 else
            'High'     if prob > 0.50 else
            'Medium'   if prob > 0.30 else
            'Low')

    return {
        'failure_probability': round(prob * 100, 3),
        'prediction_optimal':  pred_opt,
        'prediction_default':  pred_def,
        'optimal_threshold':   OPTIMAL_THR,
        'risk_level':          risk,
        'cmhs':                cmhs,
        'cmhs_tier':           tier,
        'likely_modes':        failure_modes,
        'failure_mode_scores': failure_mode_scores,
        'recommended_actions': actions,
        'power_w':             round(full_row['Power_W'], 1),
        'prediction_sources': {
            'failure_probability': 'XGBoost binary failure model',
            'failure_modes': 'XGBoost failure-mode models',
            'risk_level': 'Probability bands derived from failure model',
            'cmhs': 'Domain diagnostic, not a model prediction',
            'recommended_actions': 'Model-informed action guidance using mode scores and sensor evidence',
        },
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if 'Type' not in df_clean.columns:
        df_clean['Type'] = 'M'
    df_clean['Type_enc'] = df_clean['Type'].map(TYPE_MAP).fillna(1).astype(int)

    df_eng = engineer_features(df_clean)
    X = df_eng[FEATURES]
    X_sc = SCALER.transform(X)

    probs = MODEL.predict_proba(X_sc)[:, 1]
    mode_probs = {
        label: MODELS['models'][label].predict_proba(X_sc)[:, 1]
        for label in MODELS['labels']
    }
    preds = (probs >= OPTIMAL_THR).astype(int)

    df_out = df.copy()
    df_out['Failure_Probability_%'] = (probs * 100).round(3)
    df_out['Prediction']            = preds
    for label, values in mode_probs.items():
        df_out[f'{label}_Probability_%'] = (values * 100).round(3)
    df_out['Predicted_Failure_Modes'] = [
        ', '.join(
            label for label in MODELS['labels']
            if mode_probs[label][index] >= 0.50
        ) or 'None'
        for index in range(len(df_out))
    ]
    df_out['CMHS'] = [
        compute_cmhs(row) for _, row in df_clean[RAW_NUMERIC].iterrows()
    ]
    df_out['CMHS_Tier'] = df_out['CMHS'].apply(cmhs_tier)
    df_out['Risk_Level'] = pd.cut(
        probs, bins=[-0.001, 0.30, 0.50, 0.70, 1.001],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    return df_out


def normalize_batch_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns=CSV_COLUMN_ALIASES)
    missing = [column for column in RAW_NUMERIC if column not in normalized.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    return normalized

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           raw_features=RAW_NUMERIC,
                           safe_ranges=SAFE_RANGES,
                           optimal_thr=OPTIMAL_THR,
                           research=RS)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    try:
        raw = {f: float(data[f]) for f in RAW_NUMERIC}
        product_type = data.get('Type', 'M')
        if product_type not in TYPE_MAP:
            raise ValueError(f"Type must be one of {', '.join(TYPE_MAP)}")
        if not all(np.isfinite(value) for value in raw.values()):
            raise ValueError('Sensor values must be finite numbers')
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Missing or invalid field: {e}'}), 400

    result = predict_single(raw, product_type)
    result['input'] = raw
    result['type']  = product_type
    return jsonify(result)


@app.route('/api/batch', methods=['POST'])
def api_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Could not parse CSV: {e}'}), 400

    try:
        df_out = predict_batch(normalize_batch_columns(df))
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid sensor values: {e}'}), 400
    n_total = len(df_out)
    n_fail  = int((df_out['Prediction'] == 1).sum())
    n_crit  = int((df_out['Risk_Level'] == 'Critical').sum())

    preview = df_out.head(300).to_dict(orient='records')

    return jsonify({
        'total_machines':     n_total,
        'predicted_failures': n_fail,
        'critical_risk':      n_crit,
        'failure_rate_%':     round(n_fail / n_total * 100, 2) if n_total else 0.0,
        'preview':            preview,
        'prediction_sources': {
            'failure_probability': 'XGBoost binary failure model',
            'failure_modes': 'XGBoost failure-mode models',
            'risk_level': 'Probability bands derived from failure model',
            'cmhs': 'Domain diagnostic, not a model prediction',
            'recommended_actions': 'Domain rules, not a model prediction',
        },
    })


@app.route('/api/batch/download', methods=['POST'])
def api_batch_download():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    try:
        df_out = predict_batch(normalize_batch_columns(df))
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid sensor values: {e}'}), 400
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='ai4i_predictions.csv'
    )


@app.route('/api/model_info')
def api_model_info():
    model_params = MODEL.get_params()

    return jsonify({
        'model_type': type(MODEL).__name__,
        'n_estimators': int(model_params.get('n_estimators', model_params.get('max_iter', 0))),
        'max_depth': model_params.get('max_depth', 'n/a'),
        'learning_rate': float(model_params.get('learning_rate', 0.0)),
        'features': make_json_safe(FEATURES),
        'n_features': len(FEATURES),
        'optimal_threshold': float(OPTIMAL_THR),
        'scaler': type(SCALER).__name__,
        'failure_mode_models': MODELS['labels'],
        'failure_mode_training_rows': int(MODELS['training_rows']),
        'research': make_json_safe(RS)
    })

@app.route('/api/sample/<kind>')
def api_sample(kind):
    samples = {
        'healthy': {'Type':'M','Air_temperature_K':299.8,'Process_temperature_K':309.5,
                    'Rotational_speed_rpm':1550,'Torque_Nm':38.5,'Tool_wear_min':85},
        'tool_wear': {'Type':'L','Air_temperature_K':300.5,'Process_temperature_K':310.2,
                      'Rotational_speed_rpm':1400,'Torque_Nm':48.0,'Tool_wear_min':215},
        'heat': {'Type':'L','Air_temperature_K':303.2,'Process_temperature_K':312.8,
                 'Rotational_speed_rpm':1310,'Torque_Nm':42.0,'Tool_wear_min':95},
        'overstrain': {'Type':'L','Air_temperature_K':300.1,'Process_temperature_K':309.9,
                       'Rotational_speed_rpm':1320,'Torque_Nm':68.0,'Tool_wear_min':180},
        'power': {'Type':'M','Air_temperature_K':300.0,'Process_temperature_K':309.0,
                  'Rotational_speed_rpm':1280,'Torque_Nm':64.0,'Tool_wear_min':110},
    }
    if kind not in samples:
        return jsonify({'error': 'unknown sample'}), 404
    return jsonify(samples[kind])


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  AI4I 2020 Predictive Maintenance — Local Test System")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
