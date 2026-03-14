import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import threading
import json
from flask import Flask, request, jsonify

st.set_page_config(page_title="FITCAP", page_icon="", layout="wide", initial_sidebar_state="collapsed")

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ['model_artifacts/linear_model.pkl','model_artifacts/knn_imputer.pkl',
             'model_artifacts/scaler.pkl','model_artifacts/feature_columns.pkl',
             'model_artifacts/thresholds.pkl']
    for f in files:
        if not os.path.exists(f):
            st.error(f"Missing: {f}"); st.stop()
    return (joblib.load('model_artifacts/linear_model.pkl'),
            joblib.load('model_artifacts/knn_imputer.pkl'),
            joblib.load('model_artifacts/scaler.pkl'),
            joblib.load('model_artifacts/feature_columns.pkl'),
            joblib.load('model_artifacts/thresholds.pkl'))

model, imputer, scaler, feature_columns, thresholds = load_artifacts()

# ── FLASK API (runs in background thread on port 5050) ────────────────────────
@st.cache_resource
def start_api():
    app = Flask(__name__)

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict_endpoint():
        # CORS headers so the iframe fetch() works
        if request.method == 'OPTIONS':
            r = jsonify({})
            r.headers['Access-Control-Allow-Origin'] = '*'
            r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            r.headers['Access-Control-Allow-Methods'] = 'POST'
            return r

        try:
            d = request.get_json()
            gender     = d['gender'].lower()
            age        = float(d['age'])
            height     = float(d['height'])
            weight     = float(d['weight'])
            duration   = float(d['duration'])
            heart_rate = float(d['heart_rate'])
            body_temp  = float(d['body_temp'])
            activity   = d['activity']

            df = pd.DataFrame([{
                "Gender": gender, "Age": age, "Height": height, "Weight": weight,
                "Duration": duration, "Heart_Rate": heart_rate,
                "Body_Temp": body_temp, "Activity_Type": activity
            }])

            if df.loc[0,'Body_Temp'] < thresholds['body_temp_lower']:
                df.loc[0,'Body_Temp'] = thresholds['body_temp_median']
            df['Mass_Duration'] = df['Weight'] * df['Duration']
            if df.loc[0,'Mass_Duration'] > thresholds['mass_duration_upper']:
                df.loc[0,'Mass_Duration'] = thresholds['mass_duration_median']

            dummies = pd.get_dummies(df[['Gender','Activity_Type']], drop_first=True, dtype=int)
            df = pd.concat([df.drop(['Gender','Activity_Type'], axis=1), dummies], axis=1)
            df = df.reindex(columns=feature_columns, fill_value=0)

            pred = float(model.predict(scaler.transform(imputer.transform(df)))[0])
            resp = jsonify({"prediction": round(pred, 1)})
        except Exception as e:
            resp = jsonify({"error": str(e)})

        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    t = threading.Thread(target=lambda: app.run(port=5050, debug=False, use_reloader=False), daemon=True)
    t.start()
    return True

start_api()

# ── HIDE STREAMLIT CHROME ─────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stHeader"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}
</style>
""", unsafe_allow_html=True)

# ── FULL PAGE ─────────────────────────────────────────────────────────────────
page = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{
  --teal:#0EB89D;--teal-lt:#1EDDB8;--blue:#1A6EFF;--blue-lt:#5B9DFF;--cyan:#00D4FF;
  --bg:#060D12;--bg-2:#0B1520;--bg-3:#0F1E2E;--bg-4:#152436;
  --line:#1E3448;--muted:#FFFFFF;--body:#FFFFFF;--white:#FFFFFF;
  --grad:linear-gradient(135deg,var(--teal),var(--blue));
}
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--body);font-family:'Instrument Sans',sans-serif;min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 70% 50% at 10% 0%,rgba(14,184,157,.07),transparent 55%),
             radial-gradient(ellipse 50% 40% at 90% 20%,rgba(26,110,255,.07),transparent 55%),
             radial-gradient(ellipse 40% 30% at 50% 80%,rgba(0,212,255,.04),transparent 50%);}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(14,184,157,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(14,184,157,.03) 1px,transparent 1px);
  background-size:48px 48px;}
.wrap{position:relative;z-index:1;max-width:1140px;margin:0 auto;padding:0 28px 100px;}
header{padding:52px 0 40px;display:flex;align-items:flex-end;justify-content:space-between;
       gap:24px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:48px;}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.3em;text-transform:uppercase;color:var(--teal);margin-bottom:10px;}
h1{font-family:'Syne',sans-serif;font-weight:800;font-size:clamp(3rem,8vw,6rem);line-height:1;letter-spacing:-.02em;color:var(--white);}
h1 span{background:linear-gradient(120deg,var(--teal-lt),var(--cyan),var(--blue-lt));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.badge{display:inline-flex;align-items:center;gap:7px;background:var(--bg-3);border:1px solid var(--line);border-radius:4px;padding:7px 14px;font-family:'IBM Plex Mono',monospace;font-size:.58rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;}
.dot{width:5px;height:5px;border-radius:50%;background:var(--teal);box-shadow:0 0 6px var(--teal);animation:blink 2.4s ease infinite;display:inline-block;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}
.subtitle{font-size:.8rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;letter-spacing:.04em;}
.stat-pills{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:48px;}
.stat-pill{background:var(--bg-2);border:1px solid var(--line);border-radius:8px;padding:14px 22px;display:flex;flex-direction:column;gap:4px;min-width:140px;}
.sp-val{font-family:'Syne',sans-serif;font-weight:700;font-size:1.5rem;color:var(--white);letter-spacing:-.02em;line-height:1;}
.sp-val.accent{background:linear-gradient(120deg,var(--teal-lt),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.sp-lbl{font-family:'IBM Plex Mono',monospace;font-size:.52rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);}
.main-grid{display:grid;grid-template-columns:1fr 360px;gap:28px;align-items:start;}
.panel,.result-card{background:var(--bg-2);border:1px solid var(--line);border-radius:14px;padding:32px;}
.result-card{position:sticky;top:24px;}
.panel-title{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.25em;text-transform:uppercase;color:var(--teal);margin-bottom:28px;display:flex;align-items:center;gap:10px;}
.panel-title::after{content:'';flex:1;height:1px;background:var(--line);}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;}
.field{display:flex;flex-direction:column;gap:8px;}
.field.full{grid-column:span 2;}
.flabel{font-family:'IBM Plex Mono',monospace;font-size:.56rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);}
.fsec{grid-column:span 2;font-family:'IBM Plex Mono',monospace;font-size:.5rem;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);display:flex;align-items:center;gap:10px;margin:4px 0;}
.fsec::after{content:'';flex:1;height:1px;background:var(--line);}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--bg-3);}
.seg-btn{flex:1;padding:11px 6px;border:none;background:transparent;color:var(--muted);font-family:'Instrument Sans',sans-serif;font-size:.85rem;font-weight:500;cursor:pointer;transition:all .18s;border-right:1px solid var(--line);}
.seg-btn:last-child{border-right:none;}
.seg-btn:hover{background:var(--bg-4);color:var(--body);}
.seg-btn.on{background:rgba(14,184,157,.16);color:var(--teal-lt);}
.stepper{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--bg-3);height:44px;}
.stepper-val{flex:1;display:flex;align-items:center;justify-content:center;gap:5px;border-left:1px solid var(--line);border-right:1px solid var(--line);font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:500;color:var(--white);}
.stepper-unit{font-size:.6rem;color:var(--muted);}
.step-btn{width:42px;border:none;background:transparent;color:var(--muted);font-size:1.3rem;cursor:pointer;font-family:monospace;transition:all .15s;flex-shrink:0;}
.step-btn:hover{background:var(--bg-4);color:var(--teal-lt);}
.step-btn:active{background:rgba(14,184,157,.1);}
.rng{display:flex;flex-direction:column;gap:6px;}
.rng-head{display:flex;justify-content:space-between;align-items:baseline;}
.rng-val{font-family:'IBM Plex Mono',monospace;font-size:.85rem;font-weight:500;color:var(--teal-lt);}
input[type=range]{-webkit-appearance:none;width:100%;height:4px;background:var(--line);border-radius:2px;outline:none;cursor:pointer;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--teal);border:2px solid var(--bg);box-shadow:0 0 8px rgba(14,184,157,.5);cursor:pointer;transition:transform .15s;}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2);}
.rng-bounds{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:var(--line);}
.cta{width:100%;margin-top:28px;background:var(--grad);border:none;color:#fff;font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;letter-spacing:.1em;text-transform:uppercase;padding:17px 24px;border-radius:10px;cursor:pointer;transition:transform .15s,box-shadow .2s;box-shadow:0 4px 24px rgba(14,184,157,.3);display:flex;align-items:center;justify-content:center;gap:10px;}
.cta:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(14,184,157,.45);}
.cta:active{transform:scale(.98);}
.cta:disabled{opacity:.55;cursor:not-allowed;transform:none;}
.empty-state{text-align:center;padding:48px 0;}
.empty-ring{width:84px;height:84px;border-radius:50%;border:2px dashed var(--line);margin:0 auto 18px;display:flex;align-items:center;justify-content:center;}
.empty-state p{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:var(--muted);letter-spacing:.1em;line-height:2;}
.cal-display{text-align:center;margin-bottom:28px;}
.cal-label{font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.3em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.cal-num{font-family:'Syne',sans-serif;font-weight:800;font-size:4.8rem;line-height:1;letter-spacing:-.03em;background:linear-gradient(135deg,var(--teal-lt),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.cal-unit{font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:var(--teal);letter-spacing:.2em;margin-top:4px;}
.mini-stats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:24px;}
.mini-stat{background:var(--bg-3);border:1px solid var(--line);border-radius:8px;padding:13px;}
.msv{font-family:'Syne',sans-serif;font-weight:700;font-size:1.4rem;color:var(--white);line-height:1;letter-spacing:-.02em;}
.msl{font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:var(--muted);letter-spacing:.15em;text-transform:uppercase;margin-top:5px;}
.int-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.int-title{font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);}
.int-label{font-family:'Syne',sans-serif;font-weight:700;font-size:.75rem;letter-spacing:.05em;text-transform:uppercase;}
.int-track{height:6px;background:var(--bg-4);border-radius:3px;overflow:hidden;margin-bottom:6px;}
.int-fill{height:100%;border-radius:3px;box-shadow:0 0 8px rgba(0,212,255,.3);transition:width .5s ease;}
.int-ticks{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:.48rem;color:var(--white);}
.err-box{color:#ff6b6b;text-align:center;font-family:'IBM Plex Mono',monospace;font-size:.8rem;padding:20px;background:var(--bg-3);border-radius:10px;line-height:1.8;}
.spinner{display:inline-block;width:15px;height:15px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
hr.div{border:none;border-top:1px solid var(--line);margin:48px 0;}
.sec-block{margin-top:48px;}
.sec-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;gap:16px;flex-wrap:wrap;}
.sec-title{font-family:'Syne',sans-serif;font-weight:700;font-size:1.25rem;color:var(--white);letter-spacing:-.01em;}
.sec-desc{font-size:.82rem;color:var(--muted);margin-top:3px;font-family:'IBM Plex Mono',monospace;letter-spacing:.03em;}
.tog{background:var(--bg-3);border:1px solid var(--line);color:var(--body);font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.15em;text-transform:uppercase;padding:9px 18px;border-radius:6px;cursor:pointer;transition:all .18s;white-space:nowrap;}
.tog:hover{border-color:var(--teal);color:var(--teal-lt);}
.tog.open{border-color:var(--teal);background:rgba(14,184,157,.1);color:var(--teal-lt);}
.tbl-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;display:none;}
.tbl-wrap.vis{display:block;}
table{width:100%;border-collapse:collapse;font-size:.78rem;}
thead tr{background:var(--bg-3);border-bottom:1px solid var(--line);}
thead th{font-family:'IBM Plex Mono',monospace;font-size:.52rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);padding:12px 14px;text-align:left;white-space:nowrap;}
tbody tr{border-bottom:1px solid var(--bg-4);transition:background .15s;}
tbody tr:last-child{border-bottom:none;}
tbody tr:hover{background:var(--bg-3);}
tbody td{padding:10px 14px;color:var(--body);font-size:.8rem;white-space:nowrap;}
tbody td.num{font-family:'IBM Plex Mono',monospace;color:var(--white);}
.tag{display:inline-block;padding:3px 9px;border-radius:4px;font-size:.62rem;font-family:'IBM Plex Mono',monospace;}
.tag-Running{background:rgba(14,184,157,.15);color:var(--teal-lt);}
.tag-Walking{background:rgba(91,157,255,.15);color:var(--blue-lt);}
.tag-High_Intensity{background:rgba(255,170,0,.12);color:#ffcc44;}
.ds-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px;}
.ds-card{background:var(--bg-3);border:1px solid var(--line);border-radius:8px;padding:16px;}
.ds-val{font-family:'Syne',sans-serif;font-weight:700;font-size:1.6rem;color:var(--white);line-height:1;letter-spacing:-.02em;}
.ds-lbl{font-family:'IBM Plex Mono',monospace;font-size:.5rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-top:5px;}
.hm-wrap{display:none;overflow-x:auto;}
.hm-wrap.vis{display:block;}
canvas#hm{display:block;width:100%;}
.pipe{display:flex;flex-direction:column;}
.pipe-item{display:flex;}
.pipe-con{display:flex;flex-direction:column;align-items:center;width:40px;flex-shrink:0;padding-top:4px;}
.pipe-dot{width:12px;height:12px;border-radius:50%;background:var(--bg-3);border:2px solid var(--teal);z-index:1;}
.pipe-dot.lit{background:var(--teal);box-shadow:0 0 10px rgba(14,184,157,.5);}
.pipe-item:hover .pipe-dot{background:var(--teal);}
.pipe-line{width:2px;background:var(--line);flex:1;min-height:24px;margin-top:4px;}
.pipe-body{flex:1;background:var(--bg-2);border:1px solid var(--line);border-radius:10px;padding:16px 22px;margin-bottom:10px;transition:border-color .2s;}
.pipe-item:hover .pipe-body{border-color:rgba(14,184,157,.3);}
.pipe-n{font-family:'IBM Plex Mono',monospace;font-size:.55rem;letter-spacing:.25em;text-transform:uppercase;color:var(--teal);margin-bottom:6px;}
.pipe-t{font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;color:var(--white);}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div>
    <div class="eyebrow">Fitness Calorie Prediction Engine</div>
    <h1>FIT<span>CAP</span></h1>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:10px;">
    <div class="badge"><div class="dot"></div>Model Active</div>
  </div>
</header>

<div class="stat-pills">
  <div class="stat-pill"><div class="sp-val accent">96.8%</div><div class="sp-lbl">R&sup2; Score</div></div>
  <div class="stat-pill"><div class="sp-val">15,000</div><div class="sp-lbl">Training Records</div></div>
  <div class="stat-pill"><div class="sp-val">9</div><div class="sp-lbl">Input Features</div></div>
</div>

<div class="main-grid">

  <div class="panel">
    <div class="panel-title">Workout Parameters</div>
    <div class="form-grid">

      <div class="field">
        <div class="flabel">Gender</div>
        <div class="seg">
          <button class="seg-btn on" onclick="setSeg('gender','Female',this)">&#9792; Female</button>
          <button class="seg-btn"    onclick="setSeg('gender','Male',this)">&#9794; Male</button>
        </div>
      </div>

      <div class="field">
        <div class="flabel">Activity Type</div>
        <div class="seg">
          <button class="seg-btn"    onclick="setSeg('activity','Walking',this)">Walk</button>
          <button class="seg-btn on" onclick="setSeg('activity','Running',this)">Run</button>
          <button class="seg-btn"    onclick="setSeg('activity','High_Intensity',this)">HIIT</button>
        </div>
      </div>

      <div class="fsec">Input Parameters</div>

      <div class="field">
        <div class="flabel">Age</div>
        <div class="stepper">
          <button class="step-btn" onclick="step('age',-1,10,100)">&#8722;</button>
          <div class="stepper-val"><span id="age_d">30</span><span class="stepper-unit">yrs</span></div>
          <button class="step-btn" onclick="step('age',1,10,100)">+</button>
        </div>
        <input type="hidden" id="age" value="30"/>
      </div>

      <div class="field">
        <div class="flabel">Height</div>
        <div class="stepper">
          <button class="step-btn" onclick="step('height',-1,100,250)">&#8722;</button>
          <div class="stepper-val"><span id="height_d">170</span><span class="stepper-unit">cm</span></div>
          <button class="step-btn" onclick="step('height',1,100,250)">+</button>
        </div>
        <input type="hidden" id="height" value="170"/>
      </div>

      <div class="field">
        <div class="flabel">Weight</div>
        <div class="stepper">
          <button class="step-btn" onclick="step('weight',-1,30,200)">&#8722;</button>
          <div class="stepper-val"><span id="weight_d">70</span><span class="stepper-unit">kg</span></div>
          <button class="step-btn" onclick="step('weight',1,30,200)">+</button>
        </div>
        <input type="hidden" id="weight" value="70"/>
      </div>

      <div class="field"></div>

      <div class="fsec">Workout Vitals</div>

      <div class="field full">
        <div class="rng">
          <div class="rng-head"><div class="flabel">Duration</div><span class="rng-val"><span id="dur_d">15</span> min</span></div>
          <input type="range" id="duration" min="1" max="30" value="15" oninput="rng(this,'dur_d',1)"/>
          <div class="rng-bounds"><span>1 min</span><span>30 min</span></div>
        </div>
      </div>

      <div class="field full">
        <div class="rng">
          <div class="rng-head"><div class="flabel">Heart Rate</div><span class="rng-val"><span id="hr_d">96</span> bpm</span></div>
          <input type="range" id="heartRate" min="67" max="128" value="96" oninput="rng(this,'hr_d',1)"/>
          <div class="rng-bounds"><span>67 bpm</span><span>128 bpm</span></div>
        </div>
      </div>

      <div class="field full">
        <div class="rng">
          <div class="rng-head"><div class="flabel">Body Temperature</div><span class="rng-val"><span id="temp_d">40.0</span> &deg;C</span></div>
          <input type="range" id="bodyTemp" min="37.1" max="41.5" value="40.0" step="0.1" oninput="rng(this,'temp_d',0.1)"/>
          <div class="rng-bounds"><span>37.1 &deg;C</span><span>41.5 &deg;C</span></div>
        </div>
      </div>

    </div>

    <button class="cta" id="calcBtn" onclick="doPredict()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
      Calculate
    </button>
  </div>

  <div class="result-card">
    <div class="panel-title">Predicted Output</div>
    <div class="empty-state" id="emptyState">
      <div class="empty-ring">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#1E3448" stroke-width="1.5">
          <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/>
        </svg>
      </div>
      <p>Enter your workout<br/>details and press<br/>Calculate to see results</p>
    </div>
    <div id="resultContent" style="display:none;"></div>
  </div>

</div>

<hr class="div"/>

<div class="sec-block">
  <div class="sec-head">
    <div><div class="sec-title">Dataset Overview</div></div>
    <button class="tog" id="dsBtn" onclick="togSec('dsTable','dsBtn')">Show Dataset</button>
  </div>
  <div class="ds-row">
    <div class="ds-card"><div class="ds-val">15,000</div><div class="ds-lbl">Total Records</div></div>
    <div class="ds-card"><div class="ds-val">10</div><div class="ds-lbl">Columns</div></div>
    <div class="ds-card"><div class="ds-val">645</div><div class="ds-lbl">Missing Values</div></div>
    <div class="ds-card"><div class="ds-val">89.5</div><div class="ds-lbl">Avg Calories</div></div>
    <div class="ds-card"><div class="ds-val">15.5</div><div class="ds-lbl">Avg Duration</div></div>
    <div class="ds-card"><div class="ds-val">50/50</div><div class="ds-lbl">F / M %</div></div>
  </div>
  <div class="tbl-wrap" id="dsTable">
    <table>
      <thead><tr><th>User ID</th><th>Calories</th><th>Gender</th><th>Age</th><th>Height</th><th>Weight</th><th>Duration</th><th>Heart Rate</th><th>Body Temp</th><th>Activity</th></tr></thead>
      <tbody id="dsBody"></tbody>
    </table>
  </div>
</div>

<hr class="div"/>

<div class="sec-block">
  <div class="sec-head">
    <div><div class="sec-title">Correlation Heatmap</div><div class="sec-desc">Correlation between features</div></div>
    <button class="tog" id="hmBtn" onclick="togSec('hmWrap','hmBtn','drawHM')">Show Heatmap</button>
  </div>
  <div class="hm-wrap" id="hmWrap"><div style="min-width:500px;"><canvas id="hm"></canvas></div></div>
</div>

<hr class="div"/>

<div class="sec-block">
  <div class="sec-head" style="margin-bottom:28px;">
    <div><div class="sec-title">Model Training Phases</div><div class="sec-desc">Step-by-step process</div></div>
  </div>
  <div class="pipe">
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 01</div><div class="pipe-t">Data Collection</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 02</div><div class="pipe-t">EDA</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 03</div><div class="pipe-t">Outlier Capping</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 04</div><div class="pipe-t">Feature Engineering &mdash; Mass &times; Duration</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 05</div><div class="pipe-t">Get dummies Encoding</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 06</div><div class="pipe-t">Train-Test Splitting</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 07</div><div class="pipe-t">KNN Imputation</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div><div class="pipe-line"></div></div><div class="pipe-body"><div class="pipe-n">Step 08</div><div class="pipe-t">Normalization</div></div></div>
    <div class="pipe-item"><div class="pipe-con"><div class="pipe-dot"></div></div><div class="pipe-body"><div class="pipe-n">Step 09</div><div class="pipe-t">Model Training</div></div></div>
  </div>
</div>

</div><!-- /wrap -->

<script>
var state = { gender:'Female', activity:'Running' };

function setSeg(group, val, btn) {
  state[group] = val;
  btn.closest('.seg').querySelectorAll('.seg-btn').forEach(function(b){ b.classList.remove('on'); });
  btn.classList.add('on');
}

function step(id, delta, mn, mx) {
  var h = document.getElementById(id);
  var d = document.getElementById(id+'_d');
  var v = Math.min(mx, Math.max(mn, parseInt(h.value) + delta));
  h.value = v; d.textContent = v;
}

function rng(el, spanId, precision) {
  var v = parseFloat(el.value);
  document.getElementById(spanId).textContent = precision < 1 ? v.toFixed(1) : v;
  var pct = ((v - parseFloat(el.min)) / (parseFloat(el.max) - parseFloat(el.min))) * 100;
  el.style.background = 'linear-gradient(to right,#0EB89D 0%,#1EDDB8 '+pct+'%,#1E3448 '+pct+'%)';
}

window.addEventListener('load', function() {
  rng(document.getElementById('duration'),  'dur_d',  1);
  rng(document.getElementById('heartRate'), 'hr_d',   1);
  rng(document.getElementById('bodyTemp'),  'temp_d', 0.1);
  buildTable();
});

function doPredict() {
  var btn = document.getElementById('calcBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Computing...';

  var payload = {
    gender:     state.gender,
    age:        parseFloat(document.getElementById('age').value),
    height:     parseFloat(document.getElementById('height').value),
    weight:     parseFloat(document.getElementById('weight').value),
    duration:   parseFloat(document.getElementById('duration').value),
    heart_rate: parseFloat(document.getElementById('heartRate').value),
    body_temp:  parseFloat(document.getElementById('bodyTemp').value),
    activity:   state.activity,
  };

  fetch('http://localhost:5050/predict', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(payload)
  })
  .then(function(r){ return r.json(); })
  .then(function(data) {
    btn.disabled = false;
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> Calculate';
    if (data.error) {
      showError(data.error);
    } else {
      showResult(data.prediction, payload);
    }
  })
  .catch(function(err) {
    btn.disabled = false;
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> Calculate';
    showError('API error: ' + err.message);
  });
}

function showError(msg) {
  document.getElementById('emptyState').style.display = 'none';
  var rc = document.getElementById('resultContent');
  rc.style.display = 'block';
  rc.innerHTML = '<div class="err-box">&#9888; ' + msg + '</div>';
}

function showResult(pred, inp) {
  document.getElementById('emptyState').style.display = 'none';
  var pct = Math.min((pred / 314) * 100, 100);
  var intensity, iColor;
  if      (pct <= 25) { intensity = 'Light';      iColor = '#5B9DFF'; }
  else if (pct <= 50) { intensity = 'Moderate';   iColor = '#0EB89D'; }
  else if (pct <= 75) { intensity = 'High';        iColor = '#00D4FF'; }
  else                { intensity = 'Max Effort';  iColor = '#FF6B6B'; }

  var act = inp.activity.replace('_',' ');
  var rc  = document.getElementById('resultContent');
  rc.style.display = 'block';
  rc.innerHTML =
    '<div class="cal-display">' +
      '<div class="cal-label">Estimated Calories Burned</div>' +
      '<div class="cal-num">' + pred.toFixed(1) + '</div>' +
      '<div class="cal-unit">KCAL</div>' +
    '</div>' +
    '<div class="mini-stats">' +
      '<div class="mini-stat"><div class="msv">' + inp.duration + '</div><div class="msl">Duration min</div></div>' +
      '<div class="mini-stat"><div class="msv">' + inp.heart_rate + '</div><div class="msl">Heart Rate</div></div>' +
      '<div class="mini-stat"><div class="msv">' + inp.body_temp.toFixed(1) + '</div><div class="msl">Body Temp &deg;C</div></div>' +
      '<div class="mini-stat"><div class="msv" style="font-size:.85rem;">' + act + '</div><div class="msl">Activity</div></div>' +
    '</div>' +
    '<div class="int-header">' +
      '<span class="int-title">Intensity Level</span>' +
      '<span class="int-label" style="color:#FFFFFF;">' + intensity + '</span>' +
    '</div>' +
    '<div class="int-track"><div class="int-fill" id="intFill" style="width:0%;background:linear-gradient(90deg,var(--teal),' + iColor + ');"></div></div>' +
    '<div class="int-ticks"><span>Light</span><span>Moderate</span><span>High</span><span>Max</span></div>';

  // Animate the intensity bar
  setTimeout(function(){ document.getElementById('intFill').style.width = pct.toFixed(1) + '%'; }, 50);
}

function togSec(contentId, btnId, cb) {
  var el  = document.getElementById(contentId);
  var btn = document.getElementById(btnId);
  var open = el.classList.contains('vis');
  if (open) {
    el.classList.remove('vis'); btn.classList.remove('open');
    btn.textContent = btn.textContent.replace('Hide','Show');
  } else {
    el.classList.add('vis'); btn.classList.add('open');
    btn.textContent = btn.textContent.replace('Show','Hide');
    if (cb) window[cb]();
  }
}

var ROWS=[{"User_ID":10001159,"Calories":76,"Gender":"female","Age":67,"Height":176,"Weight":74,"Duration":12,"Heart_Rate":103,"Body_Temp":39.6,"Activity_Type":"Running"},{"User_ID":10001607,"Calories":93,"Gender":"female","Age":34,"Height":178,"Weight":79,"Duration":19,"Heart_Rate":96,"Body_Temp":40.6,"Activity_Type":"Running"},{"User_ID":10005485,"Calories":49,"Gender":"female","Age":38,"Height":178,"Weight":"","Duration":14,"Heart_Rate":82,"Body_Temp":40.5,"Activity_Type":"Walking"},{"User_ID":10005630,"Calories":36,"Gender":"female","Age":39,"Height":169,"Weight":66,"Duration":8,"Heart_Rate":90,"Body_Temp":39.6,"Activity_Type":"Running"},{"User_ID":10006441,"Calories":122,"Gender":"male","Age":23,"Height":169,"Weight":73,"Duration":25,"Heart_Rate":102,"Body_Temp":40.7,"Activity_Type":"Running"},{"User_ID":10006606,"Calories":130,"Gender":"male","Age":50,"Height":183,"Weight":89,"Duration":23,"Heart_Rate":96,"Body_Temp":40.4,"Activity_Type":"Running"},{"User_ID":10007686,"Calories":30,"Gender":"female","Age":47,"Height":145,"Weight":47,"Duration":7,"Heart_Rate":84,"Body_Temp":39.6,"Activity_Type":"Walking"},{"User_ID":10008086,"Calories":129,"Gender":"male","Age":56,"Height":165,"Weight":74,"Duration":25,"Heart_Rate":93,"Body_Temp":40.8,"Activity_Type":"Running"},{"User_ID":10011832,"Calories":264,"Gender":"male","Age":62,"Height":196,"Weight":97,"Duration":30,"Heart_Rate":112,"Body_Temp":40.8,"Activity_Type":"Running"},{"User_ID":10014668,"Calories":89,"Gender":"male","Age":30,"Height":175,"Weight":78,"Duration":18,"Heart_Rate":99,"Body_Temp":40.3,"Activity_Type":"Running"}];

function buildTable() {
  var tb = document.getElementById('dsBody');
  if (tb.innerHTML) return;
  ROWS.forEach(function(r) {
    tb.innerHTML += '<tr><td class="num">'+r.User_ID+'</td><td class="num">'+r.Calories+'</td><td>'+r.Gender+'</td><td class="num">'+r.Age+'</td><td class="num">'+(r.Height||'—')+'</td><td class="num">'+(r.Weight||'—')+'</td><td class="num">'+r.Duration+'</td><td class="num">'+(r.Heart_Rate||'—')+'</td><td class="num">'+r.Body_Temp+'</td><td><span class="tag tag-'+r.Activity_Type+'">'+r.Activity_Type+'</span></td></tr>';
  });
}

var hmDone=false;
function drawHM(){
  if(hmDone)return; hmDone=true;
  var cv=document.getElementById('hm'),n=7,dpr=window.devicePixelRatio||1,cs=64;
  var p={t:52,l:92,r:24,b:40},W=p.l+n*cs+p.r,H=p.t+n*cs+p.b;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  var c=cv.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#0B1520';c.fillRect(0,0,W,H);
  var labels=['Calories','Age','Height','Weight','Duration','Heart Rate','Body Temp'];
  var matrix=[[1.0,0.154,0.016,0.035,0.955,0.898,0.825],[0.154,1.0,0.009,0.09,0.013,0.01,0.013],[0.016,0.009,1.0,0.958,-0.006,-0.003,0.0],[0.035,0.09,0.958,1.0,-0.002,0.003,0.004],[0.955,0.013,-0.006,-0.002,1.0,0.853,0.903],[0.898,0.01,-0.003,0.003,0.853,1.0,0.772],[0.825,0.013,0.0,0.004,0.903,0.772,1.0]];
  function cc(v){return v>=0?'rgba('+Math.round(14-14*v)+','+Math.round(184+28*v)+','+Math.round(157+98*v)+','+(0.15+v*0.85)+')'  :'rgba(26,110,255,'+(0.1+(-v)*0.8)+')';}
  matrix.forEach(function(row,i){row.forEach(function(v,j){
    var x=p.l+j*cs,y=p.t+i*cs;
    c.fillStyle=cc(v);c.beginPath();c.roundRect(x+2,y+2,cs-4,cs-4,4);c.fill();
    c.fillStyle=Math.abs(v)>0.4?'#E8F4F8':'#4A6278';
    c.font="500 11px 'IBM Plex Mono',monospace";c.textAlign='center';c.textBaseline='middle';
    c.fillText(v.toFixed(2),x+cs/2,y+cs/2);
  });});
  c.fillStyle='#7A9AB0';c.font="500 10.5px 'IBM Plex Mono',monospace";
  c.textAlign='center';c.textBaseline='bottom';
  labels.forEach(function(l,j){c.fillText(l,p.l+j*cs+cs/2,p.t-10);});
  c.textAlign='right';c.textBaseline='middle';
  labels.forEach(function(l,i){c.fillText(l,p.l-10,p.t+i*cs+cs/2);});
}
</script>
</body>
</html>"""

st.components.v1.html(page, height=4200, scrolling=False)
