# Risk Prediction Engine (End-to-End)

This project builds a **30–180 day history → 90 day risk** pipeline using your four CSVs:
- `Patient_info.csv`
- `Biochemical_parameters.csv`
- `Diagnostics.csv`
- `Glucose_measurements.csv` (big; processed in chunks into daily aggregates)

## Quick start

1) Create a virtualenv and install requirements:
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

2) Open and run the notebook:
```
jupyter lab
# open RiskEngine_EndToEnd.ipynb
```

3) (Optional) Launch the demo dashboard **after** running the notebook once (so the model & features exist):
```
cd app
streamlit run streamlit_app.py
```

## Outputs
- `/mnt/data/risk_engine/outputs/` will contain metrics, plots, and the trained model for quick reuse.
