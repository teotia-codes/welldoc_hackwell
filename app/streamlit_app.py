# Streamlit Risk Prediction Dashboard (Refined & Fixed)
# -------------------------------------------------
# Compact, clinician-friendly UI with: cohort overview, patient detail,
# reliability (calibration/DCA), explainability (global + local), and
# action recommendations. Designed for non-technical users.
#
# Run:
#   pip install streamlit scikit-learn xgboost shap pandas pyarrow matplotlib altair
#   python -m streamlit run app/streamlit_app.py

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

# Optional deps
try:
    import joblib
except Exception:
    joblib = None

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ------------------------------
# Paths & basic setup
# ------------------------------
BASE = Path(__file__).resolve().parent.parent  # repo root
OUT = BASE / "outputs"
PLOTS = OUT / "plots"
MODEL_PATH = OUT / "model_calibrated.joblib"
DATASET_PATH = OUT / "feature_dataset.parquet"
METRICS_PATH = OUT / "metrics.json"

st.set_page_config(page_title="90-Day Risk Dashboard", layout="wide")

# Simple color scale & bands
RISK_BANDS = [(0.0, 0.30, "Green (Low)"), (0.30, 0.60, "Yellow (Medium)"), (0.60, 1.01, "Red (High)")]
BAND_ORDER = ["Green (Low)", "Yellow (Medium)", "Red (High)"]
BAND_COLOR = {"Green (Low)": "#d4edda", "Yellow (Medium)": "#fff3cd", "Red (High)": "#f8d7da"}

# ------------------------------
# Loaders
# ------------------------------
@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    for c in ["date", "index_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if joblib is None or not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_metrics(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

# ------------------------------
# Helpers
# ------------------------------
def band_name(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "Unknown"

@st.cache_data(show_spinner=False)
def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values(["Patient_ID", "index_date"])
          .groupby("Patient_ID")
          .tail(1)
          .reset_index(drop=True)
    )

# NOTE: underscore in _model so Streamlit won't hash it
@st.cache_data(show_spinner=False)
def score_df(df: pd.DataFrame, _model):
    if df.empty:
        return df
    out = df.copy()
    if _model is None:
        out["risk"] = np.nan
        out["risk_band"] = "Unknown"
        return out
    feature_cols = [c for c in df.columns if c not in
                    ["Patient_ID", "date", "index_date", "label_90d", "a1c_prior", "a1c_future"]]
    try:
        proba = _model.predict_proba(df[feature_cols])[:, 1]
    except Exception:
        proba = _model.predict(df[feature_cols]).astype(float)
        proba = np.clip(proba, 0, 1)
    out["risk"] = proba
    out["risk_band"] = out["risk"].apply(band_name)
    return out

# ------------------------------
# Load artifacts
# ------------------------------
df_all = load_parquet(DATASET_PATH)
model = load_model(MODEL_PATH)
metrics = load_metrics(METRICS_PATH)

st.title("🩺 90-Day Deterioration Risk")
st.caption("Simple, explainable predictions to help plan care. Green = Low, Yellow = Medium, Red = High.")

if df_all.empty:
    st.error("Feature dataset not found. Run the training notebook to create 'outputs/feature_dataset.parquet'.")
    st.stop()

# ------------------------------
# Sidebar: simple language controls
# ------------------------------
with st.sidebar:
    st.header("Controls")
    thr = st.slider("Alert threshold", 0.05, 0.95, 0.40, 0.05,
                    help="Patients at or above this score are flagged for review.")
    cap = st.number_input("Max alerts to show", min_value=0, value=50, step=5,
                          help="Limit how many patients are flagged today.")

    # Safe option lists even if columns are missing
    if "Sex" in df_all.columns:
        sex_opts = sorted(df_all["Sex"].fillna("Unknown").astype(str).unique().tolist())
    else:
        sex_opts = ["Unknown"]
    sex_sel = st.multiselect("Sex filter", sex_opts, default=sex_opts)

    if "Birth_year" in df_all.columns and df_all["Birth_year"].dropna().size > 0:
        miny = int(df_all["Birth_year"].dropna().min())
        maxy = int(df_all["Birth_year"].dropna().max())
        yr = st.slider("Birth year range", miny, maxy, (miny, maxy))
    else:
        yr = None

    if metrics:
        st.subheader("Model quality")
        st.metric("AUROC", f"{metrics.get('AUROC', float('nan')):.3f}")
        st.metric("AUPRC", f"{metrics.get('AUPRC', float('nan')):.3f}")
        st.metric("Brier", f"{metrics.get('Brier', float('nan')):.3f}")

# ------------------------------
# Prepare cohort (latest row per patient) and score
# ------------------------------
latest = latest_rows(df_all)
scored = score_df(latest, model)

# Build an aligned boolean mask
mask = pd.Series(True, index=scored.index)
if "Sex" in scored.columns:
    mask &= scored["Sex"].fillna("Unknown").astype(str).isin(sex_sel)
if yr is not None and "Birth_year" in scored.columns:
    mask &= scored["Birth_year"].between(yr[0], yr[1])

cohort = scored[mask].copy()

# Alerting logic (threshold + capacity)
if model is not None and "risk" in cohort.columns:
    cohort["alert"] = cohort["risk"] >= thr
else:
    cohort["alert"] = False

if cap and cap > 0 and model is not None:
    over = cohort[cohort["alert"]].sort_values("risk", ascending=False)
    keep_ids = set(over.head(int(cap))["Patient_ID"]) if len(over) > cap else set(over["Patient_ID"])
    cohort["alert"] = cohort["Patient_ID"].isin(keep_ids)

# Guard: if filters remove all patients
if cohort.empty:
    st.warning("No patients match the current filters.")
    st.stop()

# ------------------------------
# Layout: Tabs
# ------------------------------
TAB1, TAB2, TAB3, TAB4 = st.tabs(["Overview", "Patients", "Patient detail", "Explainability & reliability"])

# === OVERVIEW ===
with TAB1:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Patients", f"{cohort['Patient_ID'].nunique():,}")
    with c2: st.metric("Flagged today", int(cohort["alert"].sum()))
    with c3: st.metric("Median risk", f"{cohort['risk'].median():.2f}" if model is not None else "-")
    with c4: st.metric("High-risk (Red)", int((cohort.get("risk", 0) >= 0.6).sum()) if model is not None else 0)

    st.markdown("**Risk distribution**")
    if model is not None:
        chart = (
            alt.Chart(cohort.assign(band=cohort["risk_band"]))
            .mark_bar()
            .encode(
                x=alt.X("risk:Q", bin=alt.Bin(maxbins=30), title="Risk score"),
                y=alt.Y("count()", title="Patients"),
                color=alt.Color("band:N", legend=alt.Legend(title="Tier"), sort=BAND_ORDER)
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Model not loaded — distribution unavailable.")

    st.markdown("**Tier breakdown**")
    if model is not None and "risk_band" in cohort.columns:
        tier_counts = cohort["risk_band"].value_counts().reindex(BAND_ORDER, fill_value=0).reset_index()
        tier_counts.columns = ["Tier", "Patients"]
        bar = (
            alt.Chart(tier_counts)
            .mark_bar()
            .encode(
                x=alt.X("Tier:N", sort=BAND_ORDER),
                y="Patients:Q",
                color=alt.Color("Tier:N", scale=alt.Scale(range=[BAND_COLOR[t] for t in BAND_ORDER]))
            )
            .properties(height=200)
        )
        st.altair_chart(bar, use_container_width=True)

# === PATIENTS ===
with TAB2:
    st.caption("Sort by risk to see who needs attention first. Colors show risk tier.")
    # Display columns (alert may not be included here; we sort on cohort first)
    cols = ["Patient_ID", "index_date", "risk", "risk_band", "label_90d", "a1c_prior", "a1c_future"]
    cols = [c for c in cols if c in cohort.columns]

    # Sort keys (only those present)
    sort_keys = [c for c in ["alert", "risk"] if c in cohort.columns]
    if sort_keys:
        cohort_sorted = cohort.sort_values(sort_keys, ascending=[False] * len(sort_keys))
    else:
        cohort_sorted = cohort.copy()

    table = cohort_sorted[cols]

    def _color(val):
        if pd.isna(val):
            return ""
        band = band_name(val)
        return f"background-color: {BAND_COLOR.get(band, '')}"

    if "risk" in table.columns:
        st.dataframe(
            table.style.format({"risk": "{:.2f}"}).applymap(_color, subset=["risk"]),
            use_container_width=True, height=420
        )
    else:
        st.dataframe(table, use_container_width=True, height=420)

    # Download flagged list (use cohort to ensure 'alert' exists)
    flagged_cols = [c for c in cols if c in cohort.columns]
    if "alert" in cohort.columns:
        flagged = cohort.loc[cohort["alert"], flagged_cols]
    else:
        flagged = cohort.head(0)
    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("Download flagged patients (CSV)", data=csv, file_name="flagged_patients.csv", mime="text/csv")

# === PATIENT DETAIL ===
with TAB3:
    left, right = st.columns([1, 2])
    with left:
        pid_list = sorted(cohort["Patient_ID"].unique().tolist())
        if not pid_list:
            st.warning("No patients available.")
            st.stop()
        pid = st.selectbox("Choose a patient", pid_list)

        dates = df_all[df_all["Patient_ID"] == pid].sort_values("index_date")["index_date"].dt.date.unique().tolist()
        if not dates:
            st.warning("No index dates for selected patient.")
            st.stop()
        dt = st.selectbox("Index date", dates, index=len(dates) - 1)

        row = df_all[(df_all["Patient_ID"] == pid) & (df_all["index_date"].dt.date == dt)]
        if row.empty:
            st.warning("No data for this date.")
            risk = None
        else:
            feat_cols = [c for c in row.columns if c not in
                        ["Patient_ID", "date", "index_date", "label_90d", "a1c_prior", "a1c_future"]]
            risk = None
            if model is not None:
                risk = float(model.predict_proba(row[feat_cols])[:, 1][0])
                st.metric("Predicted 90-day risk", f"{risk:.2f}")
                st.metric("Tier", band_name(risk))
                st.metric("Alert at current threshold", "YES" if risk >= thr else "NO")
            if "label_90d" in row.columns:
                st.metric("Observed (label)", int(row["label_90d"].iloc[0]))
            if "a1c_prior" in row.columns:
                st.metric("A1c prior", f"{row['a1c_prior'].iloc[0]:.1f}" if pd.notna(row['a1c_prior'].iloc[0]) else "-")
            if "a1c_future" in row.columns:
                st.metric("A1c within 90d", f"{row['a1c_future'].iloc[0]:.1f}" if pd.notna(row['a1c_future'].iloc[0]) else "-")

            st.markdown("**Suggested next steps**")
            tips = []
            if risk is not None and risk >= 0.60:
                tips.append("High risk: contact within 48h; review insulin dosing and daily routine.")
            elif risk is not None and risk >= 0.30:
                tips.append("Medium risk: schedule a call in 1–2 weeks; reinforce CGM use and meal logging.")

            def _safe_val(df, col):
                return df[col].iloc[0] if col in df.columns and pd.notna(df[col].iloc[0]) else np.nan

            if float(_safe_val(row, "frac_gt_180_mean_30")) > 0.35:
                tips.append("High time above 180 mg/dL: consider basal/bolus adjustment and post-meal spikes.")
            if float(_safe_val(row, "frac_lt_70_mean_30")) > 0.05:
                tips.append("Low sugar episodes: check night-time hypos; consider dose reduction.")
            if float(_safe_val(row, "mean_glu_slope_30")) > 0:
                tips.append("Rising trend: check adherence, infections, or steroid use.")
            if tips:
                for t in tips:
                    st.markdown(f"- {t}")
            else:
                st.info("No specific alerts based on the latest features.")

    with right:
        # show recent trends (last 180 index dates)
        hist = df_all[df_all["Patient_ID"] == pid].sort_values("index_date").tail(180)
        if not hist.empty:
            nice_feats = [
                "frac_gt_180_mean_30", "tir_70_180_mean_30", "mean_glu_mean_30", "mean_glu_slope_30"
            ]
            feats = [c for c in nice_feats if c in hist.columns]
            if feats:
                trend_df = hist[["index_date"] + feats].melt("index_date", var_name="feature", value_name="value")
                line = (
                    alt.Chart(trend_df)
                    .mark_line()
                    .encode(
                        x=alt.X("index_date:T", title="Date"),
                        y=alt.Y("value:Q", title="Value"),
                        color="feature:N"
                    )
                    .properties(height=320)
                )
                st.altair_chart(line, use_container_width=True)
            else:
                st.info("Trend features not found in dataset.")
        else:
            st.info("Not enough history to draw trends.")

# === EXPLAINABILITY & RELIABILITY ===
with TAB4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Global drivers")
        p1 = PLOTS / "shap_global.png"
        p2 = PLOTS / "shap_bar.png"
        if p1.exists(): st.image(str(p1), caption="Most important features across patients")
        if p2.exists(): st.image(str(p2), caption="Average impact by feature")
        if not (p1.exists() or p2.exists()):
            st.info("Run SHAP cells in the notebook to generate global driver plots.")

    with c2:
        st.subheader("Model reliability")
        p_cal = PLOTS / "calibration_quantile.png"
        p_dca = PLOTS / "decision_curve.png"
        if p_cal.exists(): st.image(str(p_cal), caption="Calibration: predicted vs observed")
        if p_dca.exists(): st.image(str(p_dca), caption="Decision curve: net benefit by threshold")
        if not (p_cal.exists() or p_dca.exists()):
            st.info("Run evaluation cells to generate calibration and decision curve plots.")

    st.markdown("---")
    st.subheader("Local explanation (why this patient?)")
    pid_local = st.selectbox("Pick a patient for explanation", sorted(df_all["Patient_ID"].unique()))
    last_row = df_all[df_all["Patient_ID"] == pid_local].sort_values("index_date").tail(1)

    if HAS_SHAP and model is not None and not last_row.empty:
        feat_cols = [c for c in last_row.columns if c not in
                     ["Patient_ID", "date", "index_date", "label_90d", "a1c_prior", "a1c_future"]]
        try:
            # compute SHAP explanation
            exp = shap.Explainer(model.predict_proba, last_row[feat_cols], feature_names=feat_cols)(last_row[feat_cols])
            if hasattr(exp, "values") and getattr(exp.values, "ndim", 2) == 3:
                exp = exp[:, :, 1]

            # Try version-agnostic waterfall rendering
            fig = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(exp[0], max_display=15, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            # Fallback: simple top-10 feature bar chart
            try:
                vals = exp[0].values if hasattr(exp[0], "values") else np.array([])
                names = exp.feature_names if hasattr(exp, "feature_names") else [f"f{i}" for i in range(len(vals))]
                df_expl = pd.DataFrame({"feature": names, "value": vals}).dropna()
                df_expl["abs"] = df_expl["value"].abs()
                top = df_expl.sort_values("abs", ascending=False).head(10)
                top["direction"] = np.where(top["value"] >= 0, "increase risk", "decrease risk")

                bar = (
                    alt.Chart(top)
                    .mark_bar()
                    .encode(
                        x=alt.X("value:Q", title="Contribution to risk"),
                        y=alt.Y("feature:N", sort="-x", title=None),
                        color=alt.Color("direction:N", legend=alt.Legend(title="Effect"))
                    )
                    .properties(height=320)
                )
                st.altair_chart(bar, use_container_width=True)
                st.caption("Fallback explanation (top 10 contributions).")
            except Exception as e2:
                st.error(f"Could not compute SHAP: {e}")
    else:
        st.info("Install shap and ensure the model is trained to see local explanations.")
