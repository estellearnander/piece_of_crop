"""Piece of Crop | Streamlit prototype (run from repository root)."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (HAZE_LEVELS, HAZE_PROFILES, MODEL_PREFIXES, PrototypeError, apply_haze,
                  check_local_resources, checkpoint_path, display_rgb, live_predict,
                  read_multispectral_tiff)
from analytics import (critical_confusions, metrics_table, read_haze_summary, read_test_split,
                       read_verified_predictions, summary_table)
from piece_of_crop import CLASS_NAMES, insurer_cost_matrix

st.set_page_config(page_title="Piece of Crop | Land-Cover Intelligence", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.65rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.035em;}
.hero {padding: 1.65rem 1.9rem; border-radius: 18px; border: 1px solid rgba(128,150,130,.24);
 background: linear-gradient(125deg,rgba(37,83,56,.20),rgba(19,85,104,.07)); margin-bottom: 1.15rem;}
.hero .eyebrow {font-size:.76rem;letter-spacing:.16em;opacity:.75;font-weight:750;}
.hero h1 {font-size:2.05rem;margin:.35rem 0 .3rem;line-height:1.14;}
.hero p {font-size:1rem;max-width:760px;opacity:.82;margin:0;}
.good {color:#24a66b; font-weight:700;}
.smallprint {font-size:.85rem;opacity:.78;line-height:1.5;}
[data-testid="stMetric"] {border:1px solid rgba(130,140,130,.2);border-radius:14px;padding:1rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">PIECE OF CROP · RESEARCH PROTOTYPE</div>
  <h1>Land-cover intelligence from satellite imagery</h1>
  <p>Compare RGB and 13-band Sentinel-2 classifications and explore the severity of errors for crop-insurance decisions.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_verified(root_str: str, revision: tuple):
    return read_verified_predictions(Path(root_str))


def revision_token(root: Path):
    files = [root / "results" / "split.csv", *(root / "results" / "predictions").glob("*.csv")]
    return tuple(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.is_file()))


resources = check_local_resources(ROOT)
test_split = read_test_split(ROOT)
try:
    saved_runs, issues = load_verified(str(ROOT), revision_token(ROOT))
    cost_matrix = insurer_cost_matrix()
except Exception as error:
    st.error(f"Could not read project results: {error}")
    st.stop()

with st.sidebar:
    st.title("🌾 Piece of Crop")
    st.caption("EuroSAT · 10 land-cover classes · 64 × 64 pixels · 13 Sentinel-2 bands")
    st.divider()
    st.subheader("Local resources")
    st.markdown(f"{'✅' if resources['split'] else '⚠️'} Test split")
    st.markdown(f"{'✅' if resources['band_stats'] else '⚠️'} Band statistics")
    st.markdown(f"{'✅' if resources['raw_dataset'] else '⚠️'} EuroSAT GeoTIFFs")
    st.markdown(f"**{resources['checkpoints']}** local checkpoints")
    st.markdown(f"**{len(saved_runs)}/18** verified prediction files")
    if issues:
        with st.expander(f"Data quality / missing files ({len(issues)})"):
            for warning in issues[:30]:
                st.caption("• " + warning)
            if len(issues) > 30:
                st.caption(f"...and {len(issues)-30} more")
    st.divider()
    st.caption("Local app. No imagery or predictions are uploaded to a server by this code.")
    st.caption("Academic decision-support demonstration — not an insurance underwriting system.")

classification_tab, comparison_tab, insurance_tab, haze_tab = st.tabs(
    ["🛰️ Satellite Classification", "📊 Model Comparison", "🧮 Insurance Impact", "🌫️ Haze Robustness"]
)

with classification_tab:
    st.subheader("Classify a satellite patch")
    st.caption("Run the same model on RGB and 13 bands from a single original multispectral image. Live predictions use local weights; saved outputs are visibly labelled.")
    left, right = st.columns([1, 1.2], gap="large")
    raw = None
    actual_label = None
    chosen_file = None
    with left:
        source = st.radio("Image source", ["Test patch from EuroSAT", "Upload 13-band GeoTIFF"], horizontal=True)
        if source == "Test patch from EuroSAT":
            if test_split.empty:
                st.warning("No valid test split. Provide results/split.csv or upload a GeoTIFF instead.")
            else:
                class_name = st.selectbox("Ground-truth land-cover class", sorted(test_split["class_name"].unique()))
                subset = test_split[test_split["class_name"].eq(class_name)]
                chosen_file = st.selectbox("Test image", subset["file"].tolist())
                row = subset[subset["file"].eq(chosen_file)].iloc[0]
                actual_label = int(row["label"])
                tif_path = ROOT / "data" / "EuroSAT_MS" / str(row["class_name"]) / Path(chosen_file).name
                if tif_path.is_file():
                    try:
                        raw = read_multispectral_tiff(tif_path)
                    except PrototypeError as e:
                        st.error(str(e))
                else:
                    st.info("Raw TIFF unavailable locally. Verified saved test predictions may still be displayed, but live inference and imagery require the dataset.")
        else:
            upload = st.file_uploader("Choose a multispectral .tif / .tiff", type=["tif", "tiff"], help="Full 13-band, 64 × 64 EuroSAT-style TIFF with original band order and reflectance values.")
            if upload is not None:
                if upload.size > 35 * 1024 * 1024:
                    st.error("File is too large for a single 64×64 EuroSAT patch (35 MB limit).")
                else:
                    try:
                        raw = read_multispectral_tiff(upload.getvalue())
                        st.success("13-band GeoTIFF validated. Ground-truth class is unknown for uploads.")
                    except PrototypeError as e:
                        st.error(str(e))
        haze_strength = st.select_slider("Simulated haze (share of the signal at 550 nm)", options=HAZE_LEVELS,
                                         value=0.0, format_func=lambda v: "none" if v == 0 else f"{v:.0%}")
        haze_profile = HAZE_PROFILES[st.radio("Haze profile", list(HAZE_PROFILES), horizontal=True,
                                              disabled=haze_strength == 0)]
        if raw is not None and haze_strength > 0:
            try:
                raw = apply_haze(ROOT, raw, haze_strength, haze_profile)
            except PrototypeError as e:
                st.error(str(e))
                raw = None
        if raw is not None:
            haze_note = "" if haze_strength == 0 else f" · with simulated haze ({haze_strength:.0%} at 550 nm)"
            st.image(display_rgb(raw), caption="B04 / B03 / B02 · percentile stretch for visualization only" + haze_note, use_container_width=True)
        elif source == "Test patch from EuroSAT" and chosen_file:
            st.caption("Image preview requires the raw GeoTIFF dataset.")

    with right:
        model_name = st.selectbox("Model", list(MODEL_PREFIXES), index=2)
        seed = st.select_slider("Trained seed", options=[0, 1, 2], value=0)
        cpa = checkpoint_path(ROOT, model_name, "rgb", seed)
        cpb = checkpoint_path(ROOT, model_name, "13band", seed)
        st.caption(f"Checkpoints — RGB: {'available' if cpa.is_file() else 'missing'} · 13 bands: {'available' if cpb.is_file() else 'missing'}")
        if actual_label is not None:
            st.markdown(f"**Known class:** `{CLASS_NAMES[actual_label]}` (held-out test patch)")
        else:
            st.markdown("**Known class:** not provided")
        run_clicked = st.button("Compare RGB vs 13 bands", type="primary", use_container_width=True,
                                disabled=(raw is None and chosen_file is None))
        if run_clicked:
            outputs = {}
            errors = []
            for input_type in ("rgb", "13band"):
                key = (model_name, input_type, seed)
                if raw is not None and checkpoint_path(ROOT, model_name, input_type, seed).is_file():
                    try:
                        with st.spinner(f"Running {model_name} on {input_type}…"):
                            probabilities = live_predict(ROOT, raw, model_name, input_type, seed)
                        outputs[input_type] = (probabilities, "Live inference · local checkpoint")
                        continue
                    except PrototypeError as e:
                        errors.append(f"{input_type}: {e}")
                if haze_strength > 0:
                    errors.append(f"{input_type}: haze needs live inference (saved predictions are of clear images).")
                    continue
                if chosen_file is not None and actual_label is not None and key in saved_runs:
                    df = saved_runs[key]
                    matches = df[df["file"].eq(chosen_file)]
                    if len(matches) == 1:
                        probabilities = matches.iloc[0][[f"p_{i}" for i in range(10)]].to_numpy(dtype=float)
                        if np.isfinite(probabilities).all() and np.isclose(probabilities.sum(), 1, atol=1e-3):
                            outputs[input_type] = (probabilities, "Saved test prediction · NOT live inference")
                            continue
                errors.append(f"{input_type}: No compatible checkpoint or verified saved prediction available.")
            if errors:
                st.warning(" | ".join(errors))
            if not outputs:
                st.error("No genuine result can be shown. Check checkpoints, band statistics, dataset and test split.")
            else:
                summary_cols = st.columns(2)
                prob_plot = []
                for col, input_type in zip(summary_cols, ["rgb", "13band"]):
                    with col:
                        st.markdown(f"**{'RGB (3 bands)' if input_type=='rgb' else 'Multispectral (13 bands)'}**")
                        if input_type not in outputs:
                            st.info("Not available")
                            continue
                        proba, provenance = outputs[input_type]
                        pred_id = int(proba.argmax())
                        st.metric("Predicted land-cover", CLASS_NAMES[pred_id])
                        st.metric("Model confidence", f"{proba[pred_id]:.1%}")
                        st.caption(provenance + ("" if haze_strength == 0 else f" · haze {haze_strength:.0%} ({haze_profile})"))
                        if actual_label is not None:
                            severity = cost_matrix[actual_label, pred_id]
                            st.caption(f"{'✓ Correct class' if actual_label==pred_id else '✕ Incorrect class'} · Assumed severity: {severity:g} points")
                        for k, name in enumerate(CLASS_NAMES):
                            prob_plot.append({"Class": name, "Probability": float(proba[k]),
                                              "Input": "RGB" if input_type == "rgb" else "13 bands"})
                if prob_plot:
                    fig = px.bar(pd.DataFrame(prob_plot), x="Probability", y="Class", color="Input",
                                 barmode="group", orientation="h", title="Class probabilities")
                    fig.update_layout(height=460, margin=dict(l=0,r=0,t=45,b=0), yaxis_autorange="reversed")
                    fig.update_xaxes(tickformat=".0%", range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Confidence is the model's probability, not a calibrated statement of correctness.")

with comparison_tab:
    st.subheader("Model comparison on the held-out test set")
    st.caption("Metrics calculated from verified predictions only. RGB and 13-band runs are evaluated on identical test patches.")
    metrics = metrics_table(saved_runs, cost_matrix)
    summary = summary_table(metrics)
    if summary.empty:
        st.warning("No saved prediction files match the current split.csv. Run notebooks 03–06 and refresh the app.")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Verified experiments", f"{len(saved_runs)}/18")
        col_b.metric("Test patches", f"{metrics.iloc[0]['samples']:,}")
        col_c.metric("Independent seeds", int(metrics["seed"].nunique()))
        measure_label = st.selectbox("Comparison measure", ["Accuracy", "Macro-F1", "Mean insurer severity (lower is better)"])
        measure = {"Accuracy":"accuracy", "Macro-F1":"macro_f1", "Mean insurer severity (lower is better)":"mean_severity"}[measure_label]
        fig = px.bar(summary, x="model", y=measure, color="input", barmode="group",
                     text_auto=".3f", labels={"model":"Model", "input":"Input"})
        fig.update_layout(height=450, margin=dict(l=0,r=5,t=25,b=0))
        if measure in ("accuracy", "macro_f1"):
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Mean across available seeds")
        shown = summary.rename(columns={"model":"Model", "input":"Input", "accuracy":"Accuracy",
                                        "macro_f1":"Macro-F1", "mean_severity":"Severity points / patch",
                                        "critical_errors":"Critical errors / seed", "seeds":"Seeds"})
        st.dataframe(shown, hide_index=True, use_container_width=True,
                     column_config={"Accuracy":st.column_config.NumberColumn(format="%.3f"),
                                    "Macro-F1":st.column_config.NumberColumn(format="%.3f"),
                                    "Severity points / patch":st.column_config.NumberColumn(format="%.3f")})
        st.caption("Model ranking is descriptive. Class imbalance, spatial dependence and the chosen severity scenario limit generalization.")

with insurance_tab:
    st.subheader("Insurance impact · severity-weighted errors")
    st.info("All values are **illustrative severity points**, not euros, premiums, claim probabilities or validated actuarial costs. Based on the team's pre-defined cost matrix.")
    labels = list(CLASS_NAMES)
    fig = go.Figure(data=go.Heatmap(z=cost_matrix, x=labels, y=labels, colorscale="YlGnBu",
                                    zmin=0, zmax=max(9, float(cost_matrix.max())),
                                    text=cost_matrix.astype(int), texttemplate="%{text}",
                                    colorbar=dict(title="Points")))
    fig.update_layout(height=540, margin=dict(l=0,r=0,t=25,b=0),
                      xaxis_title="Predicted class", yaxis_title="Actual class")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("0 points = correct classification; 8–9 points = critical confusion under the project scenario.")
    metrics = metrics_table(saved_runs, cost_matrix)
    summary = summary_table(metrics)
    if not summary.empty:
        st.markdown("#### Which model minimizes scenario-weighted mistakes?")
        score = summary[["model", "input", "mean_severity", "critical_errors", "seeds"]].sort_values("mean_severity")
        st.dataframe(score.rename(columns={"model":"Model", "input":"Input", "mean_severity":"Mean points / patch",
                                              "critical_errors":"Critical mistakes / seed", "seeds":"Seeds"}),
                     hide_index=True, use_container_width=True)
        st.markdown("#### Explore the highest-impact confusions")
        mcol, icol, scol = st.columns(3)
        with mcol:
            selected_model = st.selectbox("Model to inspect", list(MODEL_PREFIXES), key="risk_model")
        with icol:
            selected_input = st.selectbox("Input", ["rgb", "13band"], key="risk_input")
        with scol:
            selected_seed = st.selectbox("Seed", [0, 1, 2], key="risk_seed")
        key = (selected_model, selected_input, selected_seed)
        if key in saved_runs:
            summary_pairs = critical_confusions(saved_runs[key], cost_matrix, labels)
            if not summary_pairs.empty:
                st.dataframe(summary_pairs.head(15), hide_index=True, use_container_width=True)
                st.caption("Counts are directed actual → predicted confusions for one seed. Total severity = count × points.")
        else:
            st.warning("This verified run is missing. Choose another combination or regenerate saved predictions.")
    else:
        st.warning("Saved test prediction files are needed to rank real model errors; the severity matrix is still available.")

with haze_tab:
    st.subheader("Robustness to haze · full held-out test set")
    st.caption("Results of notebook 07: the same haze is added to the test patches given to the RGB and the 13-band model. Mean over the available seeds.")
    profile_label = st.radio("Haze profile", list(HAZE_PROFILES), horizontal=True, key="haze_tab_profile")
    haze = read_haze_summary(ROOT, HAZE_PROFILES[profile_label])
    if haze.empty:
        st.warning("Missing results/haze_runs.csv. Run notebook 07 and refresh the app.")
    else:
        measure_label = st.selectbox("Measure", ["Accuracy", "Critical-error rate (lower is better)"], key="haze_measure")
        measure = {"Accuracy": "accuracy", "Critical-error rate (lower is better)": "critical_error_rate"}[measure_label]
        # Haze levels are unevenly spaced, so they are drawn at equal distances.
        position = {level: i for i, level in enumerate(HAZE_LEVELS)}
        haze["position"] = haze["haze_550"].map(position)
        fig = px.line(haze.sort_values("position"), x="position", y=measure, color="input", facet_col="model",
                      markers=True, category_orders={"model": list(MODEL_PREFIXES), "input": ["RGB", "13 bands"]},
                      labels={"position": "Haze at 550 nm", "input": "Input", measure: measure_label})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_xaxes(tickvals=list(position.values()),
                         ticktext=["none" if v == 0 else f"{v:.0%}" for v in HAZE_LEVELS])
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.add_vrect(x0=position[0.40] - 0.5, x1=position[0.63] + 0.5, fillcolor="grey", opacity=0.15,
                      line_width=0, annotation_text="models failed", annotation_position="top left")
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=45, b=0))
        st.plotly_chart(fig, use_container_width=True)
        table = haze.pivot_table(index=["model", "haze_550"], columns="input", values=measure).reset_index()
        table["13 bands − RGB"] = table["13 bands"] - table["RGB"]
        table["haze_550"] = table["haze_550"].map(lambda v: "none" if v == 0 else f"{v:.0%}")
        st.dataframe(table.rename(columns={"model": "Model", "haze_550": "Haze at 550 nm"}), hide_index=True,
                     use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["RGB", "13 bands", "13 bands − RGB"]})
        st.caption("Haze is simulated, not real cloudy imagery. From 40% at 550 nm the models fail (notebook 07), so differences there say little.")

st.divider()
st.caption("Piece of Crop · EuroSAT multispectral imagery · Research / classroom prototype · Models should be independently validated before any real-world deployment.")
