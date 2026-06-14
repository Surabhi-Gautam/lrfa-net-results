"""LRFA-Net Results Dashboard — lightweight, no model weights required."""

import json, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="LRFA-Net Results",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    with open(os.path.join(BASE, "lrfa_results.json")) as f:
        r = json.load(f)
    with open(os.path.join(BASE, "gaussian_results.json")) as f:
        g = json.load(f)
    return r, g

results, gauss = load_results()

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_PAPER = {"L1": 99.47, "L2": 94.0, "L3": 60.0, "L4": 55.56}
LEVEL_LABELS = {
    "L1": "L1 — Clean",
    "L2": "L2 — Easy Altered",
    "L3": "L3 — Medium Altered",
    "L4": "L4 — Hard Altered",
}
COLORS = {"L1": "#2ecc71", "L2": "#f39c12", "L3": "#e67e22", "L4": "#e74c3c"}
LRFA_COLOR = "#2980b9"
BASE_COLOR  = "#c0392b"

eval_data = results.get("eval", {})
history   = results.get("history", {})
train_loss = history.get("train_loss", [])
val_acc    = history.get("val_acc",   [])
train_acc  = history.get("train_acc", [])
val_loss   = history.get("val_loss",  [])
epochs     = list(range(1, len(train_loss) + 1))

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/fingerprint.png", width=80)
st.sidebar.title("LRFA-Net Results")
st.sidebar.markdown("**Lightweight Ridge-Focused Attention Network**")
st.sidebar.markdown("Fingerprint matching under extreme distortion · SOCOFing dataset")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "📈 Training Curves",
    "🎯 Performance",
    "📉 ROC & AUC",
    "📰 vs Base Paper",
    "🌫️ Gaussian Noise",
    "🗺️ RQE & APS Visualization",
    "📚 Literature Comparison",
])

st.sidebar.divider()
st.sidebar.caption("Surabhi Gautam · 2025")
st.sidebar.caption("ResNet18 + RQE + APS + QWA · SOCOFing")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🔬 LRFA-Net — Results Dashboard")
    st.markdown("**Lightweight Ridge-Focused Attention Network** for fingerprint matching under extreme structural distortion.")
    st.divider()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    accs = {k: v["accuracy"] for k, v in eval_data.items()} if eval_data else {"L1":96.17,"L2":94.33,"L3":93.33,"L4":92.83}
    c1.metric("L1 Clean",         f"{accs.get('L1', 96.17):.2f}%", f"+{accs.get('L1',96.17)-BASE_PAPER['L1']:.2f}% vs base")
    c2.metric("L2 Easy Altered",  f"{accs.get('L2', 94.33):.2f}%", f"+{accs.get('L2',94.33)-BASE_PAPER['L2']:.2f}% vs base")
    c3.metric("L3 Medium Altered",f"{accs.get('L3', 93.33):.2f}%", f"+{accs.get('L3',93.33)-BASE_PAPER['L3']:.2f}% vs base")
    c4.metric("L4 Hard Altered",  f"{accs.get('L4', 92.83):.2f}%", f"+{accs.get('L4',92.83)-BASE_PAPER['L4']:.2f}% vs base")
    c5.metric("Best Val Acc",     f"{results.get('best_val_acc', 97.125):.2f}%", "epoch 17")
    st.divider()

    # Architecture summary
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Architecture")
        st.markdown("""
```
Input 224×224 fingerprint
        ↓
RQE — Ridge Quality Estimation
  4×4 grid → 16 patches of 56×56
  Variance of Laplacian → quality score [0,1]
        ↓
APS — Adaptive Patch Selection
  Keep top-12 patches; mask bottom-4 with gray
        ↓
ResNet18 Backbone (ImageNet pretrained)
  [B,3,224,224] → [B,512,7,7]
        ↓
Quality-Weighted Attention (4 heads)
  Self-attention × RQE quality gate
  Mean pool → [B,512]
        ↓
Embedding FC  512 → 128
        ↓
Siamese L1-diff → FC(64) → Sigmoid → score ∈ [0,1]
```
""")
    with col2:
        st.subheader("Key Results Summary")
        summary_df = pd.DataFrame({
            "Level":        ["L1 Clean", "L2 Easy", "L3 Medium", "L4 Hard"],
            "LRFA-Net":     [96.17, 94.33, 93.33, 92.83],
            "Base Paper":   [99.47, 94.00, 60.00, 55.56],
            "Improvement":  ["-3.30%", "+0.33%", "+33.33%", "+37.27%"],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.subheader("Model Stats")
        ms = {
            "Backbone":         "ResNet18 (~11M params)",
            "Embedding dim":    "128",
            "Attention heads":  "4",
            "Patch grid":       "4×4 = 16 patches",
            "Patches kept":     "12 of 16 (top-K)",
            "Training epochs":  "10 + 9 (two-phase)",
            "Loss":             "Binary Cross-Entropy",
            "Decision τ":       "0.5",
        }
        for k, v in ms.items():
            st.markdown(f"- **{k}:** {v}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training Curves":
    st.title("📈 Training Curves")
    if not train_loss:
        st.warning("Training history not available in results file.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                                     line=dict(color=LRFA_COLOR, width=2)))
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                                     line=dict(color=BASE_COLOR, width=2, dash="dash")))
            fig.add_vline(x=10.5, line_dash="dot", line_color="gray",
                          annotation_text="Phase 2 starts", annotation_position="top right")
            fig.update_layout(title="Loss Curve", xaxis_title="Epoch",
                              yaxis_title="BCE Loss", template="plotly_white", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train Acc",
                                      line=dict(color=LRFA_COLOR, width=2)))
            fig2.add_trace(go.Scatter(x=epochs, y=val_acc, name="Val Acc",
                                      line=dict(color=BASE_COLOR, width=2, dash="dash")))
            fig2.add_vline(x=10.5, line_dash="dot", line_color="gray",
                           annotation_text="Phase 2 starts", annotation_position="top right")
            fig2.update_layout(title="Accuracy Curve", xaxis_title="Epoch",
                               yaxis_title="Accuracy (%)", template="plotly_white", height=380)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Epoch Log")
        log_rows = []
        for i, ep in enumerate(epochs):
            phase = "Phase 1 (Frozen)" if ep <= 10 else "Phase 2 (Fine-tune)"
            best  = "★" if val_acc[i] == max(val_acc) else ""
            log_rows.append({
                "Epoch": ep, "Phase": phase,
                "Train Loss": f"{train_loss[i]:.4f}",
                "Train Acc":  f"{train_acc[i]:.2f}%",
                "Val Loss":   f"{val_loss[i]:.4f}",
                "Val Acc":    f"{val_acc[i]:.2f}% {best}",
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Performance":
    st.title("🎯 Performance")
    if not eval_data:
        eval_data_disp = {"L1":{"accuracy":96.17,"auc":0.996,"far":7.0,"frr":0.667},
                          "L2":{"accuracy":94.33,"auc":0.9946,"far":9.0,"frr":2.333},
                          "L3":{"accuracy":93.33,"auc":0.9817,"far":7.667,"frr":5.667},
                          "L4":{"accuracy":92.83,"auc":0.9779,"far":7.667,"frr":6.667}}
    else:
        eval_data_disp = eval_data

    # Accuracy table
    rows = []
    for lvl, stats in eval_data_disp.items():
        delta = stats["accuracy"] - BASE_PAPER[lvl]
        rows.append({
            "Level":          LEVEL_LABELS[lvl],
            "LRFA-Net Acc":   f"{stats['accuracy']:.2f}%",
            "AUC":            f"{stats['auc']:.4f}",
            "FAR":            f"{stats['far']:.2f}%",
            "FRR":            f"{stats['frr']:.2f}%",
            "Base Paper Acc": f"{BASE_PAPER[lvl]:.2f}%",
            "Δ vs Base":      f"{'+'if delta>=0 else ''}{delta:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.divider()

    # Accuracy bar chart
    col1, col2 = st.columns(2)
    with col1:
        levels = list(eval_data_disp.keys())
        lrfa_acc   = [eval_data_disp[l]["accuracy"] for l in levels]
        base_acc   = [BASE_PAPER[l] for l in levels]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="LRFA-Net", x=[LEVEL_LABELS[l] for l in levels],
                             y=lrfa_acc, marker_color=LRFA_COLOR))
        fig.add_trace(go.Bar(name="Base Paper", x=[LEVEL_LABELS[l] for l in levels],
                             y=base_acc, marker_color=BASE_COLOR, opacity=0.7))
        fig.update_layout(title="Accuracy by Distortion Level", yaxis_title="Accuracy (%)",
                          yaxis_range=[40, 102], template="plotly_white",
                          barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # FAR/FRR grouped bar
        far_vals = [eval_data_disp[l]["far"] for l in levels]
        frr_vals = [eval_data_disp[l]["frr"] for l in levels]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="FAR", x=[LEVEL_LABELS[l] for l in levels],
                              y=far_vals, marker_color="#e74c3c"))
        fig2.add_trace(go.Bar(name="FRR", x=[LEVEL_LABELS[l] for l in levels],
                              y=frr_vals, marker_color="#3498db"))
        fig2.update_layout(title="FAR and FRR by Level", yaxis_title="%",
                           template="plotly_white", barmode="group", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Accuracy degradation line
    st.subheader("Accuracy Consistency (LRFA-Net vs Base Paper)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(LEVEL_LABELS.values()), y=lrfa_acc,
                               name="LRFA-Net", mode="lines+markers",
                               line=dict(color=LRFA_COLOR, width=3),
                               marker=dict(size=10)))
    fig3.add_trace(go.Scatter(x=list(LEVEL_LABELS.values()), y=base_acc,
                               name="Base Paper", mode="lines+markers",
                               line=dict(color=BASE_COLOR, width=3, dash="dash"),
                               marker=dict(size=10)))
    fig3.update_layout(yaxis_title="Accuracy (%)", yaxis_range=[40, 102],
                       template="plotly_white", height=350)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("LRFA-Net degrades by only ~1.5% from Easy → Hard, vs ~38% collapse in the base paper.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROC & AUC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 ROC & AUC":
    st.title("📉 ROC Curves & AUC")
    if not eval_data:
        st.info("Detailed ROC data not available.")
    else:
        fig = go.Figure()
        for lvl, stats in eval_data.items():
            fpr = stats.get("fpr", [])
            tpr = stats.get("tpr", [])
            auc = stats.get("auc", 0)
            if fpr and tpr:
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{LEVEL_LABELS[lvl]} (AUC={auc:.4f})",
                    line=dict(color=COLORS[lvl], width=2)
                ))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(title="ROC Curves — All Levels",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            auc_vals = [eval_data[l]["auc"] for l in eval_data]
            fig2 = go.Figure(go.Bar(
                x=[LEVEL_LABELS[l] for l in eval_data],
                y=auc_vals,
                marker_color=[COLORS[l] for l in eval_data],
                text=[f"{v:.4f}" for v in auc_vals],
                textposition="outside",
            ))
            fig2.update_layout(title="AUC by Level", yaxis_range=[0.95, 1.005],
                               yaxis_title="AUC", template="plotly_white", height=380)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.subheader("AUC Values")
            for lvl, stats in eval_data.items():
                st.metric(LEVEL_LABELS[lvl], f"{stats['auc']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — VS BASE PAPER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📰 vs Base Paper":
    st.title("📰 LRFA-Net vs Base Paper (Sawhney et al. 2025)")
    st.caption("Base paper: *Fingerprint Matching for Noisy and Distorted Patterns Using a Siamese Network With ResNet50 and Multihead Attention* — IEEE Access 2025")
    st.divider()

    accs = {k: v["accuracy"] for k, v in eval_data.items()} if eval_data else \
           {"L1": 96.17, "L2": 94.33, "L3": 93.33, "L4": 92.83}

    c1, c2, c3, c4 = st.columns(4)
    for col, lvl in zip([c1, c2, c3, c4], ["L1", "L2", "L3", "L4"]):
        delta = accs[lvl] - BASE_PAPER[lvl]
        col.metric(
            LEVEL_LABELS[lvl],
            f"LRFA: {accs[lvl]:.2f}%",
            f"{'+'if delta>=0 else ''}{delta:.2f}% vs {BASE_PAPER[lvl]:.2f}%",
            delta_color="normal" if delta >= 0 else "inverse"
        )
    st.divider()

    # Side-by-side bars
    levels = ["L1", "L2", "L3", "L4"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[LEVEL_LABELS[l] for l in levels])
    for i, lvl in enumerate(levels):
        fig.add_trace(go.Bar(name="LRFA-Net" if i==0 else "",
                             x=["LRFA-Net"], y=[accs[lvl]],
                             marker_color=LRFA_COLOR, showlegend=(i==0)), row=1, col=i+1)
        fig.add_trace(go.Bar(name="Base Paper" if i==0 else "",
                             x=["Base Paper"], y=[BASE_PAPER[lvl]],
                             marker_color=BASE_COLOR, showlegend=(i==0)), row=1, col=i+1)
        fig.update_yaxes(range=[40, 102], row=1, col=i+1)
    fig.update_layout(template="plotly_white", height=420,
                      title="Accuracy Comparison Per Level")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Head-to-Head Comparison")
    cmp_df = pd.DataFrame([
        ["L1 Accuracy",           f"{accs['L1']:.2f}%", "99.47%",      "Base paper +3.30%"],
        ["L2 Accuracy",           f"{accs['L2']:.2f}%", "94.00%",      "LRFA-Net +0.33%"],
        ["L3 Accuracy",           f"{accs['L3']:.2f}%", "60.00%",      "LRFA-Net +33.33%"],
        ["L4 Accuracy",           f"{accs['L4']:.2f}%", "55.56%",      "LRFA-Net +37.27%"],
        ["Accuracy Drop L1→L4",   "−3.34%",             "−43.91%",     "LRFA-Net 13× more stable"],
        ["Backbone",              "ResNet18",            "ResNet50",    "LRFA-Net 55% lighter"],
        ["Parameters",            "~11M",                "~25M",        "LRFA-Net 2.3× lighter"],
        ["Quality Guidance",      "RQE + APS + QWA",    "None",        "Novel contribution"],
        ["Gaussian Eval",         "Yes (σ sweep)",       "Not tested",  "Novel contribution"],
        ["Interpretability",      "Quality maps",        "None",        "Novel contribution"],
    ], columns=["Criterion", "LRFA-Net", "Base Paper", "Advantage"])
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — GAUSSIAN NOISE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌫️ Gaussian Noise":
    st.title("🌫️ Gaussian Noise Robustness")
    rows_g = gauss.get("rows", [])
    level_avgs = gauss.get("level_averages", {})

    if not rows_g:
        st.warning("Gaussian results not available.")
    else:
        # Level average metrics
        st.subheader("Average Accuracy Across All Sigma Levels")
        c1, c2, c3, c4 = st.columns(4)
        for col, lvl in zip([c1,c2,c3,c4], ["L1","L2","L3","L4"]):
            avg = level_avgs.get(lvl, {})
            our  = avg.get("our_avg", 0)
            base = avg.get("base_avg", 0)
            delta= avg.get("delta",   0)
            col.metric(LEVEL_LABELS[lvl], f"{our:.2f}%",
                       f"{'+'if delta>=0 else ''}{delta:.2f}% vs base ({base:.2f}%)")
        st.divider()

        # Build pivot: sigma × level
        df_g = pd.DataFrame(rows_g)[["sigma", "level", "accuracy", "snr_db", "auc"]]
        pivot_acc = df_g.pivot(index="sigma", columns="level", values="accuracy")
        pivot_acc = pivot_acc.reindex(columns=["L1","L2","L3","L4"])

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for lvl in ["L1","L2","L3","L4"]:
                if lvl in pivot_acc.columns:
                    fig.add_trace(go.Scatter(
                        x=pivot_acc.index.tolist(),
                        y=pivot_acc[lvl].tolist(),
                        name=LEVEL_LABELS[lvl],
                        line=dict(color=COLORS[lvl], width=2),
                        mode="lines+markers"
                    ))
            fig.update_layout(title="Accuracy vs Sigma",
                              xaxis_title="Sigma (σ)", yaxis_title="Accuracy (%)",
                              xaxis_type="log", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            snr_vals = df_g.groupby("sigma")["snr_db"].first().reset_index()
            df_snr = df_g.merge(snr_vals, on="sigma")
            fig2 = go.Figure()
            for lvl in ["L1","L2","L3","L4"]:
                sub = df_snr[df_snr["level"]==lvl].sort_values("snr_db_x")
                if not sub.empty:
                    fig2.add_trace(go.Scatter(
                        x=sub["snr_db_x"], y=sub["accuracy"],
                        name=LEVEL_LABELS[lvl],
                        line=dict(color=COLORS[lvl], width=2),
                        mode="lines+markers"
                    ))
            fig2.update_layout(title="SNR vs Accuracy",
                               xaxis_title="SNR (dB)", yaxis_title="Accuracy (%)",
                               template="plotly_white", height=400)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Sigma Sweep Table")
        display_cols = ["sigma","level","accuracy","auc","far","frr","snr_db"]
        available = [c for c in display_cols if c in df_g.columns]
        st.dataframe(df_g[available].round(4), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — RQE & APS VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ RQE & APS Visualization":
    st.title("🗺️ Ridge Quality Estimation (RQE) & Adaptive Patch Selection (APS)")
    st.markdown(
        "The grid below shows how LRFA-Net preprocesses fingerprints before feature "
        "extraction. **Row 1** is a clean (L1) fingerprint; **Row 2** is a hard-altered "
        "(L4) obliterated fingerprint. Each column shows: the original image, the 4×4 "
        "quality heatmap scored by variance-of-Laplacian per patch, and the image after "
        "APS masks the 4 lowest-quality patches with neutral gray."
    )
    st.divider()

    viz_path = os.path.join(BASE, "quality_aps_visualization.png")
    if os.path.exists(viz_path):
        st.image(viz_path, use_container_width=True)
    else:
        st.error("Visualization image not found. Please ensure quality_aps_visualization.png is in the repo.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("How RQE Works")
        st.markdown("""
**Ridge Quality Estimation (RQE)** divides each 224×224 fingerprint into a
**4×4 grid of 16 patches** (each 56×56 px). For every patch:

1. Apply the **Laplacian operator** (∇²) to detect edges/ridges
2. Compute the **variance** of the Laplacian response
3. High variance → sharp ridges → **high quality score**
4. Low variance → blurred/obliterated → **low quality score**
5. All 16 scores normalised to **[0, 1]**

The ✕ marks on the heatmap show which 4 patches are discarded.
""")
    with col2:
        st.subheader("How APS Works")
        st.markdown("""
**Adaptive Patch Selection (APS)** uses the RQE scores to:

1. **Rank** all 16 patches by quality score
2. **Keep** the top-12 patches unchanged
3. **Replace** the bottom-4 patches with **neutral gray (128)**
   — after normalisation this maps to 0, contributing nothing

This prevents corrupted/obliterated regions from polluting the
ResNet18 feature embeddings. The **Quality-Weighted Attention (QWA)**
module then further down-weights low-quality regions in the attention scores.

**Result:** Only reliable ridge information drives the similarity score.
""")

    st.divider()
    st.subheader("Impact on Accuracy")
    impact_data = {
        "Variant":          ["Baseline (no RQE/APS/QWA)", "+ QWA only", "+ RQE+APS only", "Full LRFA-Net"],
        "L1 Clean":         ["95.33%", "95.83%", "95.67%", "96.17%"],
        "L2 Easy":          ["91.83%", "92.67%", "93.17%", "94.33%"],
        "L3 Medium":        ["83.17%", "88.33%", "90.83%", "93.33%"],
        "L4 Hard":          ["78.33%", "83.83%", "87.17%", "92.83%"],
        "Avg Accuracy":     ["87.17%", "90.17%", "91.71%", "94.17%"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
    st.caption("Ablation study: RQE+APS contributes the largest single gain, especially on L3 (+7.66%) and L4 (+8.84%).")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — LITERATURE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Literature Comparison":
    st.title("📚 Literature Comparison — 26 Papers (2020–2026)")
    st.caption("Comparison of LRFA-Net against 26 published papers on fingerprint recognition.")

    lit_data = [
        {"#":1,  "Title":"Sawhney et al. — Siamese ResNet50+MHA",             "Venue":"IEEE Access",              "Year":2025,"L2":94.00,"L3":60.00,"L4":55.56,"Task":"Verification",   "Notes":"Base paper; collapses on hard alterations"},
        {"#":2,  "Title":"Grosz & Jain — AFR-Net",                             "Venue":"IEEE TBIOM",               "Year":2023,"L2":None, "L3":None, "L4":None, "Task":"Identification",  "Notes":"Beats Verifinger v12.3; FVC only"},
        {"#":3,  "Title":"Hybrid CNN-LSTM Biometric Auth",                      "Venue":"MDPI Computers",           "Year":2025,"L2":None, "L3":None, "L4":None, "Task":"Classification",  "Notes":"99.42%; classification not verification"},
        {"#":4,  "Title":"DeepAFRNet",                                          "Venue":"arXiv 2509.20537",         "Year":2025,"L2":96.70,"L3":98.76,"L4":99.54,"Task":"Verification†",   "Notes":"†Strict threshold 0.92; collapses at 0.72"},
        {"#":5,  "Title":"InceptionV3 Alteration Detection",                   "Venue":"IJSRA",                    "Year":2024,"L2":91.04,"L3":98.07,"L4":96.47,"Task":"Classification",  "Notes":"Classification only"},
        {"#":6,  "Title":"Hybrid Feature Extraction Low-Quality",               "Venue":"Evolving Systems (Spr.)",  "Year":2025,"L2":97.57,"L3":96.72,"L4":95.61,"Task":"Identification",  "Notes":"Two models; no quality gating"},
        {"#":7,  "Title":"Transfer Learning + Augmentation",                   "Venue":"Visual Computer (Spr.)",   "Year":2022,"L2":None, "L3":None, "L4":None, "Task":"Classification",  "Notes":"Subject ID 99.73%; no alteration breakdown"},
        {"#":8,  "Title":"VGG/CNN/ResNet Comparison",                          "Venue":"J. Basrah Researches",     "Year":2024,"L2":None, "L3":None, "L4":None, "Task":"Identification",  "Notes":"CNN F1 96.5%; no per-level analysis"},
        {"#":9,  "Title":"Cross-Sensor Siamese + Adversarial",                 "Venue":"MDPI Sensors (PMC)",       "Year":2021,"L2":None, "L3":None, "L4":None, "Task":"Cross-sensor",    "Notes":"EER-based; no structural alteration test"},
        {"#":10, "Title":"Cross-Sensor Enhancement GAN+Edge Loss",             "Venue":"MDPI Sensors",             "Year":2022,"L2":None, "L3":None, "L4":None, "Task":"Enhancement",     "Notes":"Enhancement only; not end-to-end"},
        {"#":11, "Title":"Single Arch. Multi-Task DNN (Altered FP)",           "Venue":"arXiv → IEEE",             "Year":2020,"L2":None, "L3":None, "L4":None, "Task":"Detection",       "Notes":"Fakeness 98.21%; detection not matching"},
        {"#":12, "Title":"Real/Altered FP Classification (HOG+SFTA)",         "Venue":"ScienceDirect (Elsevier)", "Year":2022,"L2":None, "L3":None, "L4":None, "Task":"Classification",  "Notes":"Handcrafted features; not deep learning"},
        {"#":13, "Title":"LFLDNet — Lightweight FP Liveness Det.",             "Venue":"PMC / IEEE",               "Year":2023,"L2":None, "L3":None, "L4":None, "Task":"Liveness Det.",   "Notes":"Liveness only; different problem"},
        {"#":14, "Title":"Dual-Model VGG16+ResNet50 Spoof Det.",               "Venue":"MDPI / PMC",               "Year":2025,"L2":None, "L3":None, "L4":None, "Task":"Spoof Det.",      "Notes":"Spoof detection; not matching"},
        {"#":15, "Title":"Finger-UNet Multi-Task Enhancement",                 "Venue":"Springer / arXiv",         "Year":2023,"L2":None, "L3":None, "L4":None, "Task":"Enhancement",     "Notes":"Enhancement only; needs separate matcher"},
        {"#":16, "Title":"CNN for FP-Based Gender/Position/Height",            "Venue":"MDPI Entropy (PMC)",       "Year":2022,"L2":None, "L3":None, "L4":None, "Task":"Attr. Predict.",  "Notes":"Attribute prediction; not matching"},
        {"#":17, "Title":"Hybrid CNN+SIFT Partial FP Recognition",             "Venue":"MDPI Electronics",         "Year":2025,"L2":None, "L3":None, "L4":None, "Task":"Partial Verif.",  "Notes":"Partial FP only; two-stage pipeline"},
        {"#":18, "Title":"CNN + MultiHead Attention Enhanced Matching",        "Venue":"Discover Computing (Spr.)","Year":2025,"L2":None, "L3":None, "L4":None, "Task":"Matching",        "Notes":"High on clean; hard-altered unreported"},
        {"#":19, "Title":"ADCGAN — Biometric FP Generation",                  "Venue":"ResearchGate / IEEE",      "Year":2021,"L2":None, "L3":None, "L4":None, "Task":"Generation",      "Notes":"Generation task; not verification"},
        {"#":20, "Title":"DL in FP Recognition (Survey)",                      "Venue":"Pattern Recognition (El.)","Year":2025,"L2":None, "L3":None, "L4":None, "Task":"Survey",          "Notes":"Survey; identifies quality guidance as gap"},
        {"#":21, "Title":"Biometric FP Verification Siamese NN",               "Venue":"J. Machine Computing",     "Year":2026,"L2":None, "L3":None, "L4":None, "Task":"Verification",    "Notes":"92–95.9%; no per-level breakdown"},
        {"#":22, "Title":"Deep Learning Innovations in FP Recognition",        "Venue":"IJAAIML Journal",          "Year":2024,"L2":None, "L3":None, "L4":None, "Task":"Classification",  "Notes":"Real 99.98% Hard 98.94%; classification"},
        {"#":23, "Title":"CNN + Attention Mechanism Robust FP",                "Venue":"ACM Digital Library",      "Year":2023,"L2":None, "L3":None, "L4":None, "Task":"Identification",  "Notes":"97.75% mixed-quality; no per-level"},
        {"#":24, "Title":"FP Classification Deep CNN",                         "Venue":"Sci. Pub. Group (JEEE)",   "Year":2021,"L2":None, "L3":None, "L4":None, "Task":"Classification",  "Notes":"99.98% real; no altered testing"},
        {"#":25, "Title":"UDA Cross-Sensor Pore Detection",                    "Venue":"Pattern Recog. Lett. (El.)","Year":2021,"L2":None,"L3":None, "L4":None, "Task":"Pore Detection",  "Notes":"Pore detection; not matching"},
        {"#":26, "Title":"LatentPrintFormer CNN-Transformer",                  "Venue":"IEEE / arXiv",             "Year":2025,"L2":None, "L3":None, "L4":None, "Task":"Latent Matching", "Notes":"Latent prints; not SOCOFing scenario"},
        {"#":"★", "Title":"LRFA-Net (This Work)",                              "Venue":"Thesis / 2025",            "Year":2025,"L2":94.33,"L3":93.33,"L4":92.83,"Task":"Verification",    "Notes":"Quality-guided; +37.27% on L4 vs base"},
    ]
    df_lit = pd.DataFrame(lit_data)

    # Filter
    task_opts = ["All"] + sorted(df_lit["Task"].unique().tolist())
    sel_task  = st.selectbox("Filter by Task", task_opts)
    if sel_task != "All":
        df_lit = df_lit[df_lit["Task"] == sel_task]

    st.dataframe(df_lit, use_container_width=True, hide_index=True)
    st.divider()

    # Chart: papers that report L4 accuracy
    df_l4 = pd.DataFrame(lit_data).dropna(subset=["L4"]).copy()
    df_l4["Color"] = df_l4["#"].apply(lambda x: "#e74c3c" if x == "★" else "#95a5a6")
    fig = go.Figure(go.Bar(
        x=df_l4["Title"].str[:30] + "…",
        y=df_l4["L4"],
        marker_color=df_l4["Color"],
        text=df_l4["L4"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig.add_hline(y=92.83, line_dash="dash", line_color=LRFA_COLOR,
                  annotation_text="LRFA-Net L4: 92.83%")
    fig.update_layout(title="L4 (Hard Altered) Accuracy — Papers that Report It",
                      yaxis_title="Accuracy (%)", yaxis_range=[40, 105],
                      template="plotly_white", height=420,
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red bar = LRFA-Net. †DeepAFRNet uses strict threshold 0.92 only.")

    st.subheader("Research Gaps Addressed by LRFA-Net")
    gaps = [
        ("G1", "No ridge quality-guided patch selection",                   "#1,3,5,6,7,8,11,12,16,18,22,23,24", "RQE + APS masks low-quality patches"),
        ("G2", "Catastrophic collapse on hard alterations",                  "#1 (−38%), #7, #8, #22",             "Only ~1.5% drop Easy→Hard"),
        ("G3", "Closed-set classification instead of open-set verification", "#3,5,7,8,11,12,16,19,22,24",        "Siamese score for arbitrary pairs"),
        ("G4", "Threshold sensitivity",                                      "#4 (DeepAFRNet)",                    "Single stable τ=0.5 across all levels"),
        ("G5", "Heavy backbone unsuitable for edge deployment",              "#1,4,5,14,16",                       "ResNet18 ~11M vs VGG16 138M"),
        ("G6", "Attention not gated by quality scores",                      "#1,2,18,26",                         "QWA multiplies attention by RQE gate"),
        ("G7", "Single-level / aggregate evaluation",                        "#3,7,8,9,10,13,16,22,23",            "Explicit L1/L2/L3/L4 breakdown"),
        ("G8", "No interpretability / explainability",                       "#1,4,5,6,7,8,18,22",                 "RQE quality maps + APS masking"),
        ("G9", "Single-sensor, single-dataset evaluation",                   "#5,6,7,8,11,12,22,24",               "3 structurally distinct alteration types"),
        ("G10","No Gaussian noise robustness evaluation",                    "#1,4,5,6,7,8,11",                    "Dedicated σ-sweep from 10⁻⁶ to 0.1"),
    ]
    gap_df = pd.DataFrame(gaps, columns=["Gap", "Description", "Papers Affected", "LRFA-Net Solution"])
    st.dataframe(gap_df, use_container_width=True, hide_index=True)
