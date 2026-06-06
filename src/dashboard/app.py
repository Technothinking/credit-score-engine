"""
Credit Score Engine — Streamlit Dashboard
3 pages: Model Performance | Applicant Scorer | Portfolio Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.utils import prob_to_score, get_risk_band, get_risk_band_color
from src.api.scorer import _engineer_features

MODELS = ROOT / "models"
PROC   = ROOT / "data" / "processed"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Score Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model      = joblib.load(MODELS / "xgb_tuned.pkl")
    explainer  = joblib.load(MODELS / "shap_explainer.pkl")
    feat_cols  = joblib.load(MODELS / "feature_cols.pkl")
    X_test     = pd.read_csv(PROC / "X_test.csv")
    y_test     = pd.read_csv(PROC / "y_test.csv").squeeze()
    return model, explainer, feat_cols, X_test, y_test

model, explainer, feat_cols, X_test, y_test = load_artifacts()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Credit Score Engine")
st.sidebar.caption("Alternative data credit scoring")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Model Performance", "Applicant Scorer", "Portfolio Analytics"],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption("Model: XGBoost v1.0")
st.sidebar.caption(f"Test set: {len(X_test):,} applicants")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Model Performance":
    st.title("Model Performance")
    st.caption("Evaluated on held-out test set (20% of 30,000 applicants)")

    # ── Metric cards ──────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
    auc  = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1
    ap   = average_precision_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks   = float((tpr - fpr).max())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",       f"{auc:.4f}")
    c2.metric("Gini",          f"{gini:.4f}")
    c3.metric("KS Statistic",  f"{ks:.4f}")
    c4.metric("Avg Precision", f"{ap:.4f}")

    st.divider()

    # ── ROC curve + KS chart ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            line=dict(color='#4C9BE8', width=2),
            name=f"XGBoost (AUC={auc:.3f})",
            fill='tozeroy', fillcolor='rgba(76,155,232,0.08)'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            name='Random baseline', showlegend=True
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380, margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.subheader("KS separation chart")
        # Sort by predicted probability
        ks_df = pd.DataFrame({'prob': y_prob, 'label': y_test.values})
        ks_df = ks_df.sort_values('prob').reset_index(drop=True)
        n = len(ks_df)
        ks_df['cum_bad']  = (ks_df['label'] == 1).cumsum() / (ks_df['label'] == 1).sum()
        ks_df['cum_good'] = (ks_df['label'] == 0).cumsum() / (ks_df['label'] == 0).sum()
        ks_df['pct']      = np.arange(1, n+1) / n

        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(
            x=ks_df['pct'], y=ks_df['cum_bad'],
            mode='lines', name='Defaulters',
            line=dict(color='#E8654C', width=2)
        ))
        fig_ks.add_trace(go.Scatter(
            x=ks_df['pct'], y=ks_df['cum_good'],
            mode='lines', name='Non-defaulters',
            line=dict(color='#4C9BE8', width=2)
        ))
        ks_idx = (ks_df['cum_bad'] - ks_df['cum_good']).abs().idxmax()
        fig_ks.add_vline(
            x=ks_df.loc[ks_idx, 'pct'],
            line_dash="dot", line_color="gray",
            annotation_text=f"KS={ks:.3f}",
            annotation_position="top right"
        )
        fig_ks.update_layout(
            xaxis_title="Population percentile",
            yaxis_title="Cumulative %",
            height=380, margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_ks, use_container_width=True)

    st.divider()

    # ── SHAP feature importance ───────────────────────────────────────────────
    st.subheader("Global feature importance (mean |SHAP|)")

    @st.cache_data
    def compute_shap_importance():
        sv = explainer.shap_values(X_test.iloc[:500])
        imp = pd.DataFrame({
            'feature':   feat_cols,
            'importance': np.abs(sv).mean(axis=0)
        }).sort_values('importance', ascending=True).tail(20)
        return imp

    imp_df = compute_shap_importance()

    fig_imp = go.Figure(go.Bar(
        x=imp_df['importance'], y=imp_df['feature'],
        orientation='h',
        marker_color='#4C9BE8',
        marker_line_width=0,
    ))
    fig_imp.update_layout(
        xaxis_title="Mean |SHAP value|",
        height=520, margin=dict(l=20, r=20, t=10, b=40)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Score distribution ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Score distribution by default status")

    scores_all = np.array([prob_to_score(p) for p in y_prob])
    dist_df = pd.DataFrame({'score': scores_all, 'label': y_test.values})

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=dist_df[dist_df['label']==0]['score'],
        name='Non-default', nbinsx=40,
        marker_color='rgba(76,155,232,0.7)',
        marker_line_width=0
    ))
    fig_dist.add_trace(go.Histogram(
        x=dist_df[dist_df['label']==1]['score'],
        name='Default', nbinsx=40,
        marker_color='rgba(232,101,76,0.7)',
        marker_line_width=0
    ))
    fig_dist.update_layout(
        barmode='overlay',
        xaxis_title="Credit score",
        yaxis_title="Count",
        height=360, margin=dict(l=40, r=20, t=10, b=40)
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — APPLICANT SCORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Applicant Scorer":
    st.title("Applicant Scorer")
    st.caption("Enter applicant details to generate a live credit score with explanation.")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.expander("📋 Applicant details", expanded=True):
        st.markdown("**Bureau / demographic data**")
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        limit_bal  = r1c1.number_input("Credit limit",    value=50000,  step=5000)
        age        = r1c2.number_input("Age",              value=35,     min_value=18, max_value=100)
        sex        = r1c3.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        education  = r1c4.selectbox("Education", [1,2,3,4],
                                     format_func=lambda x: {1:"Graduate",2:"University",3:"High school",4:"Others"}[x])
        marriage   = r1c5.selectbox("Marriage", [1,2,3],
                                     format_func=lambda x: {1:"Married",2:"Single",3:"Others"}[x])

        st.markdown("**Repayment status (−2=no use, −1=paid in full, 0=on time, 1–8=months delayed)**")
        rs1,rs2,rs3,rs4,rs5,rs6 = st.columns(6)
        pay_0 = rs1.number_input("Sep", value=0, min_value=-2, max_value=8)
        pay_2 = rs2.number_input("Aug", value=0, min_value=-2, max_value=8)
        pay_3 = rs3.number_input("Jul", value=0, min_value=-2, max_value=8)
        pay_4 = rs4.number_input("Jun", value=0, min_value=-2, max_value=8)
        pay_5 = rs5.number_input("May", value=0, min_value=-2, max_value=8)
        pay_6 = rs6.number_input("Apr", value=0, min_value=-2, max_value=8)

        st.markdown("**Bill amounts (last 6 months)**")
        b1,b2,b3,b4,b5,b6 = st.columns(6)
        bill_amt1 = b1.number_input("Sep bill", value=12000, step=500)
        bill_amt2 = b2.number_input("Aug bill", value=11000, step=500)
        bill_amt3 = b3.number_input("Jul bill", value=10500, step=500)
        bill_amt4 = b4.number_input("Jun bill", value=9800,  step=500)
        bill_amt5 = b5.number_input("May bill", value=9200,  step=500)
        bill_amt6 = b6.number_input("Apr bill", value=8900,  step=500)

        st.markdown("**Payment amounts (last 6 months)**")
        p1,p2,p3,p4,p5,p6 = st.columns(6)
        pay_amt1 = p1.number_input("Sep paid", value=2000, step=100)
        pay_amt2 = p2.number_input("Aug paid", value=1800, step=100)
        pay_amt3 = p3.number_input("Jul paid", value=1700, step=100)
        pay_amt4 = p4.number_input("Jun paid", value=1600, step=100)
        pay_amt5 = p5.number_input("May paid", value=1500, step=100)
        pay_amt6 = p6.number_input("Apr paid", value=1400, step=100)

        st.markdown("**UPI / alternative data**")
        u1,u2,u3,u4,u5 = st.columns(5)
        monthly_txn_count     = u1.number_input("Monthly txns",       value=52,   step=1)
        avg_txn_amount        = u2.number_input("Avg txn amount",     value=1800, step=100)
        txn_amount_std        = u3.number_input("Txn amount std",     value=600,  step=50)
        merchant_diversity    = u4.number_input("Merchant diversity", value=18,   step=1)
        months_active         = u5.number_input("Months active",      value=18,   step=1)

        u6,u7,u8,u9,u10 = st.columns(5)
        txn_regularity_score  = u6.slider("Txn regularity",    0.0, 1.0, 0.82)
        utility_payment_ratio = u7.slider("Utility payment",   0.0, 1.0, 0.91)
        recharge_consistency  = u8.slider("Recharge consist.", 0.0, 1.0, 0.88)
        essential_spend_ratio = u9.slider("Essential spend",   0.0, 1.0, 0.64)
        balance_volatility    = u10.slider("Balance volatility",0.0, 1.0, 0.21)

        u11,u12,u13,u14,u15 = st.columns(5)
        salary_credit_regular = u11.selectbox("Salary regular", [1,0],
                                               format_func=lambda x: "Yes" if x==1 else "No")
        late_payment_flag     = u12.selectbox("Late payment",   [0,1],
                                               format_func=lambda x: "No" if x==0 else "Yes")
        emi_to_income_proxy   = u13.slider("EMI/income proxy", 0.0, 1.0, 0.28)
        weekend_spend_ratio   = u14.slider("Weekend spend",    0.0, 1.0, 0.31)
        p2p_to_merchant_ratio = u15.slider("P2P/merchant",     0.0, 1.0, 0.22)

    # ── Score button ──────────────────────────────────────────────────────────
    if st.button("Generate credit score", type="primary", use_container_width=True):

        features = {
            "limit_bal": limit_bal, "sex": sex, "education": education,
            "marriage": marriage, "age": age,
            "pay_0": pay_0, "pay_2": pay_2, "pay_3": pay_3,
            "pay_4": pay_4, "pay_5": pay_5, "pay_6": pay_6,
            "bill_amt1": bill_amt1, "bill_amt2": bill_amt2, "bill_amt3": bill_amt3,
            "bill_amt4": bill_amt4, "bill_amt5": bill_amt5, "bill_amt6": bill_amt6,
            "pay_amt1": pay_amt1, "pay_amt2": pay_amt2, "pay_amt3": pay_amt3,
            "pay_amt4": pay_amt4, "pay_amt5": pay_amt5, "pay_amt6": pay_amt6,
            "monthly_txn_count": monthly_txn_count, "avg_txn_amount": avg_txn_amount,
            "txn_amount_std": txn_amount_std, "txn_regularity_score": txn_regularity_score,
            "salary_credit_regular": salary_credit_regular,
            "recharge_consistency": recharge_consistency,
            "utility_payment_ratio": utility_payment_ratio,
            "late_payment_flag": late_payment_flag,
            "emi_to_income_proxy": emi_to_income_proxy,
            "merchant_diversity": merchant_diversity,
            "weekend_spend_ratio": weekend_spend_ratio,
            "essential_spend_ratio": essential_spend_ratio,
            "p2p_to_merchant_ratio": p2p_to_merchant_ratio,
            "months_active": months_active,
            "balance_volatility": balance_volatility,
        }

        df_raw = pd.DataFrame([features])
        df_eng = _engineer_features(df_raw)
        X      = df_eng.reindex(columns=feat_cols, fill_value=0)

        prob       = float(model.predict_proba(X)[:, 1][0])
        score      = prob_to_score(prob)
        band       = get_risk_band(score)
        band_color = get_risk_band_color(band)

        shap_vals     = explainer.shap_values(X)[0]
        feature_impact= list(zip(feat_cols, shap_vals))
        sorted_impact = sorted(feature_impact, key=lambda x: x[1])

        # ── Persist everything in session state ───────────────────────────────
        st.session_state['score_result'] = {
            'score':          score,
            'prob':           prob,
            'band':           band,
            'band_color':     band_color,
            'sorted_impact':  sorted_impact,
            'X':              X,
        }

    # ── Render results (persists across re-runs) ──────────────────────────────
    if 'score_result' in st.session_state:
        r          = st.session_state['score_result']
        score      = r['score']
        prob       = r['prob']
        band       = r['band']
        band_color = r['band_color']
        sorted_impact = r['sorted_impact']
        X          = r['X']

        top_negative = sorted_impact[:5]
        top_positive = sorted_impact[-5:][::-1]

        # ── Score output ──────────────────────────────────────────────────────
        st.divider()
        col_gauge, col_details = st.columns([1, 1])

        with col_gauge:
            st.subheader("Credit score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0,1], 'y': [0,1]},
                number={'font': {'size': 52}},
                gauge={
                    'axis': {'range': [300, 900],
                             'tickvals': [300,450,550,650,750,900]},
                    'bar':  {'color': band_color, 'thickness': 0.25},
                    'steps': [
                        {'range': [300, 450], 'color': '#fde8e8'},
                        {'range': [450, 550], 'color': '#fef3cd'},
                        {'range': [550, 650], 'color': '#fff9c4'},
                        {'range': [650, 750], 'color': '#d4edda'},
                        {'range': [750, 900], 'color': '#c3e6cb'},
                    ],
                    'threshold': {
                        'line': {'color': band_color, 'width': 4},
                        'thickness': 0.85,
                        'value': score
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_details:
            st.subheader("Decision summary")
            st.markdown(f"""
            | Field | Value |
            |---|---|
            | **Score** | {score} / 900 |
            | **Risk band** | **{band}** |
            | **Default probability** | {prob:.2%} |
            | **Recommendation** | {"✅ Approve" if score >= 600 else "⚠️ Review" if score >= 500 else "❌ Decline"} |
            """)

            st.markdown("**Top positive factors** (lowering risk)")
            for feat, val in top_positive[:3]:
                st.success(f"↑ `{feat}` &nbsp; SHAP: +{val:.4f}")

            st.markdown("**Top negative factors** (increasing risk)")
            for feat, val in top_negative[:3]:
                st.error(f"↓ `{feat}` &nbsp; SHAP: {val:.4f}")

        # ── SHAP waterfall ────────────────────────────────────────────────────
        st.divider()
        st.subheader("SHAP explanation — why this score?")

        top10      = sorted_impact[-5:][::-1] + sorted_impact[:5]
        feat_names = [f[0] for f in top10]
        feat_vals  = [f[1] for f in top10]
        colors     = ['#2ECC71' if v > 0 else '#E74C3C' for v in feat_vals]

        fig_shap = go.Figure(go.Bar(
            x=feat_vals, y=feat_names,
            orientation='h',
            marker_color=colors,
            marker_line_width=0,
        ))
        fig_shap.add_vline(x=0, line_color='gray', line_width=1)
        fig_shap.update_layout(
            xaxis_title="SHAP value (impact on default probability)",
            height=400, margin=dict(l=20, r=20, t=10, b=40)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # ── What-if simulator ─────────────────────────────────────────────────
        st.divider()
        st.subheader("What-if simulator")
        st.caption("Adjust a feature and see how the score changes in real time.")

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            sim_feature  = st.selectbox("Feature to adjust", feat_cols,
                                         key="sim_feature")
            current_val  = float(X[sim_feature].iloc[0])
            step_size    = max(current_val * 0.05, 0.01) if current_val > 1 else 0.01
            min_val      = float(min(current_val * 0.1, current_val - 10)) \
                           if current_val > 0 else -2.0
            max_val      = float(max(current_val * 3, current_val + 10))

            sim_val = st.slider(
                f"Value for `{sim_feature}` (original: {current_val:.3f})",
                min_value=min_val,
                max_value=max_val,
                value=current_val,
                step=step_size,
                key="sim_slider"
            )

        with sim_col2:
            X_sim             = X.copy()
            X_sim[sim_feature]= sim_val
            sim_prob          = float(model.predict_proba(X_sim)[:, 1][0])
            sim_score         = prob_to_score(sim_prob)
            delta             = sim_score - score

            st.metric(
                label="Simulated score",
                value=sim_score,
                delta=f"{delta:+d} vs original {score}"
            )
            st.metric(
                label="Simulated default probability",
                value=f"{sim_prob:.2%}",
                delta=f"{sim_prob - prob:+.2%}",
                delta_color="inverse"
            )
            if delta > 0:
                st.success(f"Score improves by {delta} points")
            elif delta < 0:
                st.error(f"Score drops by {abs(delta)} points")
            else:
                st.info("No change in score")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Analytics":
    st.title("Portfolio Analytics")
    st.caption("Upload a CSV of applicants to score the entire portfolio at once.")

    # ── File upload ───────────────────────────────────────────────────────────
    st.subheader("Batch scoring")
    uploaded = st.file_uploader(
        "Upload applicant CSV (must match feature schema)",
        type=["csv"],
        help="CSV must contain all raw feature columns."
    )

    use_sample = st.checkbox(
        "Use test set sample instead (first 500 rows)", value=True
    )

    if use_sample or uploaded is not None:
        if use_sample:
            df_input = X_test.iloc[:500].copy()
            st.info("Using 500-row test set sample.")
        else:
            df_input = pd.read_csv(uploaded)
            st.success(f"Uploaded {len(df_input):,} applicants.")

        # Score all
        with st.spinner("Scoring portfolio..."):
            df_eng    = _engineer_features(df_input.copy())
            X_batch   = df_eng.reindex(columns=feat_cols, fill_value=0)
            probs     = model.predict_proba(X_batch)[:, 1]
            scores    = np.array([prob_to_score(p) for p in probs])
            bands     = [get_risk_band(s) for s in scores]

            df_results = df_input.copy()
            df_results['predicted_score']       = scores
            df_results['default_probability']   = probs.round(4)
            df_results['risk_band']             = bands

        # ── Summary metrics ───────────────────────────────────────────────────
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total applicants",  f"{len(df_results):,}")
        m2.metric("Avg credit score",  f"{scores.mean():.0f}")
        m3.metric("Avg default prob",  f"{probs.mean():.2%}")
        m4.metric("High risk (≥ HIGH)", f"{(pd.Series(bands).isin(['HIGH','VERY_HIGH'])).sum():,}")

        st.divider()

        # ── Score distribution ────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Score distribution")
            fig_hist = px.histogram(
                df_results, x='predicted_score',
                nbins=40, color_discrete_sequence=['#4C9BE8']
            )
            fig_hist.update_layout(
                xaxis_title="Credit score",
                yaxis_title="Count",
                height=340, margin=dict(l=20,r=20,t=10,b=40),
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Risk band breakdown")
            band_order = ['VERY_LOW','LOW','MEDIUM','HIGH','VERY_HIGH']
            band_colors= ['#2ECC71','#82E0AA','#F39C12','#E74C3C','#922B21']
            band_counts= pd.Series(bands).value_counts().reindex(band_order, fill_value=0)

            fig_pie = go.Figure(go.Pie(
                labels=band_counts.index,
                values=band_counts.values,
                marker_colors=band_colors,
                hole=0.4,
                textinfo='label+percent'
            ))
            fig_pie.update_layout(
                height=340, margin=dict(l=20,r=20,t=10,b=20),
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Score by segment ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Average score by segment")

        seg_col1, seg_col2 = st.columns(2)

        with seg_col1:
            if 'age' in df_results.columns:
                df_results['age_band'] = pd.cut(
                    df_results['age'],
                    bins=[0,25,35,45,60,100],
                    labels=['≤25','26-35','36-45','46-60','60+']
                )
                age_seg = df_results.groupby('age_band', observed=True)['predicted_score'].mean().round(0)
                fig_age = px.bar(
                    x=age_seg.index.astype(str), y=age_seg.values,
                    color_discrete_sequence=['#4C9BE8'],
                    labels={'x':'Age band','y':'Avg score'}
                )
                fig_age.update_layout(height=300, margin=dict(l=20,r=20,t=10,b=40))
                st.plotly_chart(fig_age, use_container_width=True)

        with seg_col2:
            if 'limit_bal' in df_results.columns:
                df_results['limit_band'] = pd.cut(
                    df_results['limit_bal'],
                    bins=[0,50000,100000,200000,float('inf')],
                    labels=['<50K','50–100K','100–200K','>200K']
                )
                lim_seg = df_results.groupby('limit_band', observed=True)['predicted_score'].mean().round(0)
                fig_lim = px.bar(
                    x=lim_seg.index.astype(str), y=lim_seg.values,
                    color_discrete_sequence=['#9B59B6'],
                    labels={'x':'Credit limit band','y':'Avg score'}
                )
                fig_lim.update_layout(height=300, margin=dict(l=20,r=20,t=40,b=40))
                st.plotly_chart(fig_lim, use_container_width=True)

        # ── Download results ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Download scored results")

        csv = df_results[['predicted_score','default_probability','risk_band']].to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="scored_portfolio.csv",
            mime="text/csv",
            use_container_width=True
        )