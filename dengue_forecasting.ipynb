import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue Sentinel · Sri Lanka",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b0f1a; color: #dce6f5; }

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-family: 'Syne', sans-serif; color: #f0a500; }

.hero-wrap {
    background: linear-gradient(135deg, #0f1e35 0%, #0b0f1a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: "🦟";
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.07;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    line-height: 1.05;
    background: linear-gradient(120deg, #f0a500 0%, #e05c00 60%, #c0392b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
}
.hero-sub { font-size: 1.05rem; color: #7fa8d0; font-weight: 300; margin: 0; }

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0a500;
    border-left: 4px solid #e05c00;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem;
}

.interpret-box {
    background: #0f1e35;
    border-left: 4px solid #2563eb;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #a8c8e8;
    font-size: 0.9rem;
    line-height: 1.6;
}
.interpret-box strong { color: #60a5fa; }

.warn-box {
    background: #1a1000;
    border-left: 4px solid #f0a500;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #c8a050;
    font-size: 0.88rem;
    line-height: 1.6;
}
.success-box {
    background: #051a10;
    border-left: 4px solid #22c55e;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #86efac;
    font-size: 0.9rem;
    line-height: 1.6;
}
.alert-box {
    background: #1a0505;
    border-left: 4px solid #ef4444;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #fca5a5;
    font-size: 0.9rem;
    line-height: 1.6;
}

.div-line { border: none; border-top: 1px solid #1e2a3a; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#0b0f1a", plot_bgcolor="#111827",
    font=dict(family="DM Sans", color="#dce6f5", size=12),
    title_font=dict(family="Syne", size=15, color="#f0a500"),
    xaxis=dict(gridcolor="#1e2a3a", linecolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    yaxis=dict(gridcolor="#1e2a3a", linecolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    legend=dict(bgcolor="#111827", bordercolor="#1e2a3a", borderwidth=1),
    margin=dict(t=55, b=45, l=55, r=20),
)
C = dict(high="#ef4444", medium="#f59e0b", low="#22c55e",
         primary="#f0a500", blue="#3b82f6", accent="#e05c00")

POPULATION = {
    'Western': 6149000, 'Central': 2766000, 'Southern': 2654000,
    'Northern': 1143000, 'Eastern': 1729000, 'North Western': 2551000,
    'North central': 1377000, 'Uva': 1376000, 'Sabaragamuwa': 2058000
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    raw = pd.read_csv(file)
    n_raw = len(raw)
    missing_before = raw.isnull().sum().copy()
    raw.dropna(inplace=True)
    n_clean = len(raw)
    df = raw.copy()

    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df = df.sort_values('Date').reset_index(drop=True)

    # district aggregations
    dist_monthly = df.groupby(['District', 'Year'])['Cases'].mean().reset_index()
    dist_yearly  = dist_monthly.groupby('District')['Cases'].mean().reset_index()

    # province aggregations
    prov_map = df[['District', 'Province']].drop_duplicates()
    dist_prov = pd.merge(dist_yearly, prov_map, on='District', how='left')
    prov = dist_prov.groupby('Province')['Cases'].sum().reset_index(name='total_cases')
    prov['Population'] = prov['Province'].map(POPULATION)
    prov['incidence_per_100k'] = (prov['total_cases'] / prov['Population']) * 100_000

    low_q  = prov['incidence_per_100k'].quantile(0.33)
    high_q = prov['incidence_per_100k'].quantile(0.66)

    def classify(v, lq=low_q, hq=high_q):
        if v <= lq:  return 'Low Risk'
        if v <= hq:  return 'Medium Risk'
        return 'High Risk'

    prov['risk_level'] = prov['incidence_per_100k'].apply(classify)

    return df, dist_yearly, prov, missing_before, n_raw, n_clean, low_q, high_q

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 Dengue Sentinel")
    st.markdown("**Sri Lanka · Epidemiological Dashboard**")
    st.markdown("---")
    uploaded = st.file_uploader("Upload `dengue.csv`", type=["csv"])
    st.markdown("---")
    st.markdown("### 📋 Navigation")
    SECTIONS = [
        "📊 Dataset Explorer",
        "⚠️ Risk Classification",
        "🔬 Hypothesis Testing",
        "📈 Time Series & Forecast",
    ]
    section = st.radio("Go to section:", SECTIONS, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Sri Lanka dengue surveillance data · 2019–2021")

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">Dengue Sentinel</p>
  <p class="hero-sub">Interactive Epidemiological Analysis &nbsp;·&nbsp; Sri Lanka 2019–2021</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("👈 Please upload the **dengue.csv** file using the sidebar to begin.")
    st.stop()

df, dist_yearly, prov_totals, missing_before, n_raw, n_clean, low_q, high_q = load_data(uploaded)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if section == "📊 Dataset Explorer":

    st.markdown('<p class="section-header">📊 Dataset Explorer</p>', unsafe_allow_html=True)

    # top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{n_clean:,}")
    m2.metric("Rows Removed (NaN)", f"{n_raw - n_clean:,}", delta=f"-{n_raw-n_clean}", delta_color="inverse")
    m3.metric("Districts", df['District'].nunique())
    m4.metric("Provinces", df['Province'].nunique())

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # missing value report
    with st.expander("🔍 Missing Value Report (before cleaning)", expanded=True):
        mv = missing_before.reset_index()
        mv.columns = ["Column", "Missing Values"]
        mv["Status"] = mv["Missing Values"].apply(lambda x: "✅ None" if x == 0 else f"⚠️ {x} rows")
        st.dataframe(mv, use_container_width=True, hide_index=True)
        st.markdown(f"""
        <div class="interpret-box">
        <strong>Data Cleaning:</strong> The raw dataset had <strong>{n_raw:,}</strong> rows.
        Rows with any missing value were removed with <code>dropna()</code>, leaving
        <strong>{n_clean:,}</strong> clean records. This ensures all statistical models
        operate on complete observations and prevents biased estimates.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # ── filters ──
    st.markdown("#### 🎛️ Interactive Filters")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_years = st.multiselect("Year(s)", sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
    with fc2:
        sel_provs = st.multiselect("Province(s)", sorted(df['Province'].unique()), default=sorted(df['Province'].unique()))
    with fc3:
        case_range = st.slider("Case Range", int(df['Cases'].min()), int(df['Cases'].max()),
                               (int(df['Cases'].min()), int(df['Cases'].max())))

    fdf = df[
        df['Year'].isin(sel_years) &
        df['Province'].isin(sel_provs) &
        df['Cases'].between(case_range[0], case_range[1])
    ].copy()
    st.caption(f"Showing **{len(fdf):,}** records")

    show_cols = ['Year', 'Month', 'Province', 'District', 'Cases']
    st.dataframe(fdf[show_cols].reset_index(drop=True), use_container_width=True, height=300)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # ── descriptive stats ──
    st.markdown("#### 📐 Descriptive Statistics")
    grp = st.selectbox("Group by:", ["Overall", "Province", "Year", "District"])
    if grp == "Overall":
        desc = fdf['Cases'].describe().reset_index()
        desc.columns = ["Statistic", "Value"]
        desc["Value"] = desc["Value"].round(2)
        st.dataframe(desc, use_container_width=True, hide_index=True)
    else:
        st.dataframe(fdf.groupby(grp)['Cases'].describe().round(2).reset_index(),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Reading the statistics:</strong> <em>Mean</em> is the average monthly cases;
    <em>std</em> measures variability — a high std relative to the mean signals strong
    seasonal swings. The 25th–75th percentile range shows where most months fall.
    High <em>max</em> values indicate outbreak spikes.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

   

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RISK CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif section == "⚠️ Risk Classification":

    st.markdown('<p class="section-header">⚠️ Province Risk Classification</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Methodology:</strong> Each province is classified as <strong>Low / Medium / High Risk</strong>
    using its <em>incidence rate per 100,000 people</em> — a population-adjusted measure that removes
    bias from simply counting cases in densely populated areas. Thresholds default to the
    <strong>33rd and 66th percentiles</strong> of the provincial incidence distribution,
    but you can adjust them below.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # threshold controls
    st.markdown("#### 🎛️ Adjust Classification Thresholds")
    inc_min = float(prov_totals['incidence_per_100k'].min())
    inc_max = float(prov_totals['incidence_per_100k'].max())

    tc1, tc2 = st.columns(2)
    with tc1:
        user_low = st.slider("Low → Medium threshold (per 100k)",
                             min_value=inc_min, max_value=inc_max,
                             value=float(round(low_q, 1)), step=0.5)
    with tc2:
        user_high = st.slider("Medium → High threshold (per 100k)",
                              min_value=inc_min, max_value=inc_max,
                              value=float(round(high_q, 1)), step=0.5)

    if user_low >= user_high:
        st.error("Low threshold must be strictly less than the High threshold.")
        st.stop()

    pt = prov_totals.copy()
    pt['risk_level'] = pt['incidence_per_100k'].apply(
        lambda v: 'Low Risk' if v <= user_low else ('Medium Risk' if v <= user_high else 'High Risk')
    )

    # province filter
    sel_pr = st.multiselect("Filter provinces:", sorted(pt['Province'].tolist()),
                             default=sorted(pt['Province'].tolist()), key="rp")
    pt_show = pt[pt['Province'].isin(sel_pr)].copy()

    # classification table
    st.markdown("#### 📋 Classification Results")
    disp = pt_show[['Province', 'total_cases', 'Population', 'incidence_per_100k', 'risk_level']].copy()
    disp.columns = ['Province', 'Avg Annual Cases', 'Population', 'Incidence /100k', 'Risk Level']
    disp['Incidence /100k'] = disp['Incidence /100k'].round(2)
    disp['Avg Annual Cases'] = disp['Avg Annual Cases'].round(1)
    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # ── incidence bar chart ──
    st.markdown("#### 📊 Incidence per 100,000 People by Province")

    sort_opt = st.radio("Sort by:",
                        ["Incidence (High→Low)", "Incidence (Low→High)", "Province Name"],
                        horizontal=True, key="sort_r")
    if sort_opt == "Incidence (High→Low)":
        pt_show = pt_show.sort_values('incidence_per_100k', ascending=False)
    elif sort_opt == "Incidence (Low→High)":
        pt_show = pt_show.sort_values('incidence_per_100k', ascending=True)
    else:
        pt_show = pt_show.sort_values('Province')

    cmap = {"High Risk": C['high'], "Medium Risk": C['medium'], "Low Risk": C['low']}

    fig_inc = px.bar(
        pt_show, x='Province', y='incidence_per_100k',
        color='risk_level', color_discrete_map=cmap,
        text=pt_show['incidence_per_100k'].round(1),
        custom_data=['risk_level', 'total_cases', 'Population']
    )
    fig_inc.update_traces(
        textposition='outside',
        textfont=dict(size=11, color='#dce6f5'),
        hovertemplate=(
            "<b>%{x}</b><br>Incidence /100k: %{y:.1f}<br>"
            "Risk: %{customdata[0]}<br>Avg Cases: %{customdata[1]:.0f}<br>"
            "Population: %{customdata[2]:,}<extra></extra>"
        )
    )
    fig_inc.add_hline(y=user_low,  line_dash="dot", line_color="#22c55e",
                      annotation_text=f"Low/Medium: {user_low:.1f}", annotation_font_color="#22c55e")
    fig_inc.add_hline(y=user_high, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"Medium/High: {user_high:.1f}", annotation_font_color="#f59e0b")
    fig_inc.update_layout(**PL,
        title="Dengue Incidence per 100,000 by Province",
        xaxis_title="Province", yaxis_title="Incidence per 100,000",
        legend_title="Risk Level")
    st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Interpretation:</strong> Bars above the <span style="color:#f59e0b">amber line</span>
    are High Risk — these provinces carry the greatest per-capita dengue burden and should be
    prioritised for vector control and health resource allocation. The
    <span style="color:#22c55e">green line</span> separates Low from Medium risk.
    Adjust the threshold sliders to simulate alternative classification policies.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # ── district bar chart ──
    st.markdown("#### 🏙️ Average Annual Cases by District")

    dist_prov_map = df[['District', 'Province']].drop_duplicates()
    dist_full = pd.merge(dist_yearly, dist_prov_map, on='District', how='left')

    dp_filter = st.multiselect("Filter by Province:",
                                sorted(dist_full['Province'].unique()),
                                default=sorted(dist_full['Province'].unique()), key="dp")
    dist_show = dist_full[dist_full['Province'].isin(dp_filter)].sort_values('Cases', ascending=False)

    top_n = st.slider("Show Top N Districts:", 5, len(dist_show), min(20, len(dist_show)), key="tn")
    dist_show = dist_show.head(top_n)

    fig_dist = px.bar(
        dist_show, x='District', y='Cases', color='Province',
        text=dist_show['Cases'].round(0).astype(int),
        labels={'Cases': 'Avg Annual Cases'}
    )
    fig_dist.update_traces(textposition='outside', textfont=dict(size=9, color='#dce6f5'))
    fig_dist.update_layout(**PL,
        title=f"Top {top_n} Districts by Average Annual Dengue Cases",
        xaxis_title="District", yaxis_title="Avg Annual Cases",
        xaxis_tickangle=-45, legend_title="Province")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>District View:</strong> Colombo and other urban Western Province districts dominate
    in raw case counts due to higher population density and greater vector breeding opportunity.
    Use the province filter to focus on specific regions and the top-N slider to declutter the chart.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

   

    st.markdown("""
    <div class="warn-box">
    <strong>Limitations:</strong> Thresholds are relative (quantile-based) so all three risk
    categories will always contain provinces regardless of absolute incidence levels.
    Population estimates are from census data and may not perfectly match 2019–2021 values.
    Only three years of data are analysed, so longer-term trends are not captured.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔬 Hypothesis Testing":

    st.markdown('<p class="section-header">🔬 Hypothesis Testing: Before vs. After 2020</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Research Question:</strong> Did dengue case counts change significantly in or after 2020?
    The COVID-19 pandemic began in 2020 and may have affected dengue surveillance,
    health-seeking behaviour, and transmission dynamics.As the data are not normally distributed,We use a non-parametric test:
    <strong>Mann-Whitney test</strong> to test whether mean monthly cases differ
    between the two periods.
    <br><br>
    <strong>H₀:</strong> Mean dengue cases before the cut-off = mean cases from the cut-off onward.<br>
    <strong>H₁:</strong> The means differ significantly.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # controls
    st.markdown("#### 🎛️ Test Parameters")
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        cutoff = st.slider("Cut-off Year", int(df['Year'].min()), int(df['Year'].max()), 2020)
    with hc2:
        alpha  = st.select_slider("Significance Level (α)", options=[0.01, 0.05, 0.10], value=0.05)
    with hc3:
        scope  = st.selectbox("Scope", ["All Sri Lanka"] + sorted(df['Province'].unique().tolist()))

    scope_df = df if scope == "All Sri Lanka" else df[df['Province'] == scope]
    before   = scope_df[scope_df['Year'] < cutoff]['Cases']
    after    = scope_df[scope_df['Year'] >= cutoff]['Cases']

    if len(before) < 5 or len(after) < 5:
        st.warning("Not enough data in one group. Adjust the cut-off year or scope.")
        st.stop()

    # step 1 — normality
    st.markdown("#### 🧪 Step 1 — Normality (Shapiro-Wilk Test)")
    sw_b = stats.shapiro(before.sample(min(len(before), 5000), random_state=42))
    sw_a = stats.shapiro(after.sample(min(len(after), 5000), random_state=42))

    nc1, nc2 = st.columns(2)
    with nc1:
        st.metric(f"Before {cutoff} — W", f"{sw_b.statistic:.4f}")
        st.metric(f"Before {cutoff} — p", f"{sw_b.pvalue:.4f}")
    with nc2:
        st.metric(f"From {cutoff} — W", f"{sw_a.statistic:.4f}")
        st.metric(f"From {cutoff} — p", f"{sw_a.pvalue:.4f}")

    both_normal = (sw_b.pvalue > alpha) and (sw_a.pvalue > alpha)
    if both_normal:
        st.markdown("""<div class="success-box">✅ Both groups appear <strong>normally distributed</strong>
        (p &gt; α). The t-test normality assumption is satisfied.</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warn-box">⚠️ One or both groups deviate from normality (p ≤ α).
        The t-test is still reported below — for large samples the Central Limit Theorem
        justifies its use, but a Mann-Whitney U test would be more conservative.</div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    
    # step 2 — Mann-Whitney U test
    st.markdown("#### 🧪 Step 2— Mann-Whitney U Test Result")
    from scipy.stats import mannwhitneyu
    mw_stat, p_val = mannwhitneyu(before, after, alternative='two-sided')

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Mann-Whitney U Statistic", f"{mw_stat:.2f}")
    rc2.metric("p-value", f"{p_val:.4f}")
    rc3.metric("Median Before / After", f"{before.median():.1f} / {after.median():.1f}")

    if p_val < alpha:
        direction = "higher" if after.median() > before.median() else "lower"
        st.markdown(f"""<div class="alert-box">
        🔴 <strong>Reject H₀</strong> — p = {p_val:.4f} &lt; α = {alpha}<br><br>
        There is a <strong>statistically significant difference</strong> in dengue cases
        before and after {cutoff}. Cases from {cutoff} onward have a significantly
        <strong>{direction}</strong> median than before
        (median: {before.median():.1f} → {after.median():.1f}).
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="success-box">
        🟢 <strong>Fail to Reject H₀</strong> — p = {p_val:.4f} ≥ α = {alpha}<br><br>
        No statistically significant difference found. The data does not provide
        sufficient evidence of a meaningful change in dengue cases around {cutoff}.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)


    # distribution chart
    st.markdown("#### 📊 Distribution Comparison")
    viz = st.radio("Chart type:", ["Box Plot", "Violin Plot", "Histogram Overlay"], horizontal=True)

    grp_df = pd.concat([
        before.rename('Cases').to_frame().assign(Period=f"Before {cutoff}"),
        after.rename('Cases').to_frame().assign(Period=f"From {cutoff}")
    ], ignore_index=True)

    if viz == "Box Plot":
        fig_h = px.box(grp_df, x='Period', y='Cases', color='Period',
                       color_discrete_sequence=[C['blue'], C['primary']], points='outliers')
    elif viz == "Violin Plot":
        fig_h = px.violin(grp_df, x='Period', y='Cases', color='Period',
                          color_discrete_sequence=[C['blue'], C['primary']], box=True)
    else:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(x=before, name=f'Before {cutoff}',
                                     marker_color=C['blue'], opacity=0.75, nbinsx=40))
        fig_h.add_trace(go.Histogram(x=after, name=f'From {cutoff}',
                                     marker_color=C['primary'], opacity=0.75, nbinsx=40))
        fig_h.update_layout(barmode='overlay')

    fig_h.update_layout(**PL,
        title=f"Dengue Cases: Before vs. After {cutoff} — {scope}",
        xaxis_title="Period", yaxis_title="Cases")
    st.plotly_chart(fig_h, use_container_width=True)

    # summary table
    st.markdown("#### 📐 Descriptive Summary")
    cmp = pd.DataFrame({
        'Statistic': ['n', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        f'Before {cutoff}': [len(before), before.mean().round(2), before.median().round(2),
                              before.std().round(2), before.min(), before.max()],
        f'From {cutoff}':  [len(after),  after.mean().round(2),  after.median().round(2),
                              after.std().round(2),  after.min(),  after.max()],
    })
    st.dataframe(cmp, use_container_width=True, hide_index=True)

   

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TIME SERIES & FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 Time Series & Forecast":

    st.markdown('<p class="section-header">📈 Time Series Analysis & ARIMA Forecast</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Methodology:</strong> An <strong>ARIMA(p,d,q)</strong> model is fitted to monthly
    dengue case data for a selected district. The workflow: (1) check stationarity with the
    Augmented Dickey-Fuller (ADF) test, (2) difference the series until stationary (this gives <em>d</em>),
    (3) inspect ACF and PACF plots to guide the AR order <em>p</em> and MA order <em>q</em>,
    (4) fit the model and produce a multi-month forecast with confidence intervals.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # controls
    st.markdown("#### 🎛️ Analysis Controls")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sel_dist = st.selectbox("District",
            sorted(df['District'].unique()),
            index=list(sorted(df['District'].unique())).index('Colombo')
            if 'Colombo' in df['District'].values else 0)
    with sc2:
        n_lags = st.slider("ACF/PACF Lags", 5, 24, 15)
    with sc3:
        fc_steps = st.slider("Forecast Months", 3, 18, 6)

    ac1, ac2, ac3 = st.columns(3)
    arima_p = ac1.select_slider("p  (AR order)",  options=[0,1,2], value=0)
    arima_d = ac2.select_slider("d  (Differencing)", options=[0,1,2], value=2)
    arima_q = ac3.select_slider("q  (MA order)",  options=[0,1,2], value=2)

    # build time series
    ts = df[df['District'] == sel_dist].groupby('Date')['Cases'].sum()
    ts = ts.asfreq('MS').interpolate()

    if len(ts) < 10:
        st.warning("Not enough data for this district. Please select another.")
        st.stop()

    # ── observed series ──
    st.markdown(f"#### 📉 Observed Monthly Cases — {sel_dist}")
    fig_obs = go.Figure()
    fig_obs.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode='lines+markers',
        line=dict(color=C['primary'], width=2.5),
        marker=dict(size=5, color=C['accent']),
        fill='tozeroy', fillcolor='rgba(240,165,0,0.07)',
        name='Observed Cases'
    ))
    fig_obs.update_layout(**PL,
        title=f"{sel_dist} — Monthly Dengue Cases",
        xaxis_title="Date", yaxis_title="Number of Cases")
    st.plotly_chart(fig_obs, use_container_width=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # ── stationarity table ──
    st.markdown("#### 🔎 Stationarity Check (Augmented Dickey-Fuller Test)")

    adf_raw = adfuller(ts.dropna())
    adf_d1  = adfuller(ts.diff().dropna())
    adf_d2  = adfuller(ts.diff().diff().dropna())

    adf_tbl = pd.DataFrame({
        'Series':        ['Original', '1st Difference', '2nd Difference'],
        'ADF Statistic': [round(adf_raw[0],4), round(adf_d1[0],4), round(adf_d2[0],4)],
        'p-value':       [round(adf_raw[1],4), round(adf_d1[1],4), round(adf_d2[1],4)],
        'Stationary?':   [
            'Yes' if adf_raw[1]<0.05 else 'No',
            'Yes' if adf_d1[1]<0.05  else 'No',
            'Yes' if adf_d2[1]<0.05  else 'No'
        ]
    })
    st.dataframe(adf_tbl, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="interpret-box">
    <strong>Stationarity:</strong> ARIMA requires the series to have a constant mean and variance over time.
    The ADF test checks this — a <strong>p &lt; 0.05</strong> confirms stationarity.
    Each round of differencing removes one layer of trend.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # ── ACF / PACF ──
    st.markdown("#### 📊 ACF & PACF Plots")

    series_acf = ts.copy()
    for _ in range(arima_d):
        series_acf = series_acf.diff().dropna()

    if len(series_acf) < n_lags + 2:
        st.warning("Series too short for selected lags after differencing. Reduce lags or d.")
        st.stop()

    acf_res  = acf(series_acf, nlags=n_lags, fft=True, alpha=0.05)
    pacf_res = pacf(series_acf, nlags=n_lags, alpha=0.05)
    acf_arr, acf_ci   = acf_res[0],  acf_res[1]
    pacf_arr, pacf_ci = pacf_res[0], pacf_res[1]

    lags_arr  = np.arange(len(acf_arr))
    sig_bound = 1.96 / np.sqrt(len(series_acf))

    fig_ap = make_subplots(rows=1, cols=2,
        subplot_titles=("Autocorrelation Function (ACF)",
                        "Partial Autocorrelation Function (PACF)"))

    # ACF bars
    fig_ap.add_trace(go.Bar(
        x=lags_arr, y=acf_arr,
        marker_color=[C['primary'] if abs(v) > sig_bound else '#2a3a52' for v in acf_arr],
        name='ACF',
        error_y=dict(type='data', symmetric=False,
                     array=acf_ci[:,1]-acf_arr,
                     arrayminus=acf_arr-acf_ci[:,0],
                     color='rgba(255,255,255,0.25)')
    ), row=1, col=1)

    # PACF bars
    fig_ap.add_trace(go.Bar(
        x=lags_arr, y=pacf_arr,
        marker_color=[C['accent'] if abs(v) > sig_bound else '#2a3a52' for v in pacf_arr],
        name='PACF',
        error_y=dict(type='data', symmetric=False,
                     array=pacf_ci[:,1]-pacf_arr,
                     arrayminus=pacf_arr-pacf_ci[:,0],
                     color='rgba(255,255,255,0.25)')
    ), row=1, col=2)

    for ci in [1, 2]:
        fig_ap.add_hline(y= sig_bound, line_dash="dash", line_color="#22c55e", line_width=1, row=1, col=ci)
        fig_ap.add_hline(y=-sig_bound, line_dash="dash", line_color="#22c55e", line_width=1, row=1, col=ci)
        fig_ap.add_hline(y=0, line_color="#2a3a52", line_width=1, row=1, col=ci)

    fig_ap.update_layout(**PL,
        title=f"ACF & PACF — {sel_dist} (after d={arima_d} differencing)",
        showlegend=False, height=390)
    fig_ap.update_xaxes(title_text="Lag (months)", row=1, col=1)
    fig_ap.update_xaxes(title_text="Lag (months)", row=1, col=2)
    fig_ap.update_yaxes(title_text="Correlation",  row=1, col=1)

    st.plotly_chart(fig_ap, use_container_width=True)

    st.markdown(f"""
    <div class="interpret-box">
    <strong>How to read ACF/PACF:</strong>
    The <strong>ACF</strong> (left) shows how strongly each past time point correlates with the present.
    The <strong>PACF</strong> (right) shows the <em>direct</em> correlation at each lag after removing
    intermediate effects. Bars exceeding the
    <span style="color:#22c55e">green dashed lines</span> (±{sig_bound:.3f}) are statistically significant.
    <br><br>
    <strong>Choosing p and q:</strong> If PACF cuts off after lag <em>p</em> → use AR(p).
    If ACF cuts off after lag <em>q</em> → use MA(q). Both gradually decaying → use ARMA(p,q).
    Use the <em>p / q</em> sliders above to experiment.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='div-line'>", unsafe_allow_html=True)

    # ── ARIMA FORECAST ──
    st.markdown(f"#### 🔮 ARIMA({arima_p},{arima_d},{arima_q}) with the lowest AIC — {fc_steps}-Month Forecast")

    with st.spinner(f"Fitting ARIMA({arima_p},{arima_d},{arima_q}) …"):
        try:
            model  = ARIMA(ts, order=(arima_p, arima_d, arima_q))
            result = model.fit()

            fc_obj    = result.get_forecast(steps=fc_steps)
            fc_mean   = fc_obj.predicted_mean.clip(lower=0)
            fc_ci     = fc_obj.conf_int()
            fc_ci[fc_ci < 0] = 0
            fitted    = result.fittedvalues.clip(lower=0)

            fig_fc = go.Figure()

            # observed
            fig_fc.add_trace(go.Scatter(
                x=ts.index, y=ts.values,
                mode='lines', line=dict(color=C['primary'], width=2.5),
                name='Observed'
            ))
            # fitted
            fig_fc.add_trace(go.Scatter(
                x=fitted.index, y=fitted.values,
                mode='lines', line=dict(color=C['blue'], width=1.5, dash='dot'),
                name='In-sample Fit'
            ))
            # CI band
            fig_fc.add_trace(go.Scatter(
                x=list(fc_ci.index) + list(fc_ci.index[::-1]),
                y=list(fc_ci.iloc[:,1]) + list(fc_ci.iloc[:,0][::-1]),
                fill='toself', fillcolor='rgba(239,68,68,0.13)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% CI'
            ))
            # forecast
            fig_fc.add_trace(go.Scatter(
                x=fc_mean.index, y=fc_mean.values,
                mode='lines+markers',
                line=dict(color=C['high'], width=2.5),
                marker=dict(size=8, symbol='diamond', color=C['high']),
                name='Forecast'
            ))
            # divider
            fig_fc.add_vline(
                x=ts.index[-1].timestamp()*1000,
                line_dash="dash", line_color="#5a7fa0",
                annotation_text="Forecast start",
                annotation_font_color="#5a7fa0"
            )

            fig_fc.update_layout(**PL,
                title=f"{sel_dist} — ARIMA({arima_p},{arima_d},{arima_q}) {fc_steps}-Month Forecast",
                xaxis_title="Date", yaxis_title="Cases", height=450)
            st.plotly_chart(fig_fc, use_container_width=True)

            st.markdown(f"""
            <div class="interpret-box">
            <strong>Forecast Interpretation:</strong>
            The <span style="color:{C['high']}">red diamond line</span> is the point forecast for
            <strong>{sel_dist}</strong> over the next <strong>{fc_steps} months</strong>.
            The <span style="color:rgba(239,68,68,0.6)">shaded band</span> is the 95% confidence
            interval — the true future value is expected to fall within this range 95% of the time.
            A widening band reflects growing uncertainty with time.
            The <span style="color:{C['blue']}">dotted blue line</span> is the in-sample fitted
            value — how well the model explained past data.
            <br><br>
            <strong>Model Selection:</strong> <em>AIC = {result.aic:.2f}</em>.
            Try different p/q combinations using the sliders; the order with the
            lowest AIC is generally preferred. The original group project used ARIMA(0,2,2).
            </div>""", unsafe_allow_html=True)

            # forecast table
            st.markdown("##### Forecast Values")
            fc_tbl = pd.DataFrame({
                'Month': fc_mean.index.strftime('%B %Y'),
                'Forecast (cases)': fc_mean.values.round(0).astype(int),
                'Lower 95% CI':     fc_ci.iloc[:,0].values.round(0).astype(int),
                'Upper 95% CI':     fc_ci.iloc[:,1].values.round(0).astype(int),
            })
            st.dataframe(fc_tbl, use_container_width=True, hide_index=True)

            # model quality
            qc1, qc2, qc3 = st.columns(3)
            qc1.metric("AIC",  f"{result.aic:.2f}", help="Lower = better fit-complexity balance")
            qc2.metric("BIC",  f"{result.bic:.2f}", help="Lower = better fit-complexity balance")
            qc3.metric("Mean Forecast / Month", f"{fc_mean.mean():.0f} cases")

            with st.expander("📋 Full ARIMA Model Summary", expanded=False):
                st.text(result.summary().as_text())

            
            st.markdown("""
            <div class="warn-box">
            <strong>Limitations:</strong> ARIMA assumes future patterns will resemble past ones.
            It cannot account for exogenous shocks (new dengue serotypes, climate events,
            sudden policy changes). The 3-year training window is short for seasonal modelling,
            and forecast uncertainty grows rapidly beyond 3–6 months.
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Model fitting failed: {e}. Try a different p/d/q combination or district.")
