import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Veridi Logistics — Delivery Audit",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = {
    "primary":  "#0D0D0D",
    "surface":  "#141414",
    "card":     "#1C1C1C",
    "border":   "#2A2A2A",
    "accent":   "#E94560",
    "warning":  "#F5A623",
    "safe":     "#0F9B8E",
    "muted":    "#6C757D",
    "text":     "#F0F0F0",
    "subtext":  "#A0A0A0",
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0D0D0D;
    color: #F0F0F0;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 100%; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2A2A2A;
}
section[data-testid="stSidebar"] * {
    color: #F0F0F0 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background-color: #1C1C1C;
    border: 1px solid #2A2A2A;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    color: #A0A0A0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #F0F0F0 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6C757D !important;
    border-bottom: 2px solid transparent;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #E94560 !important;
    border-bottom: 2px solid #E94560 !important;
}

/* Select boxes and sliders */
div[data-baseweb="select"] * {
    background-color: #1C1C1C !important;
    border-color: #2A2A2A !important;
    color: #F0F0F0 !important;
    font-family: 'DM Mono', monospace !important;
}

/* Dividers */
hr { border-color: #2A2A2A; }

/* Plotly charts background */
.js-plotly-plot .plotly { background: transparent !important; }

/* Cards */
.vcard {
    background: #1C1C1C;
    border: 1px solid #2A2A2A;
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #E94560;
    margin-bottom: 0.3rem;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #F0F0F0;
    margin-bottom: 1.5rem;
    line-height: 1.2;
}

.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6C757D;
}

.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
}

.kpi-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #6C757D;
    margin-top: 0.2rem;
}

.finding-card {
    background: #1C1C1C;
    border-left: 3px solid #E94560;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-radius: 0 4px 4px 0;
}

.finding-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #E94560;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.finding-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #F0F0F0;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout ─────────────────────────────────────────────────────────
def chart_layout(**kw):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1C1C1C",
        font=dict(family="DM Mono, monospace", color="#A0A0A0", size=11),
        title_font=dict(family="Syne, sans-serif", size=16, color="#F0F0F0"),
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(
            bgcolor="#1C1C1C",
            font_size=12,
            font_family="DM Mono, monospace",
            bordercolor="#2A2A2A",
        ),
        xaxis=dict(gridcolor="#2A2A2A", linecolor="#2A2A2A", zerolinecolor="#2A2A2A"),
        yaxis=dict(gridcolor="#2A2A2A", linecolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    )
    base.update(kw)
    return base

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    import requests
    import io

    FILE_IDS = {
        "orders":      "138GX-CCP8UGHUMWhRo6Fms0FOdqg-Kft",
        "reviews":     "1GW2-GtnONv9XW2FTV-ARZTWPKmS5JNu5",
        "customers":   "1QblPTjmwPvC5_q7T59poLVXNTYsStHvJ",
        "products":    "1Rkm4mB7wA-IMNAsGsRl71R0Sk4qlt2M4",
        "order_items": "138GX-CCP8UGHUMWhRo6Fms0FOdqg-Kft",
        "sellers":     "1v-ydEwlUqzvdcSrtPEVkpBkPLSulZQM_",
        "payments":    "1LIopgN1ZHLEH4WVdU5H0NZ3KJjPMKYE0",
        "translation": "1Gi6RT6nxtUFrn0tyT8Lq6bp7OfmSTFrN",
    }

    def download_gdrive(file_id):
        session = requests.Session()
        url = "https://drive.google.com/uc?export=download"
        response = session.get(url, params={"id": file_id}, stream=True)
        # Handle large file confirmation
        for key_name, value in response.cookies.items():
            if key_name.startswith("download_warning"):
                response = session.get(url, params={"id": file_id, "confirm": value}, stream=True)
                break
        content = b"".join(response.iter_content(chunk_size=32768))
        return pd.read_csv(io.BytesIO(content), low_memory=False)

    data = {}
    for key, fid in FILE_IDS.items():
        data[key] = download_gdrive(fid)

    orders      = data["orders"]
    reviews     = data["reviews"]
    customers   = data["customers"]
    products    = data["products"]
    order_items = data["order_items"]
    sellers     = data["sellers"]
    payments    = data["payments"]
    translation = data["translation"]

    DATE_COLS = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in DATE_COLS:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    reviews["review_creation_date"] = pd.to_datetime(
        reviews["review_creation_date"], errors="coerce"
    )

    reviews_clean = (
        reviews.sort_values("review_creation_date", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        [["order_id", "review_score", "review_comment_message"]]
    )
    first_items = (
        order_items.sort_values("order_item_id")
        .drop_duplicates(subset="order_id", keep="first")
        [["order_id", "product_id", "price", "freight_value", "seller_id"]]
    )
    payment_agg = (
        payments.groupby("order_id")
        .agg(total_payment=("payment_value","sum"),
             installments=("payment_installments","max"),
             payment_type=("payment_type","first"))
        .reset_index()
    )
    products_en = products.merge(translation, on="product_category_name", how="left")

    df = (
        orders
        .merge(reviews_clean, on="order_id", how="left")
        .merge(customers[["customer_id","customer_state","customer_city"]], on="customer_id", how="left")
        .merge(first_items, on="order_id", how="left")
        .merge(products_en[["product_id","product_category_name_english",
                             "product_weight_g","product_description_lenght"]], on="product_id", how="left")
        .merge(sellers[["seller_id","seller_state"]], on="seller_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
    )

    df_del = (
        df[df["order_status"] == "delivered"].copy()
        .dropna(subset=["order_delivered_customer_date","order_estimated_delivery_date"])
    )
    df_del["delay_days"] = (
        df_del["order_delivered_customer_date"] - df_del["order_estimated_delivery_date"]
    ).dt.days
    df_del["actual_transit_days"]   = (df_del["order_delivered_customer_date"] - df_del["order_purchase_timestamp"]).dt.days
    df_del["promised_transit_days"] = (df_del["order_estimated_delivery_date"]  - df_del["order_purchase_timestamp"]).dt.days
    df_del["approval_lag_hours"]    = (df_del["order_approved_at"] - df_del["order_purchase_timestamp"]).dt.total_seconds() / 3600
    df_del["order_month"]   = df_del["order_purchase_timestamp"].dt.month
    df_del["order_quarter"] = df_del["order_purchase_timestamp"].dt.quarter
    df_del["order_year"]    = df_del["order_purchase_timestamp"].dt.year
    df_del["order_dow"]     = df_del["order_purchase_timestamp"].dt.dayofweek
    df_del["cross_state"]   = (df_del["customer_state"] != df_del["seller_state"]).astype(int)

    def classify(d):
        if d <= 0:   return "On Time"
        elif d <= 5: return "Late"
        else:        return "Critical"

    df_del["delivery_class"] = df_del["delay_days"].apply(classify)

    STATE_NAMES = {
        "AC":"Acre","AL":"Alagoas","AP":"Amapa","AM":"Amazonas","BA":"Bahia",
        "CE":"Ceara","DF":"Distrito Federal","ES":"Espirito Santo","GO":"Goias",
        "MA":"Maranhao","MT":"Mato Grosso","MS":"Mato Grosso do Sul",
        "MG":"Minas Gerais","PA":"Para","PB":"Paraiba","PR":"Parana",
        "PE":"Pernambuco","PI":"Piaui","RJ":"Rio de Janeiro",
        "RN":"Rio Grande do Norte","RS":"Rio Grande do Sul","RO":"Rondonia",
        "RR":"Roraima","SC":"Santa Catarina","SP":"Sao Paulo",
        "SE":"Sergipe","TO":"Tocantins",
    }

    state_stats = (
        df_del.groupby("customer_state").agg(
            order_count    = ("order_id","count"),
            late_count     = ("delivery_class", lambda x: (x != "On Time").sum()),
            critical_count = ("delivery_class", lambda x: (x == "Critical").sum()),
            avg_delay      = ("delay_days","mean"),
            avg_review     = ("review_score","mean"),
            avg_transit    = ("actual_transit_days","mean"),
        ).reset_index()
    )
    state_stats["pct_late"]     = (state_stats["late_count"]     / state_stats["order_count"] * 100).round(2)
    state_stats["pct_critical"] = (state_stats["critical_count"] / state_stats["order_count"] * 100).round(2)
    state_stats["state_name"]   = state_stats["customer_state"].map(STATE_NAMES)
    state_stats = state_stats.sort_values("pct_late", ascending=False).reset_index(drop=True)

    return df_del, state_stats

# ── ML model ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(df_del):
    ml = df_del.copy().dropna(subset=["customer_state","promised_transit_days"])
    ml["is_late"]         = (ml["delay_days"] > 0).astype(int)
    REMOTE                = {"AM","PA","AC","RO","RR","AP","TO","MA","PI"}
    ml["is_remote"]       = ml["customer_state"].isin(REMOTE).astype(int)
    ml["cross_state_del"] = ml["cross_state"].fillna(1).astype(int)
    ml["price_filled"]    = ml["price"].fillna(ml["price"].median())
    ml["price_quartile"]  = pd.qcut(ml["price_filled"], q=4, labels=[0,1,2,3]).astype(int)
    ml["freight_ratio"]   = (ml["freight_value"] / (ml["price_filled"] + 1)).clip(0, 5)
    ml["approval_lag_h"]  = ml["approval_lag_hours"].clip(0, 72).fillna(24)
    le = LabelEncoder()
    ml["state_enc"] = le.fit_transform(ml["customer_state"])
    FEATURES = [
        "state_enc","order_month","order_dow","order_quarter",
        "promised_transit_days","is_remote","cross_state_del",
        "price_quartile","freight_ratio","approval_lag_h",
    ]
    X = ml[FEATURES]
    y = ml["is_late"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=15,
        class_weight="balanced", max_features="sqrt",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, FEATURES, accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob)

# ── Brazil GeoJSON ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_geojson():
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    with urllib.request.urlopen(url) as r:
        geo = json.loads(r.read())
    for f in geo["features"]:
        f["id"] = f["properties"]["sigla"]
    return geo

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800; color: #F0F0F0; line-height: 1.2;'>
            VERIDI<br>LOGISTICS
        </div>
        <div style='font-family: DM Mono, monospace; font-size: 0.6rem; letter-spacing: 0.2em; color: #E94560; margin-top: 0.4rem; text-transform: uppercase;'>
            Delivery Audit — 2016–2018
        </div>
    </div>
    <hr style='border-color: #2A2A2A; margin: 0 0 1.5rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Overview", "Geographic Analysis", "Sentiment Analysis",
         "Category Breakdown", "Trend Analysis", "ML Prediction Model"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color: #2A2A2A; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df_del, state_stats = load_data()

    year_filter = st.multiselect(
        "Year",
        options=sorted(df_del["order_year"].unique()),
        default=sorted(df_del["order_year"].unique()),
    )
    class_filter = st.multiselect(
        "Delivery Class",
        options=["On Time", "Late", "Critical"],
        default=["On Time", "Late", "Critical"],
    )

    st.markdown("<hr style='border-color: #2A2A2A; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 0.62rem; color: #444; line-height: 1.8;'>
        Dataset: Olist Brazilian<br>
        E-Commerce Public Dataset<br>
        Source: Kaggle<br>
        <span style='color: #E94560;'>▲</span> Veridi Logistics Audit
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
df = df_del[
    df_del["order_year"].isin(year_filter) &
    df_del["delivery_class"].isin(class_filter)
].copy()

total      = len(df_del)
on_time_n  = (df_del["delivery_class"] == "On Time").sum()
late_n     = (df_del["delivery_class"] == "Late").sum()
critical_n = (df_del["delivery_class"] == "Critical").sum()
avg_ot     = df_del[df_del["delivery_class"] == "On Time"]["review_score"].mean()
avg_crit   = df_del[df_del["delivery_class"] == "Critical"]["review_score"].mean()

CLASS_COLORS = {
    "On Time":  PALETTE["safe"],
    "Late":     PALETTE["warning"],
    "Critical": PALETTE["accent"],
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <div style='padding: 2rem 0 1rem 0;'>
        <div class='section-label'>Executive Overview</div>
        <div class='section-title'>Delivery Performance Audit</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>Total Orders</div>
            <div class='kpi-value' style='color:#F0F0F0'>{total:,}</div>
            <div class='kpi-sub'>Sep 2016 – Aug 2018</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>On Time Rate</div>
            <div class='kpi-value' style='color:{PALETTE["safe"]}'>{on_time_n/total:.1%}</div>
            <div class='kpi-sub'>{on_time_n:,} orders</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>Late (1–5 days)</div>
            <div class='kpi-value' style='color:{PALETTE["warning"]}'>{late_n/total:.1%}</div>
            <div class='kpi-sub'>{late_n:,} orders</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>Critical (>5 days)</div>
            <div class='kpi-value' style='color:{PALETTE["accent"]}'>{critical_n/total:.1%}</div>
            <div class='kpi-sub'>{critical_n:,} orders</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        drop = avg_ot - avg_crit
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>Satisfaction Drop</div>
            <div class='kpi-value' style='color:{PALETTE["accent"]}'>-{drop:.2f}</div>
            <div class='kpi-sub'>stars: on-time vs critical</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        counts = df_del["delivery_class"].value_counts()
        fig = go.Figure()
        fig.add_trace(go.Pie(
            values=counts.values,
            labels=counts.index,
            hole=0.65,
            marker=dict(
                colors=[CLASS_COLORS.get(k, "#999") for k in counts.index],
                line=dict(color="#0D0D0D", width=3),
            ),
            textinfo="percent",
            textfont=dict(size=13, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value:,} orders<br>%{percent}<extra></extra>",
            sort=False,
        ))
        fig.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:11px'>orders</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#F0F0F0", family="Syne, sans-serif"),
        )
        fig.update_layout(**chart_layout(
            title="Overall Delivery Performance",
            showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center",
                       font=dict(color="#A0A0A0")),
            height=380,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        data_hist = df_del["delay_days"].clip(-40, 80)
        fig = go.Figure()
        for x0, x1, color in [(-40,0,PALETTE["safe"]),(0,5,PALETTE["warning"]),(5,80,PALETTE["accent"])]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.06, line_width=0)
        fig.add_trace(go.Histogram(
            x=data_hist, nbinsx=80,
            marker=dict(color="#3A3A3A", line=dict(color=PALETTE["accent"], width=0.3)),
            hovertemplate="Delay: %{x}d<br>Orders: %{y:,}<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=PALETTE["muted"], dash="dash", width=1),
                     annotation_text="Promised", annotation_font_color=PALETTE["muted"])
        fig.update_layout(**chart_layout(
            title="Distribution of Delivery Delays",
            xaxis=dict(title="Days Late (negative = early)", gridcolor="#2A2A2A"),
            yaxis=dict(title="Order Count", gridcolor="#2A2A2A"),
            height=380, showlegend=False,
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Key findings
    st.markdown("<hr style='border-color: #2A2A2A;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
        <div class='finding-card'>
            <div class='finding-num'>Finding 01 — Satisfaction</div>
            <div class='finding-text'>Critical late orders receive {avg_crit:.2f} stars vs {avg_ot:.2f} for on-time — a {drop:.2f} point collapse in satisfaction</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        worst = state_stats.iloc[0]
        st.markdown(f"""
        <div class='finding-card'>
            <div class='finding-num'>Finding 02 — Geography</div>
            <div class='finding-text'>{worst['state_name']} is the worst state with {worst['pct_late']:.1f}% late deliveries — northeast Brazil is systemically failing</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class='finding-card'>
            <div class='finding-num'>Finding 03 — Prediction</div>
            <div class='finding-text'>A Random Forest model predicts late deliveries with 79.4% accuracy — enabling proactive intervention before shipment</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Geographic Analysis":

    st.markdown("""
    <div style='padding: 2rem 0 1rem 0;'>
        <div class='section-label'>Story 03 — Geographic Heatmap</div>
        <div class='section-title'>Where Are We Failing?</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        try:
            geo = load_geojson()
            fig = px.choropleth(
                state_stats,
                geojson=geo,
                locations="customer_state",
                color="pct_late",
                hover_name="state_name",
                hover_data={"pct_late":":.1f","avg_delay":":.1f",
                           "avg_review":":.2f","order_count":True},
                color_continuous_scale=[
                    [0.0, PALETTE["safe"]],
                    [0.4, PALETTE["warning"]],
                    [1.0, PALETTE["accent"]],
                ],
                fitbounds="locations",
                basemap_visible=False,
            )
            fig.update_geos(
                showframe=False, showcoastlines=False,
                showland=True, landcolor="#1C1C1C",
                showocean=True, oceancolor="#141414",
                bgcolor="#0D0D0D",
            )
            fig.update_layout(**chart_layout(
                title="Late Delivery Rate by State — Brazil",
                coloraxis_colorbar=dict(
                    title="% Late", ticksuffix="%", thickness=12,
                    tickfont=dict(color="#A0A0A0"), title_font=dict(color="#A0A0A0"),
                ),
                height=520,
            ))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Map unavailable — showing bar chart instead.")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=state_stats["pct_late"],
            y=state_stats["customer_state"],
            orientation="h",
            marker=dict(
                color=state_stats["pct_late"],
                colorscale=[[0,PALETTE["safe"]],[0.4,PALETTE["warning"]],[1,PALETTE["accent"]]],
                showscale=False,
                line=dict(width=0),
            ),
            text=state_stats["pct_late"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
            textfont=dict(size=10, color="#A0A0A0"),
            customdata=state_stats[["state_name","avg_review","order_count"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Late: %{x:.1f}%<br>"
                "Avg review: %{customdata[1]:.2f}<br>"
                "Orders: %{customdata[2]:,}<extra></extra>"
            ),
        ))
        fig.update_layout(**chart_layout(
            title="States Ranked by Late Rate",
            xaxis=dict(title="% Late Deliveries", gridcolor="#2A2A2A"),
            yaxis=dict(autorange="reversed", tickfont=dict(size=10), gridcolor="#2A2A2A"),
            height=520,
        ))
        st.plotly_chart(fig, use_container_width=True)

    # State detail table
    st.markdown("<hr style='border-color: #2A2A2A;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">State Detail</div>', unsafe_allow_html=True)

    display = state_stats[["customer_state","state_name","order_count",
                           "pct_late","pct_critical","avg_review","avg_delay"]].copy()
    display.columns = ["Code","State","Orders","% Late","% Critical","Avg Review","Avg Delay (d)"]
    display["% Late"]      = display["% Late"].apply(lambda x: f"{x:.1f}%")
    display["% Critical"]  = display["% Critical"].apply(lambda x: f"{x:.1f}%")
    display["Avg Review"]  = display["Avg Review"].apply(lambda x: f"{x:.2f}")
    display["Avg Delay (d)"] = display["Avg Delay (d)"].apply(lambda x: f"{x:.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sentiment Analysis":

    st.markdown("""
    <div style='padding: 2rem 0 1rem 0;'>
        <div class='section-label'>Story 04 — Sentiment Correlation</div>
        <div class='section-title'>Does Lateness Kill Reviews?</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        ORDER_MAP = {"On Time": 0, "Late": 1, "Critical": 2}
        sentiment = (
            df_del.groupby("delivery_class")
            .agg(avg_review=("review_score","mean"),
                 std_review=("review_score","std"),
                 count=("order_id","count"))
            .reset_index()
            .assign(sort_key=lambda d: d["delivery_class"].map(ORDER_MAP))
            .sort_values("sort_key").reset_index(drop=True)
        )
        colors = [PALETTE["safe"], PALETTE["warning"], PALETTE["accent"]]
        fig = go.Figure()
        for i, row in sentiment.iterrows():
            fig.add_trace(go.Bar(
                x=[row["delivery_class"]], y=[row["avg_review"]],
                error_y=dict(type="data", array=[row["std_review"]],
                            color=colors[i], thickness=2, width=6),
                marker=dict(color=colors[i], opacity=0.85, line=dict(width=0)),
                width=0.45,
                text=f"{row['avg_review']:.3f}",
                textposition="outside",
                textfont=dict(size=14, color="#F0F0F0", family="Syne, sans-serif"),
                showlegend=False,
            ))
        drop = avg_ot - avg_crit
        fig.add_annotation(
            x="Critical", y=avg_crit + 0.5,
            text=f"<b>-{drop:.2f} pts</b>",
            showarrow=False,
            font=dict(color=PALETTE["accent"], size=13, family="Syne, sans-serif"),
            bgcolor="#1C1C1C", bordercolor=PALETTE["accent"],
            borderwidth=1, borderpad=6,
        )
        fig.update_layout(**chart_layout(
            title="Avg Review Score by Delivery Class",
            yaxis=dict(range=[0,6], title="Avg Review Score (1–5)", gridcolor="#2A2A2A"),
            xaxis=dict(gridcolor="#2A2A2A"),
            height=420,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df_del["delay_bucket"] = df_del["delay_days"].clip(-30, 60)
        dr = (
            df_del.groupby("delay_bucket")
            .agg(avg_review=("review_score","mean"), count=("order_id","count"))
            .reset_index().query("count >= 20")
        )
        fig = px.scatter(
            dr, x="delay_bucket", y="avg_review",
            size="count", size_max=28,
            color="avg_review",
            color_continuous_scale=[
                [0.0, PALETTE["accent"]],
                [0.5, PALETTE["warning"]],
                [1.0, PALETTE["safe"]],
            ],
            range_color=[1, 5],
            trendline="lowess",
            trendline_color_override="#F0F0F0",
            custom_data=["count"],
        )
        fig.update_traces(
            hovertemplate="Delay: <b>%{x}d</b><br>Avg review: <b>%{y:.3f}</b><br>Orders: %{customdata[0]:,}<extra></extra>",
            selector=dict(mode="markers"),
        )
        fig.add_vline(x=0, line=dict(color=PALETTE["muted"], dash="dash", width=1),
                     annotation_text="Promised date",
                     annotation_font_color=PALETTE["muted"])
        fig.update_layout(**chart_layout(
            title="Delay vs Review Score — Correlation",
            xaxis=dict(title="Days Late (negative = early)", gridcolor="#2A2A2A"),
            yaxis=dict(range=[0.8, 5.3], title="Avg Review Score", gridcolor="#2A2A2A"),
            coloraxis_showscale=False,
            height=420,
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Review score distribution
    st.markdown("<hr style='border-color: #2A2A2A;'>", unsafe_allow_html=True)
    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        subset = df_del[df_del["delivery_class"] == cls]["review_score"].dropna()
        fig.add_trace(go.Histogram(
            x=subset, name=cls,
            marker_color=color, opacity=0.7,
            nbinsx=5,
            hovertemplate=f"<b>{cls}</b><br>Score: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(**chart_layout(
        title="Review Score Distribution by Delivery Class",
        barmode="overlay",
        xaxis=dict(title="Review Score (1–5)", gridcolor="#2A2A2A", dtick=1),
        yaxis=dict(title="Count", gridcolor="#2A2A2A"),
        legend=dict(font=dict(color="#A0A0A0"), bgcolor="rgba(0,0,0,0)"),
        height=350,
    ))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CATEGORY BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Category Breakdown":

    st.markdown("""
    <div style='padding: 2rem 0 1rem 0;'>
        <div class='section-label'>Bonus Story — Translation Challenge</div>
        <div class='section-title'>Which Categories Are Hardest to Ship?</div>
    </div>
    """, unsafe_allow_html=True)

    cat_stats = (
        df_del.dropna(subset=["product_category_name_english"])
        .groupby("product_category_name_english")
        .agg(order_count=("order_id","count"),
             pct_late=("delivery_class", lambda x: (x != "On Time").mean() * 100),
             avg_review=("review_score","mean"),
             avg_delay=("delay_days","mean"))
        .reset_index()
        .query("order_count >= 200")
        .sort_values("pct_late", ascending=False)
    )

    n_cats = st.slider("Number of categories to display", 10, 30, 20)
    display_cats = cat_stats.head(n_cats).sort_values("pct_late", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display_cats["pct_late"],
        y=display_cats["product_category_name_english"],
        orientation="h",
        marker=dict(
            color=display_cats["avg_review"],
            colorscale=[[0,PALETTE["accent"]],[0.5,PALETTE["warning"]],[1,PALETTE["safe"]]],
            showscale=True,
            colorbar=dict(
                title="Avg Review",
                thickness=12,
                tickfont=dict(color="#A0A0A0"),
                title_font=dict(color="#A0A0A0"),
            ),
            line=dict(width=0),
        ),
        text=display_cats["pct_late"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(size=10, color="#A0A0A0"),
        customdata=display_cats[["order_count","avg_review","avg_delay"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Late rate: %{x:.1f}%<br>"
            "Orders: %{customdata[0]:,}<br>"
            "Avg review: %{customdata[1]:.2f}<br>"
            "Avg delay: %{customdata[2]:.1f}d<extra></extra>"
        ),
    ))
    fig.update_layout(**chart_layout(
        title=f"Late Delivery Rate by Product Category (Top {n_cats})",
        xaxis=dict(title="% Late Deliveries", gridcolor="#2A2A2A"),
        yaxis=dict(tickfont=dict(size=10), gridcolor="#2A2A2A"),
        height=max(400, n_cats * 28),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Trend Analysis":

    st.markdown("""
    <div style='padding: 2rem 0 1rem 0;'>
        <div class='section-label'>Temporal Analysis</div>
        <div class='section-title'>How Has Performance Changed Over Time?</div>
    </div>
    """, unsafe_allow_html=True)

    monthly = (
        df_del.groupby(["order_year","order_month"])
        .agg(total=("order_id","count"),
             pct_late=("delivery_class", lambda x: (x != "On Time").mean() * 100),
             avg_review=("review_score","mean"))
        .reset_index()
    )
    monthly["period"] = pd.to_datetime(
        monthly["order_year"].astype(str) + "-" +
        monthly["order_month"].astype(str).str.zfill(2)
    )
    monthly = monthly[monthly["total"] >= 80].sort_values("period").reset_index(drop=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly["period"], y=monthly["total"],
        name="Order Volume",
        marker=dict(color=PALETTE["muted"], opacity=0.2, line=dict(width=0)),
        hovertemplate="%{x|%b %Y}<br>Volume: %{y:,}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["period"], y=monthly["pct_late"],
        name="% Late", mode="lines+markers",
        line=dict(color=PALETTE["accent"], width=2.5),
        marker=dict(size=5),
        hovertemplate="%{x|%b %Y}<br>Late rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=monthly["period"], y=monthly["avg_review"],
        name="Avg Review", mode="lines+markers",
        line=dict(color=PALETTE["safe"], width=2.5, dash="dot"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="%{x|%b %Y}<br>Avg review: %{y:.2f}<extra></extra>",
    ), secondary_y=True)

    layout = chart_layout(
        title="Monthly Trend — Volume, Late Rate & Satisfaction",
        xaxis=dict(title="Month", gridcolor="#2A2A2A"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center",
                   font=dict(color="#A0A0A0"), bgcolor="rgba(0,0,0,0)"),
        height=460,
    )
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Order Volume", secondary_y=False,
                    gridcolor="#2A2A2A", color="#A0A0A0")
    fig.update_yaxes(title_text="% Late / Avg Review", secondary_y=True,
                    gridcolor="#2A2A2A", color="#A0A0A0", showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # Quarterly heatmap
    st.markdown("<hr style='border-color: #2A2A2A;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Quarterly Late Rate Heatmap</div>', unsafe_allow_html=True)

    quarterly = (
        df_del.groupby(["order_year","order_quarter"])
        .agg(pct_late=("delivery_class", lambda x: (x != "On Time").mean() * 100))
        .reset_index()
    )
    pivot = quarterly.pivot(index="order_year", columns="order_quarter", values="pct_late")
    pivot.columns = [f"Q{c}" for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,PALETTE["safe"]],[0.5,PALETTE["warning"]],[1,PALETTE["accent"]]],
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        hovertemplate="Year: %{y}<br>Quarter: %{x}<br>Late rate: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**chart_layout(
        title="Late Rate by Year and Quarter",
        xaxis=dict(title="Quarter"),
        yaxis=dict(title="Year"),
        height=280,
    ))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ML Prediction Model":

    st.markdown("""
    <div style='padding: 2rem 0 1rem 0;'>
        <div class='section-label'>Candidate's Choice — Predictive Intelligence</div>
        <div class='section-title'>Can We Predict Late Deliveries Before They Happen?</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training model..."):
        model, FEATURES, accuracy, roc_auc = train_model(df_del)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>Test Accuracy</div>
            <div class='kpi-value' style='color:{PALETTE["safe"]}'>{accuracy:.1%}</div>
            <div class='kpi-sub'>On held-out 20% test set</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>ROC-AUC Score</div>
            <div class='kpi-value' style='color:{PALETTE["safe"]}'>{roc_auc:.4f}</div>
            <div class='kpi-sub'>Discrimination ability</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='vcard'>
            <div class='kpi-label'>Model</div>
            <div class='kpi-value' style='color:#F0F0F0; font-size:1.2rem'>Random Forest</div>
            <div class='kpi-sub'>300 trees — max depth 10</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    FEATURE_LABELS = {
        "state_enc":            "Customer State",
        "order_month":          "Order Month",
        "order_dow":            "Day of Week",
        "order_quarter":        "Quarter",
        "promised_transit_days":"Promised Transit Days",
        "is_remote":            "Remote State Flag",
        "cross_state_del":      "Cross-State Delivery",
        "price_quartile":       "Price Segment",
        "freight_ratio":        "Freight-to-Price Ratio",
        "approval_lag_h":       "Approval Lag (hours)",
    }

    imp = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
        .sort_values("importance", ascending=True)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=imp["importance"], y=imp["label"],
            orientation="h",
            marker=dict(
                color=imp["importance"],
                colorscale=[[0,"#2A2A2A"],[1,PALETTE["safe"]]],
                showscale=False,
                line=dict(width=0),
            ),
            text=imp["importance"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
            textfont=dict(size=10, color="#A0A0A0"),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(**chart_layout(
            title="Feature Importance — What Predicts Late Delivery?",
            xaxis=dict(title="Importance Score", tickformat=".0%", gridcolor="#2A2A2A"),
            yaxis=dict(tickfont=dict(size=11), gridcolor="#2A2A2A"),
            height=420,
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Business value explanation
        st.markdown("""
        <div class='vcard' style='margin-top: 0;'>
            <div class='section-label'>Why This Matters</div>
            <br>
            <div style='font-family: DM Mono, monospace; font-size: 0.82rem; line-height: 2; color: #A0A0A0;'>
                This model answers the most valuable question a logistics company can ask:
                <br><br>
                <span style='color: #F0F0F0; font-weight: 500;'>"Can we flag an order as at-risk BEFORE it ships — not after the customer complains?"</span>
                <br><br>
                By predicting late deliveries at purchase time, Veridi can:
            </div>
            <br>
            <div class='finding-card'>
                <div class='finding-num'>Action 01</div>
                <div class='finding-text'>Route at-risk orders to faster carriers automatically</div>
            </div>
            <div class='finding-card'>
                <div class='finding-num'>Action 02</div>
                <div class='finding-text'>Send proactive customer notifications to set expectations</div>
            </div>
            <div class='finding-card'>
                <div class='finding-num'>Action 03</div>
                <div class='finding-text'>Offer compensation vouchers before complaints are filed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("<hr style='border-color: #2A2A2A;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Strategic Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size: 1.2rem;">What Veridi Should Do Next</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    recs = [
        ("01", "Audit Carrier SLAs", "Renegotiate contracts with carriers serving Alagoas, Maranhao, and Sergipe — the 3 worst-performing states."),
        ("02", "Recalibrate Windows", "Shorten promised delivery windows. Under-promising and over-delivering costs nothing and protects review scores."),
        ("03", "Deploy the Model", "Integrate the ML model into the order management system to flag at-risk orders at the moment of purchase."),
        ("04", "Remote Corridors", "Investigate dedicated logistics partners for the remote northern states — a structural problem requires a structural fix."),
    ]
    for col, (num, title, body) in zip([r1,r2,r3,r4], recs):
        with col:
            st.markdown(f"""
            <div class='vcard'>
                <div style='font-family: DM Mono, monospace; font-size: 0.6rem; color: {PALETTE["accent"]}; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.5rem;'>REC {num}</div>
                <div style='font-family: Syne, sans-serif; font-size: 0.9rem; font-weight: 700; color: #F0F0F0; margin-bottom: 0.5rem;'>{title}</div>
                <div style='font-family: DM Mono, monospace; font-size: 0.72rem; color: #6C757D; line-height: 1.7;'>{body}</div>
            </div>""", unsafe_allow_html=True)
