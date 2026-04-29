# EducatEd Choice - Professional Horizontal Streamlit Dashboard
# Improved layout with filters + side-by-side charts + stronger dashboard experience

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="EducatEd Choice Dashboard",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("master_school_table_v5_2023_24.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Make sure master_school_table_v5_2023_24.csv is in the same folder.")
    st.stop()

# =========================
# FEATURES
# =========================
features = [
    "grad_rate",
    "cohort_size",
    "sat_total",
    "mobility_rate",
    "mobility_count",
    "discipline_percent",
    "hope_eligible_percent"
]

required_cols = features + ["school_name"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.title("Filters")

# School type filter if available
if "SCHOOL_TYPE_TEXT" in df.columns:
    school_types = sorted(df["SCHOOL_TYPE_TEXT"].dropna().unique())
    selected_type = st.sidebar.multiselect(
        "School Type",
        school_types,
        default=school_types
    )
    df = df[df["SCHOOL_TYPE_TEXT"].isin(selected_type)]

# School level filter if available
if "SCHOOL_LEVEL" in df.columns:
    school_levels = sorted(df["SCHOOL_LEVEL"].dropna().unique())
    selected_level = st.sidebar.multiselect(
        "School Level",
        school_levels,
        default=school_levels
    )
    df = df[df["SCHOOL_LEVEL"].isin(selected_level)]

# City filter if available
if "LCITY" in df.columns:
    cities = sorted(df["LCITY"].dropna().unique())
    selected_city = st.sidebar.multiselect(
        "City",
        cities,
        default=cities
    )
    df = df[df["LCITY"].isin(selected_city)]

# Graduation rate filter
if "grad_rate" in df.columns:
    min_grad = float(df["grad_rate"].min())
    max_grad = float(df["grad_rate"].max())
    grad_range = st.sidebar.slider(
        "Graduation Rate Range",
        min_value=min_grad,
        max_value=max_grad,
        value=(min_grad, max_grad)
    )
    df = df[(df["grad_rate"] >= grad_range[0]) & (df["grad_rate"] <= grad_range[1])]

# =========================
# HEADER
# =========================
st.title("EducatEd Choice Dashboard")
st.write(
    "Helping families compare schools using academic performance, mobility, and PCA-driven school matching."
)

# =========================
# KPI CARDS
# =========================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Schools", f"{len(df):,}")

with k2:
    st.metric("Avg Graduation", f"{df['grad_rate'].mean():.1f}%")

with k3:
    st.metric("Avg SAT", f"{df['sat_total'].mean():.0f}")

with k4:
    st.metric("Avg Mobility", f"{df['mobility_rate'].mean():.1f}%")

# =========================
# SECTION 1: RELATIONSHIPS
# =========================
st.subheader("How School Factors Relate")

left, right = st.columns(2)

with left:
    corr = df[features].corr().round(2)
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with right:
    fig_sat = px.scatter(
        df,
        x="grad_rate",
        y="sat_total",
        hover_name="school_name",
        title="Graduation Rate vs SAT Score"
    )
    st.plotly_chart(fig_sat, use_container_width=True)

# =========================
# SECTION 2: MOBILITY + PCA
# =========================
second_left, second_right = st.columns(2)

with second_left:
    fig_mobility = px.scatter(
        df,
        x="mobility_rate",
        y="grad_rate",
        hover_name="school_name",
        title="Mobility Rate vs Graduation Rate"
    )
    st.plotly_chart(fig_mobility, use_container_width=True)

# PCA preparation
pca_input = df[features].copy()
pca_input = pca_input.fillna(pca_input.median(numeric_only=True))

X_scaled = StandardScaler().fit_transform(pca_input)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = explained_var.cumsum()

with second_right:
    variance_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(explained_var))],
        "Individual Variance": explained_var,
        "Cumulative Variance": cum_var
    })

    fig_var = px.bar(
        variance_df,
        x="Principal Component",
        y="Individual Variance",
        text_auto=".1%",
        title="Explained Variance"
    )

    fig_var.add_scatter(
        x=variance_df["Principal Component"],
        y=variance_df["Cumulative Variance"],
        mode="lines+markers",
        name="Cumulative Variance"
    )

    st.plotly_chart(fig_var, use_container_width=True)

# =========================
# SECTION 3: SCHOOL MATCHING MAP
# =========================
st.subheader("School Matching Map")

pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["school_name"] = df["school_name"].reset_index(drop=True)
pca_df["grad_rate"] = df["grad_rate"].reset_index(drop=True)
pca_df["sat_total"] = df["sat_total"].reset_index(drop=True)
pca_df["mobility_rate"] = df["mobility_rate"].reset_index(drop=True)

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="grad_rate",
    hover_name="school_name",
    hover_data={
        "sat_total": True,
        "mobility_rate": True
    },
    title="Performance vs Stability"
)

st.plotly_chart(fig_pca, use_container_width=True)

st.info(
    "PC1 mainly represents academic performance. PC2 helps explain mobility and structural differences. This helps families compare schools beyond a single score."
)

# =========================
# DATASET PREVIEW
# =========================
st.subheader("Dataset Explorer")
st.dataframe(df.head(50), use_container_width=True)