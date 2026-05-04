import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import folium
from streamlit_folium import st_folium

# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="EducatEd Choice Dashboard",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
/* Main app background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Titles */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 28px;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 28px;
    margin-bottom: 12px;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.32);
    min-height: 138px;
}
.kpi-title {
    font-size: 14px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 700;
}
.kpi-value {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 10px;
}
.kpi-caption {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 8px;
}

/* Text cards */
.info-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #e5e7eb;
    padding: 20px;
    border-radius: 16px;
    line-height: 1.6;
    margin-top: 14px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.25);
}

/* Make dataframe area readable */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">EducatEd Choice Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A data-driven school matching dashboard using PCA, academic outcomes, and school stability indicators.</div>',
    unsafe_allow_html=True
)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("master_school_table_v5_2023_24.csv")
    all_schools = pd.read_csv("all_schools.csv")
    all_schools = all_schools.dropna(subset=["lat", "lon"])
    return df, all_schools

try:
    df, all_schools = load_data()
except:
    st.error("Error loading datasets. Check your CSV files.")
    st.stop()
# =========================
# REQUIRED FEATURES
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

missing_cols = [col for col in features + ["school_name"] if col not in df.columns]
if missing_cols:
    st.error(f"Missing required column(s): {', '.join(missing_cols)}")
    st.stop()

# Convert numeric columns safely
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# PCA MODEL
# =========================
df_pca = df[features].copy()
df_pca = df_pca.fillna(df_pca.median(numeric_only=True))

X_scaled = StandardScaler().fit_transform(df_pca)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = explained_var.cumsum()

pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["school_name"] = df["school_name"]
pca_df["grad_rate"] = df["grad_rate"]
pca_df["sat_total"] = df["sat_total"]
pca_df["hope_eligible_percent"] = df["hope_eligible_percent"]
pca_df["mobility_rate"] = df["mobility_rate"]


def assign_profile(row):
    if row["PC1"] >= 0 and row["PC2"] >= 0:
        return "High Performance + Stable"
    if row["PC1"] >= 0 and row["PC2"] < 0:
        return "High Performance + Less Stable"
    if row["PC1"] < 0 and row["PC2"] >= 0:
        return "Lower Performance + Stable"
    return "Lower Performance + Less Stable"


pca_df["school_profile"] = pca_df.apply(assign_profile, axis=1)

# =========================
# SIDEBAR MENU
# =========================
view = st.sidebar.radio(
    "Select Section",
    ["Dashboard", "Dataset"]
)
# =========================
# SIDEBAR FILTERS (ADD THIS)
# =========================

selected_schools = st.sidebar.multiselect(
    "Select Schools",
    df["school_name"].dropna().unique(),
    default=[]
)

selected_city = st.sidebar.selectbox(
    "Select City",
    ["All"] + sorted(all_schools["city"].dropna().unique().tolist())
)

selected_type = st.sidebar.selectbox(
    "Select School Type",
    ["All"] + sorted(all_schools["school_type"].dropna().unique().tolist())
)

grad_min, grad_max = st.sidebar.slider(
    "Graduation Rate Range",
    0, 100, (0, 100)
)

sat_min, sat_max = st.sidebar.slider(
    "SAT Score Range",
    int(df["sat_total"].min()),
    int(df["sat_total"].max()),
    (int(df["sat_total"].min()), int(df["sat_total"].max()))
)
# =========================
# FILTERS
# =========================

filtered_df = df.copy()

# School selector
if selected_schools:
    filtered_df = filtered_df[
        filtered_df["school_name"].isin(selected_schools)
    ]

# City filter (comes from all_schools, so we match names)
if selected_city != "All":
    city_schools = all_schools[all_schools["city"] == selected_city]["school_name"]
    filtered_df = filtered_df[
        filtered_df["school_name"].isin(city_schools)
    ]

# School type filter
if selected_type != "All":
    type_schools = all_schools[all_schools["school_type"] == selected_type]["school_name"]
    filtered_df = filtered_df[
        filtered_df["school_name"].isin(type_schools)
    ]

# Numeric filters
filtered_df = filtered_df[
    (filtered_df["grad_rate"] >= grad_min) &
    (filtered_df["grad_rate"] <= grad_max) &
    (filtered_df["sat_total"] >= sat_min) &
    (filtered_df["sat_total"] <= sat_max)
]
# =========================
# DASHBOARD PAGE
# =========================
if view == "Dashboard":
    filtered_df = df[
        (df["school_name"].isin(selected_schools)) &
        (df["grad_rate"] >= grad_min) &
        (df["grad_rate"] <= grad_max) &
        (df["sat_total"] >= sat_min) &
        (df["sat_total"] <= sat_max)
    ]
    # KPI values
    total_schools = filtered_df["school_name"].nunique()
    avg_grad = filtered_df["grad_rate"].mean()
    avg_sat = filtered_df["sat_total"].mean()
    avg_mobility = filtered_df["mobility_rate"].mean()
    pc1_pct = explained_var[0] * 100
    pc2_pct = explained_var[1] * 100
    pc12_pct = (explained_var[0] + explained_var[1]) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Schools</div>
            <div class="kpi-value">{total_schools:,}</div>
            <div class="kpi-caption">Schools included in the analysis</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Graduation Rate</div>
            <div class="kpi-value">{avg_grad:.1f}%</div>
            <div class="kpi-caption">Overall academic outcome</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg SAT Score</div>
            <div class="kpi-value">{avg_sat:.0f}</div>
            <div class="kpi-caption">Average SAT total score</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Mobility Rate</div>
            <div class="kpi-value">{avg_mobility:.1f}%</div>
            <div class="kpi-caption">Lower rate suggests more stability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">School Matching Map</div>', unsafe_allow_html=True)

    # =========== MAP
    st.markdown("### School Map")

    if not all_schools.empty:
        map_df = all_schools.copy()

    # Apply filters safely
    if selected_city != "All" and "city" in map_df.columns:
        map_df = map_df[map_df["city"] == selected_city]

    if selected_type != "All" and "school_type" in map_df.columns:
        map_df = map_df[map_df["school_type"] == selected_type]

    map_df = map_df.dropna(subset=["lat","lon"])

    if map_df.empty:
        st.warning("No map data for selected filters")
    else:
        # Drop missing coordinates (VERY IMPORTANT)
        map_df = map_df.dropna(subset=["lat", "lon"])

        if map_df.empty:
            st.warning("No valid coordinates to display on map")
        else:
            m = folium.Map(
            location=[map_df["lat"].mean(), map_df["lon"].mean()],
            zoom_start=11
            )

            for _, row in map_df.iterrows():
                color = {
                    "Public": "blue",
                    "Private": "red",
                    "Charter": "green"
                }.get(row.get("school_type", ""), "gray")

                folium.Marker(

                [row["lat"],row["lon"]],
                popup=f"{row['school_name']} ({row.get('city','')})",
                icon=folium.Icon(color=color)
            ).add_to(m)

        st_folium(m, width=None, height=500)

    # ===========

    fig_map = px.scatter(
        pca_df[pca_df["school_name"].isin(filtered_df["school_name"])],
        x="PC1",
        y="PC2",
        color="grad_rate",
        hover_name="school_name",
        hover_data={
            "school_profile": True,
            "grad_rate": ":.1f",
            "sat_total": ":,.0f",
            "hope_eligible_percent": ":.1f",
            "mobility_rate": ":.1f",
            "PC1": False,
            "PC2": False
        },
        labels={
            "PC1": f"Academic Performance Pattern - PC1 ({pc1_pct:.1f}%)",
            "PC2": f"Stability / Structure Pattern - PC2 ({pc2_pct:.1f}%)",
            "grad_rate": "Graduation Rate",
            "school_profile": "School Profile",
            "sat_total": "SAT Total",
            "hope_eligible_percent": "HOPE Eligible (%)",
            "mobility_rate": "Mobility Rate (%)"
        },
        color_continuous_scale="Viridis",
        opacity=0.82,
        title="School Matching Map: Performance vs Stability"
        )

    fig_map.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig_map.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig_map.update_traces(marker=dict(size=8, line=dict(width=0.6, color="white")))
    fig_map.update_layout(
        height=610,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111827"),
        title_font=dict(size=20, color="#111827"),
        margin=dict(l=40, r=40, t=70, b=50)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="section-title">Key Drivers</div>', unsafe_allow_html=True)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=features,
            columns=[f"PC{i+1}" for i in range(len(features))]
        )

        loadings_df = loadings["PC1"].sort_values(key=abs).reset_index()
        loadings_df.columns = ["Feature", "Contribution"]

        def feature_group(feature):
            if feature in ["grad_rate", "sat_total", "hope_eligible_percent"]:
                return "Academic Performance"
            if feature == "cohort_size":
                return "School Scale"
            return "School Structure"

        loadings_df["Group"] = loadings_df["Feature"].apply(feature_group)

        fig_loadings = px.bar(
            loadings_df,
            x="Contribution",
            y="Feature",
            color="Group",
            orientation="h",
            title="Feature Contributions to PC1",
            color_discrete_map={
                "Academic Performance": "#38bdf8",
                "School Structure": "#fb923c",
                "School Scale": "#94a3b8"
            }
        )
        fig_loadings.update_layout(
            height=500,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#111827"),
            title_font=dict(size=18, color="#111827"),
            xaxis_title="Contribution Strength",
            yaxis_title=""
        )
        st.plotly_chart(fig_loadings, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="section-title">Explained Variance</div>', unsafe_allow_html=True)

        variance_df = pd.DataFrame({
            "Principal Component": [f"PC{i+1}" for i in range(len(explained_var))],
            "Individual Variance": explained_var,
            "Cumulative Variance": cum_var
        })

        fig_var = px.bar(
            variance_df,
            x="Principal Component",
            y="Individual Variance",
            title="Explained Variance by Principal Component",
            text_auto=".1%"
        )
        fig_var.add_scatter(
            x=variance_df["Principal Component"],
            y=variance_df["Cumulative Variance"],
            mode="lines+markers",
            name="Cumulative Variance"
        )
        fig_var.update_layout(
            height=500,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#111827"),
            title_font=dict(size=18, color="#111827"),
            yaxis_tickformat=".0%",
            yaxis_title="Variance Explained"
        )
        st.plotly_chart(fig_var, use_container_width=True)

    # =========================
    # KEY RELATIONSHIPS
    # =========================
    st.markdown('<div class="section-title">Key Relationships</div>', unsafe_allow_html=True)

    scatter_col, bar_col = st.columns(2)

    with scatter_col:
        # Exclude structural outlier: schools with mobility > 30% are alternative programs
        # with fundamentally different student populations that skew the trend line
        scatter_data = pca_df[(pca_df["mobility_rate"] < 30)&(pca_df["school_name"].isin(filtered_df["school_name"]))].copy()
        if len(scatter_data) < 2:
            st.warning("Not enough data to display this chart. Adjust filters.")
        else:
            fit = np.polyfit(scatter_data["mobility_rate"], scatter_data["grad_rate"], 1)

            x_line = np.linspace(
                scatter_data["mobility_rate"].min(),
                scatter_data["mobility_rate"].max(),
                100
            )
            y_line = np.polyval(fit, x_line)

            fig_scatter = px.scatter(
                scatter_data,
                x="mobility_rate",
                y="grad_rate",
                hover_name="school_name",
                color="grad_rate",
                color_continuous_scale="Blues",
                title="Mobility Rate vs. Graduation Rate",
            )

            fig_scatter.add_scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="#f59e0b", dash="dash"),
                name="Trend"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with bar_col:
        # Split schools at median for each metric, compare avg outcomes across groups
        med_mob  = pca_df["mobility_rate"].median()
        med_grad = pca_df["grad_rate"].median()
        med_hope = pca_df["hope_eligible_percent"].median()

        filtered_pca = pca_df[pca_df["school_name"].isin(filtered_df["school_name"])]
    

        subgroups = {
            "High Mobility":  pca_df[pca_df["mobility_rate"]          >= med_mob],
            "Low Mobility":   pca_df[pca_df["mobility_rate"]          <  med_mob],
            "High Grad Rate": pca_df[pca_df["grad_rate"]              >= med_grad],
            "Low Grad Rate":  pca_df[pca_df["grad_rate"]              <  med_grad],
            "High HOPE":      pca_df[pca_df["hope_eligible_percent"]  >= med_hope],
            "Low HOPE":       pca_df[pca_df["hope_eligible_percent"]  <  med_hope],
        }
        bar_rows = []
        for label, subset in subgroups.items():
            bar_rows.append({
                "Subgroup": label,
                "Metric":   "Avg Grad Rate %",
                "Value":    round(subset["grad_rate"].mean(), 1),
            })
            bar_rows.append({
                "Subgroup": label,
                "Metric":   "Avg HOPE Elig %",
                "Value":    round(subset["hope_eligible_percent"].mean(), 1),
            })

        bar_df = pd.DataFrame(bar_rows)

        fig_bar = px.bar(
            bar_df,
            x="Subgroup",
            y="Value",
            color="Metric",
            barmode="group",
            title="Avg Outcomes by Subgroup (split at median)",
            text="Value",
            color_discrete_map={
                "Avg Grad Rate %": "#38bdf8",
                "Avg HOPE Elig %": "#fb923c",
            },
        )
        fig_bar.update_traces(
            textposition="outside",
            textfont=dict(size=10, color="#111827"),
            marker_line_width=0,
        )
        fig_bar.update_layout(
            height=460,
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#111827"),
            title_font=dict(size=16, color="#111827"),
            yaxis=dict(title="Percentage (%)", range=[0, 115]),
            xaxis=dict(title=""),
            legend=dict(title="", orientation="h", y=-0.2),
            margin=dict(l=20, r=20, t=60, b=70),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <b>How to read this dashboard:</b><br><br>
        The top cards summarize the dataset in simple numbers. The map turns several school factors into two main patterns:
        academic performance and school stability/structure. Schools closer together have more similar profiles, while schools farther apart
        have more different patterns. The driver chart explains which variables influence the map the most, and the explained variance chart
        shows how much information PCA keeps from the original dataset.
        The Mobility vs. Graduation Rate scatter plot shows how student turnover relates to graduation outcomes.
        Hover over any dot to see the school name and values, and use the dashed trend line to read the overall direction.
        The Subgroup Comparison chart splits schools at the median for each metric and compares average graduation rate and HOPE eligibility side by side,
        making it easy to see how much outcomes differ between groups.
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # DATASET PAGE
    # =========================
elif view == "Dataset":
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    rows, cols = df.shape
    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Rows</div>
            <div class="kpi-value">{rows:,}</div>
            <div class="kpi-caption">School-level records</div>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Columns</div>
            <div class="kpi-value">{cols:,}</div>
            <div class="kpi-caption">Available dataset fields</div>
        </div>
        """, unsafe_allow_html=True)

    with d3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">PCA Features</div>
            <div class="kpi-value">{len(features)}</div>
            <div class="kpi-caption">Variables used for analysis</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        This dataset includes school-level academic and structural variables such as graduation rate, SAT score,
        mobility, discipline, cohort size, and HOPE eligibility. These variables are used to build the PCA model
        and support the school matching logic shown in the dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Preview Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown('<div class="section-title">PCA Feature Summary</div>', unsafe_allow_html=True)
    summary = df[features].describe().T
    st.dataframe(summary, use_container_width=True)
