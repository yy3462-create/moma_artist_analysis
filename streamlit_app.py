# ===========================================
# 🎨 MoMA Gender Representation Dashboard (Pro v5: Age Analysis)
# ===========================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------------------------------------------
# ⚙️ Page setup
# -------------------------------------------
st.set_page_config(page_title="MoMA Gender Representation (Pro v5)", layout="wide")
st.title("The Visibility of Women Artists in MoMA’s Collection (Pro v5)")

# -------------------------------------------
# 📂 Load datasets
# -------------------------------------------
# 使用相对路径（基于当前脚本位置），便于在不同机器/目录运行
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(BASE_DIR, "data", "processed")


@st.cache_data
def load_data():
    by_year = pd.read_csv(f"{base}/gender_by_year_acq.csv")
    by_dept = pd.read_csv(f"{base}/gender_by_dept_acq.csv")
    by_create = pd.read_csv(f"{base}/gender_by_creation_year_artworks.csv")
    geo = pd.read_csv(f"{base}/female_geo.csv")

    # 清洗性别
    for df in [by_year, by_dept, by_create]:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.title()

    # 加载原始 artists & artworks（用于年龄计算）
    artists = pd.read_csv(os.path.join(BASE_DIR, "artists.csv"))
    artworks = pd.read_csv(os.path.join(BASE_DIR, "artworks.csv"))
    artists["Artist ID"] = artists["Artist ID"].astype(str).str.strip()
    artworks["Artist ID"] = artworks["Artist ID"].astype(str).str.strip()
    artworks["Artist ID"] = artworks["Artist ID"].str.split(",").str[0].str.strip()

    artists["Gender"] = artists["Gender"].astype(str).str.strip().str.title()
    artists["Birth Year"] = pd.to_numeric(artists["Birth Year"], errors="coerce")
    artworks["Acquisition Date"] = pd.to_datetime(
        artworks["Acquisition Date"], errors="coerce"
    )
    artworks["year_acq"] = artworks["Acquisition Date"].dt.year

    df_age = artworks.merge(
        artists[["Artist ID", "Gender", "Birth Year"]], on="Artist ID", how="left"
    )
    df_age = df_age.dropna(subset=["Birth Year", "year_acq"])
    df_age["acquisition_age"] = df_age["year_acq"] - df_age["Birth Year"]
    df_age = df_age[
        (df_age["acquisition_age"] > 10) & (df_age["acquisition_age"] < 100)
    ]
    return by_year, by_dept, by_create, geo, df_age


by_year, by_dept, by_create, geo, df_age = load_data()

# -------------------------------------------
# 🎛 Sidebar filters
# -------------------------------------------
st.sidebar.header("Filters")

min_year, max_year = int(by_year["year"].min()), int(by_year["year"].max())
year_range = st.sidebar.slider(
    "Acquisition Year Range", min_year, max_year, (1950, 2020)
)
departments = sorted(by_dept["Department"].dropna().unique().tolist())
dept_sel = st.sidebar.multiselect(
    "Select Departments", departments, default=departments[:3]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Yue Yao** · Columbia SIPA")

# ===========================================
# 1️⃣ Acquisition Trends
# ===========================================
st.subheader("Acquisition Trends: Gender Representation Over Time")

filtered = by_year.query("year >= @year_range[0] and year <= @year_range[1]")
fig1 = px.line(
    filtered,
    x="year",
    y="share",
    color="Gender",
    title="Overall Gender Representation in MoMA Acquisitions",
    labels={"share": "Share of Acquisitions", "year": "Year"},
    template="plotly_white",
)
fig1.update_traces(line=dict(width=3))
st.plotly_chart(fig1, use_container_width=True)

# --- Regression prediction 2030 ---
female = filtered[filtered["Gender"] == "Female"]
model = LinearRegression().fit(female[["year"]], female["share"])
pred_2030 = model.predict(np.array([[2030]]))[0]
st.markdown(f"**Predicted female share in 2030:** {pred_2030 * 100:.1f}%")

# ===========================================
# 2️⃣ Department Comparison
# ===========================================
st.subheader("Department-Level Trends")

fig2 = px.line(
    by_dept[
        (by_dept["Department"].isin(dept_sel))
        & (by_dept["Gender"] == "Female")
        & (by_dept["year"].between(year_range[0], year_range[1]))
    ],
    x="year",
    y="share",
    color="Department",
    title="Female Representation by Department",
    template="plotly_white",
)
fig2.update_traces(line=dict(width=3))
st.plotly_chart(fig2, use_container_width=True)

# ===========================================
# 3️⃣ Age at Time of Acquisition (Interactive Mosaic View)
# ===========================================
st.subheader("Age at Time of Acquisition — Interactive Mosaic View")

# --- 年龄段分箱 ---
bins = [0, 29, 39, 49, 59, 69, 79, 100]
labels = ["<30", "30–39", "40–49", "50–59", "60–69", "70+", "80+"]
df_age["age_group"] = pd.cut(
    df_age["acquisition_age"], bins=bins, labels=labels, include_lowest=True
)

# --- 每年 × 年龄段 × 性别 统计 ---
age_share = (
    df_age.groupby(["year_acq", "age_group", "Gender"]).size().reset_index(name="count")
)
age_share["total"] = age_share.groupby(["year_acq", "age_group"])["count"].transform(
    "sum"
)
age_share["share"] = age_share["count"] / age_share["total"]

# --- 平均年龄差距 ---
avg_age_m = df_age[df_age["Gender"] == "Male"]["acquisition_age"].mean()
avg_age_f = df_age[df_age["Gender"] == "Female"]["acquisition_age"].mean()
age_gap = avg_age_m - avg_age_f

col1, col2 = st.columns(2)
col1.metric("Average Age (Male)", f"{avg_age_m:.1f} years")
col2.metric("Average Age (Female)", f"{avg_age_f:.1f} years")
st.markdown(
    f"**Gender Age Gap at Acquisition:** {age_gap:.1f} years (positive = men older)"
)


fig_ani = px.bar(
    age_share.query("Gender == 'Female' and year_acq >= 1950 and year_acq <= 2020"),
    x="age_group",
    y="share",
    animation_frame="year_acq",
    range_y=[0, 1],
    title="Animated Age Distribution of Female Artists (1950–2020)",
    template="plotly_white",
)
st.plotly_chart(fig_ani, use_container_width=True)
# ===========================================
# 3️⃣.5 Age at Time of Creation — Comparison
# ===========================================
st.subheader("Age at Time of Creation — Comparing Artistic vs Institutional Timelines")


# 提取作品创作年份
def parse_creation_year(txt):
    if pd.isna(txt):
        return np.nan
    s = str(txt).lower()
    # 提取四位年份，如 1950, 1992
    years = [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", s)]
    if not years:
        return np.nan
    return int(np.mean(years))


import re

df_age["year_created"] = (
    df_age.get("Date").apply(parse_creation_year)
    if "Date" in df_age.columns
    else np.nan
)
df_age = df_age.dropna(subset=["year_created"])
df_age["creation_age"] = df_age["year_created"] - df_age["Birth Year"]

# 去掉异常（负数或>100）
df_age = df_age[(df_age["creation_age"] > 10) & (df_age["creation_age"] < 100)]

# 平均年龄差
avg_create_m = df_age[df_age["Gender"] == "Male"]["creation_age"].mean()
avg_create_f = df_age[df_age["Gender"] == "Female"]["creation_age"].mean()
create_gap = avg_create_m - avg_create_f

col1, col2 = st.columns(2)
col1.metric("Avg Age at Creation (Male)", f"{avg_create_m:.1f} years")
col2.metric("Avg Age at Creation (Female)", f"{avg_create_f:.1f} years")

st.markdown(
    f"**Gender Gap in Creation Age:** {create_gap:.1f} years (positive = men older)"
)

# 可视化：动画比较年龄结构变化
age_create_bins = [10, 19, 29, 39, 49, 59, 69, 79, 89, 100]
age_create_labels = [
    "<20",
    "20–29",
    "30–39",
    "40–49",
    "50–59",
    "60–69",
    "70–79",
    "80–89",
    "90+",
]

df_age["creation_age_group"] = pd.cut(
    df_age["creation_age"],
    bins=age_create_bins,
    labels=age_create_labels,
    include_lowest=True,
)

# 每年创作的作品中不同年龄段的分布
create_share = (
    df_age.groupby(["year_created", "creation_age_group", "Gender"])
    .size()
    .reset_index(name="count")
)
create_share["total"] = create_share.groupby(["year_created", "Gender"])[
    "count"
].transform("sum")
create_share["share"] = create_share["count"] / create_share["total"]

# Animated bar
fig_create = px.bar(
    create_share.query(
        "Gender == 'Female' and year_created >= 1900 and year_created <= 2020"
    ),
    x="creation_age_group",
    y="share",
    animation_frame="year_created",
    range_y=[0, 1],
    title="Animated Age Distribution of Female Artists at Time of Creation (1900–2020)",
    labels={"creation_age_group": "Age at Creation", "share": "Female Share"},
    color="creation_age_group",
    color_discrete_sequence=px.colors.sequential.Magenta,
    template="plotly_white",
)
st.plotly_chart(fig_create, use_container_width=True)

# --- Recognition lag for both genders ---
lag_f = avg_age_f - avg_create_f
lag_m = avg_age_m - avg_create_m
lag_gap = lag_m - lag_f

st.markdown(
    f"""
    **Insight:**  
    • Female artists created their works at an average age of **{avg_create_f:.1f}**, and MoMA acquired them at **{avg_age_f:.1f}**,  
      implying an institutional delay of **{lag_f:.1f} years**.  
    • Male artists created their works at **{avg_create_m:.1f}**, with acquisition around **{avg_age_m:.1f}**,  
      a delay of **{lag_m:.1f} years**.  
    • Overall, the institutional recognition lag is **{lag_gap:+.1f} years longer for men** (positive = men wait longer).  

   
    """
)

# ===========================================
# 4️⃣ Global Geography
# ===========================================
st.subheader("Global Distribution of Women Artists")

if not geo.empty:
    color_col = "female_share" if "female_share" in geo.columns else "female_artists"

    fig_geo = px.choropleth(
        geo,
        locations="Nationality",  # 列名必须和 CSV 一致
        locationmode="country names",
        color=color_col,
        color_continuous_scale="Reds",
        title="Geographic Distribution of Women Artists (MoMA Collection)",
    )
    st.plotly_chart(fig_geo, use_container_width=True)
else:
    st.info("No nationality data available for mapping.")


# ===========================================
# 5️⃣ Institutional Equity Explorer (Interactive)
# ===========================================
st.markdown("## Institutional Equity Explorer")

# --- Forecast simulator
st.markdown("### Forecast Simulator — Explore 2030 and Beyond")
female = by_year[by_year["Gender"] == "Female"]
model = LinearRegression().fit(female[["year"]], female["share"])

future_year = st.slider("Select forecast year", 2020, 2050, 2030, step=1)
pred_future = model.predict(np.array([[future_year]]))[0]

st.markdown(
    f"#### Predicted female share in **{future_year}**: **{pred_future * 100:.1f}%**"
)

# --- Forecast line chart
years_ext = np.arange(1950, future_year + 1)
pred_trend = model.predict(years_ext.reshape(-1, 1))
fig_pred = px.line(
    x=years_ext,
    y=pred_trend,
    labels={"x": "Year", "y": "Predicted Female Share"},
    title="Forecasted Trajectory of Female Representation",
    template="plotly_white",
)
fig_pred.add_scatter(
    x=female["year"],
    y=female["share"],
    mode="markers",
    name="Historical Data",
    marker_color="deeppink",
)
st.plotly_chart(fig_pred, use_container_width=True)


# dynamic color cue
def color_for_value(val, low, high):
    pct = (val - low) / (high - low)
    pct = max(0, min(1, pct))
    hue = 120 - pct * 120  # green→red
    return f"hsl({hue:.0f},70%,45%)"


st.markdown("---")
st.caption(
    "Data Source: The Museum of Modern Art (MoMA) Public Dataset — Analysis and Visualization by Yue Yao"
)
