import os
import re
import numpy as np
import pandas as pd

# ============ 路径 ============
# 使用相对路径（基于当前脚本位置），便于在不同机器/目录运行
BASE = os.path.dirname(os.path.abspath(__file__))
ARTISTS_CSV = os.path.join(BASE, "artists.csv")
ARTWORKS_CSV = os.path.join(BASE, "artworks.csv")
OUTDIR = os.path.join(BASE, "data", "processed")
os.makedirs(OUTDIR, exist_ok=True)

# ============ 读取 ============
artists = pd.read_csv(ARTISTS_CSV)
artworks = pd.read_csv(ARTWORKS_CSV)

# ============ 统一主键：Artist ID ============
artists["Artist ID"] = artists["Artist ID"].astype(str).str.strip()
artworks["Artist ID"] = artworks["Artist ID"].astype(str).str.strip()
# 如有“1234, 5678”取第一个（简化版本，后期可做“爆炸”处理）
artworks["Artist ID"] = artworks["Artist ID"].str.split(",").str[0].str.strip()


# ============ Gender 归一化 ============
def norm_gender(x: str) -> str:
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "unknown", "unspecified", "n/a", "null"):
        return "Unknown"
    # 常见写法
    if s.startswith("f"):
        return "Female"
    if s.startswith("m"):
        return "Male"
    if "non" in s or "nb" in s or "non-b" in s or "binary" in s:
        return "Non-binary"
    return s.title()


artists["Gender"] = artists["Gender"].apply(norm_gender)
# 若你希望“只统计已知性别”，可去掉 Unknown：
# artists = artists[artists["Gender"].isin(["Male","Female","Non-binary"])]

# 补充常用字段：出生年、国籍，转数字
artists["Birth Year"] = pd.to_numeric(artists.get("Birth Year"), errors="coerce")
artists["Nationality"] = artists.get("Nationality")

# ============ 合并艺术家信息到作品 ============
df = artworks.merge(
    artists[["Artist ID", "Gender", "Birth Year", "Nationality"]],
    on="Artist ID",
    how="left",
)

# ============ 时间口径 1：收藏年份（Acquisition Year） ============
df["Acquisition Date"] = pd.to_datetime(df.get("Acquisition Date"), errors="coerce")
df["year_acq"] = df["Acquisition Date"].dt.year

# —— 严谨计数口径：
# “每位艺术家在同一年只计一次”，避免某艺术家在同年多件作品导致重复计数
df_acq_unique = df.dropna(subset=["year_acq"]).drop_duplicates(
    ["Artist ID", "year_acq"]
)

# 1A) 按年 × 性别
gender_by_year_acq = (
    df_acq_unique.groupby(["year_acq", "Gender"])
    .size()
    .reset_index(name="count")
    .rename(columns={"year_acq": "year"})
)
gender_by_year_acq["total"] = gender_by_year_acq.groupby("year")["count"].transform(
    "sum"
)
gender_by_year_acq["share"] = gender_by_year_acq["count"] / gender_by_year_acq["total"]

# 1B) 按年 × 部门 × 性别（同样按“艺术家-年”去重后的视角）
df_acq_unique = df_acq_unique.dropna(subset=["Department"])
gender_by_dept_acq = (
    df_acq_unique.groupby(["year_acq", "Department", "Gender"])
    .size()
    .reset_index(name="count")
    .rename(columns={"year_acq": "year"})
)
gender_by_dept_acq["total"] = gender_by_dept_acq.groupby(["year", "Department"])[
    "count"
].transform("sum")
gender_by_dept_acq["share"] = gender_by_dept_acq["count"] / gender_by_dept_acq["total"]

# ============ 时间口径 2：艺术家出生年（Birth Year） ============
# 这里按“人”为单位（每位艺术家只算一次），看不同出生年里男女比例
artists_birth = artists.dropna(subset=["Birth Year"]).copy()
gender_by_birth_year = (
    artists_birth.groupby(["Birth Year", "Gender"])
    .size()
    .reset_index(name="count")
    .rename(columns={"Birth Year": "birth_year"})
)
gender_by_birth_year["total"] = gender_by_birth_year.groupby("birth_year")[
    "count"
].transform("sum")
gender_by_birth_year["share"] = (
    gender_by_birth_year["count"] / gender_by_birth_year["total"]
)

# 也可以做“出生年代（十年为单位）”
artists_birth["birth_decade"] = (artists_birth["Birth Year"] // 10) * 10
gender_by_birth_decade = (
    artists_birth.groupby(["birth_decade", "Gender"]).size().reset_index(name="count")
)
gender_by_birth_decade["total"] = gender_by_birth_decade.groupby("birth_decade")[
    "count"
].transform("sum")
gender_by_birth_decade["share"] = (
    gender_by_birth_decade["count"] / gender_by_birth_decade["total"]
)


# ============ 时间口径 3：作品创作年（Creation Year） ============
# MoMA 的作品日期字段常在 artworks["Date"]，包含 “c. 1950”, “1950–52”, “1990s” 等
def parse_creation_year(txt) -> float | None:
    if pd.isna(txt):
        return np.nan
    s = str(txt).lower().strip()
    # 1990s → 1990
    m_decade = re.search(r"(\d{3})0s", s)
    if m_decade:
        try:
            return int(m_decade.group(1) + "0")
        except Exception:
            pass
    # 抓取所有四位数年份（1000–2099范围内）
    years = [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", s)]
    if not years:
        return np.nan
    # 若有范围，取中位数；若单一年份，直接用
    return int(round(np.mean(years)))


df["year_created"] = (
    df.get("Date").apply(parse_creation_year) if "Date" in df.columns else np.nan
)

# 3A) 以“作品”为单位：每年创作的作品中性别占比
# （此口径下，一个艺术家某年多件作品会多次计数）
gender_by_creation_year_artworks = (
    df.dropna(subset=["year_created"])
    .groupby(["year_created", "Gender"])
    .size()
    .reset_index(name="count")
    .rename(columns={"year_created": "year"})
)
gender_by_creation_year_artworks["total"] = gender_by_creation_year_artworks.groupby(
    "year"
)["count"].transform("sum")
gender_by_creation_year_artworks["share"] = (
    gender_by_creation_year_artworks["count"]
    / gender_by_creation_year_artworks["total"]
)

# 3B) 以“人”为单位：某年“有作品创作”的唯一艺术家人数占比
df_created_unique = df.dropna(subset=["year_created"]).drop_duplicates(
    ["Artist ID", "year_created"]
)
gender_by_creation_year_artists = (
    df_created_unique.groupby(["year_created", "Gender"])
    .size()
    .reset_index(name="count")
    .rename(columns={"year_created": "year"})
)
gender_by_creation_year_artists["total"] = gender_by_creation_year_artists.groupby(
    "year"
)["count"].transform("sum")
gender_by_creation_year_artists["share"] = (
    gender_by_creation_year_artists["count"] / gender_by_creation_year_artists["total"]
)

# ============ 附：女性艺术家国籍分布（供地图用） ============
female_geo = (
    df_acq_unique[df_acq_unique["Gender"] == "Female"]
    .groupby("Nationality")["Artist ID"]
    .nunique()
    .reset_index(name="female_artists")
)
# （如果要显示全部性别，可切换为不筛选 Gender）

# ============ 导出 ============
gender_by_year_acq.to_csv(f"{OUTDIR}/gender_by_year_acq.csv", index=False)
gender_by_dept_acq.to_csv(f"{OUTDIR}/gender_by_dept_acq.csv", index=False)
gender_by_birth_year.to_csv(f"{OUTDIR}/gender_by_birth_year.csv", index=False)
gender_by_birth_decade.to_csv(f"{OUTDIR}/gender_by_birth_decade.csv", index=False)
gender_by_creation_year_artworks.to_csv(
    f"{OUTDIR}/gender_by_creation_year_artworks.csv", index=False
)
gender_by_creation_year_artists.to_csv(
    f"{OUTDIR}/gender_by_creation_year_artists.csv", index=False
)
female_geo.to_csv(f"{OUTDIR}/female_geo.csv", index=False)

print("✅ Saved to:", OUTDIR)
for f in [
    "gender_by_year_acq.csv",
    "gender_by_dept_acq.csv",
    "gender_by_birth_year.csv",
    "gender_by_birth_decade.csv",
    "gender_by_creation_year_artworks.csv",
    "gender_by_creation_year_artists.csv",
    "female_geo.csv",
]:
    print(" -", f)
