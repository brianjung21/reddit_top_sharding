# streamlit_multi_sector.py
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# -----------------------------
# Strings (EN + KO)
# -----------------------------
STR = {
    "en": {
        "title": "Reddit Brand Mentions — Multi-Sector Dashboard",
        "sector": "Sector",
        "freq_daily": "Daily",
        "freq_weekly": "Weekly",
        "metric_raw": "Raw counts",
        "metric_sov": "Share of Voice (%)",
        "no_data": "No data in the selected date range.",
        "pick_brands": "Pick at least one brand to plot from the sidebar.",
        "subs_title": "Top subreddits (selected window)",
        "subs_limit_to_brands": "Limit to selected brands",
        "subs_top_n": "Top N",
        "subs_axis_sub": "Subreddit",
        "subs_axis_count": "Mentions",
        "subs_table": "Show subreddit table",
        "subs_stats": "Subreddit stats",
        "subs_stat_distinct": "Distinct subreddits",
        "subs_stat_top5": "Top 5 coverage (%)",
        "controls": "Controls",
        "frequency": "Frequency",
        "date_range": "Date range",
        "brands": "Brands",
        "metric": "Metric",
        "rolling_days": "Rolling window (days)",
        "rolling_weeks": "Rolling window (weeks)",
        "show_data_table": "Show data table",
        "language": "Language",
        "data_folder": "Data folder",
        "subs_brand_picker": "Brand for subreddit table",
        "subs_brand_table": "Top subreddits for selected brand",
        "subs_no_brand_data": "No subreddit data for this brand in the selected window.",
        "subs_examples_title": "Example posts for selected brand",
        "col_date": "Date",
        "col_title": "Title",
        "col_selftext": "Selftext",
        "col_comments": "Comments"
    },
    "ko": {
        "title": "레딧 브랜드 언급 — 멀티 섹터 대시보드",
        "sector": "섹터",
        "freq_daily": "일간",
        "freq_weekly": "주간",
        "metric_raw": "원시 수치",
        "metric_sov": "점유율 (%)",
        "no_data": "선택한 날짜 범위에 데이터가 없습니다.",
        "pick_brands": "사이드바에서 적어도 하나의 브랜드를 선택하세요.",
        "subs_title": "상위 서브레딧 (선택 구간)",
        "subs_limit_to_brands": "선택한 브랜드만 집계",
        "subs_top_n": "상위 N",
        "subs_axis_sub": "서브레딧",
        "subs_axis_count": "언급 수",
        "subs_table": "서브레딧 표 보기",
        "subs_stats": "서브레딧 통계",
        "subs_stat_distinct": "고유 서브레딧 수",
        "subs_stat_top5": "상위 5개 비중 (%)",
        "controls": "컨트롤",
        "frequency": "빈도",
        "date_range": "날짜 범위",
        "brands": "브랜드",
        "metric": "지표",
        "rolling_days": "이동 평균 (일)",
        "rolling_weeks": "이동 평균 (주)",
        "show_data_table": "데이터 표 보기",
        "language": "언어",
        "data_folder": "데이터 폴더",
        "subs_brand_picker": "서브레딧 표용 브랜드",
        "subs_brand_table": "선택한 브랜드의 상위 서브레딧",
        "subs_no_brand_data": "선택 구간에 이 브랜드의 서브레딧 데이터가 없습니다.",
        "subs_examples_title": "선택한 브랜드의 예시 게시글",
        "col_date": "날짜",
        "col_title": "제목",
        "col_selftext": "본문",
        "col_comments": "댓글"
    }
}

def T(lang: str, key: str) -> str:
    return STR.get(lang, STR["en"]).get(key, key)

# -----------------------------
# Config
# -----------------------------
# Map sectors to their base directories relative to THIS file
BASE_DIR = Path(__file__).parent.resolve()
SECTORS = {
    "beauty": BASE_DIR / "beauty_top_sharding",
    "fnb":    BASE_DIR / "fnb_top_sharding",
    "kpop":   BASE_DIR / "kpop_top_sharding",
}

DEFAULT_TOPN = 5

st.set_page_config(page_title=T("en", "title"), layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def sector_paths(sector: str) -> Tuple[Path, Path]:
    """Return pivot and raw paths for a given sector."""
    root = SECTORS.get(sector)
    if not root:
        raise ValueError(f"Unknown sector: {sector}")
    data_dir = (root / "data").resolve()
    pivot = data_dir / "reddit_brand_pivoted.csv"
    raw = data_dir / "reddit_brand_harvest_top_month.csv"
    return pivot, raw

@st.cache_data(show_spinner=False)
def load_pivot(pivot_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pivot_path)
    df.columns = [c.strip() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Input pivot must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_raw(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    needed = {"brand", "subreddit"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise ValueError(f"Raw file missing columns: {missing}")
    if "created_iso" in df.columns:
        dt = pd.to_datetime(df["created_iso"], errors="coerce")
    elif "created_utc" in df.columns:
        dt = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    else:
        raise ValueError("Raw file missing 'created_iso'/'created_utc'")
    df["date"] = dt.dt.normalize()
    df["subreddit"] = df["subreddit"].astype(str)
    df = df.dropna(subset=["date", "brand", "subreddit"])
    return df

@st.cache_data(show_spinner=False)
def resample_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return df.set_index("date").asfreq("D").fillna(0).reset_index()
    return df.set_index("date").resample("W").sum().reset_index()

def to_long(df: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    use_cols = ["date"] + brands
    return df[use_cols].melt(id_vars="date", var_name="brand", value_name="value")

@st.cache_data(show_spinner=False)
def compute_sov(df_period: pd.DataFrame) -> pd.DataFrame:
    brand_cols = [c for c in df_period.columns if c != "date"]
    totals = df_period[brand_cols].sum(axis=1).replace(0, np.nan)
    sov = df_period.copy()
    for c in brand_cols:
        sov[c] = (sov[c] / totals) * 100.0
    return sov.fillna(0)

def apply_rolling(long_df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        long_df["smoothed"] = long_df["value"]
        return long_df
    out = []
    for _, g in long_df.groupby("brand", as_index=False):
        g = g.sort_values("date").copy()
        g["smoothed"] = g["value"].rolling(int(window), min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    # Language toggle first (bilingual label so it's understandable before language is chosen)
    lang = st.radio("Language / 언어", ["en", "ko"], index=0, horizontal=True)

    st.header(T(lang, "controls"))

    # Sector toggle
    sector = st.selectbox(T(lang, "sector"), options=list(SECTORS.keys()), index=0)

    # Load sector data
    pivot_path, raw_path = sector_paths(sector)
    st.caption(f"{T(lang, 'data_folder')}: {pivot_path.parent}")

    # Frequency toggle
    freq_label = st.radio(T(lang, "frequency"), [T(lang, "freq_daily"), T(lang, "freq_weekly")],
                          index=0, horizontal=True)
    freq = "D" if freq_label == T(lang, "freq_daily") else "W"

    # Load pivot & resample
    df = load_pivot(pivot_path)
    all_brands = [c for c in df.columns if c != "date"]
    df_freq = resample_freq(df, freq)

    # Date window selector
    min_d, max_d = df_freq["date"].min().date(), df_freq["date"].max().date()
    date_range = st.date_input(T(lang, "date_range"), value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, date_range
    mask = (df_freq["date"].dt.date >= start_date) & (df_freq["date"].dt.date <= end_date)
    df_win = df_freq.loc[mask].copy()

    # Default top brands in window
    totals = df_win.drop(columns=["date"]).sum(axis=0).sort_values(ascending=False)
    defaults = totals.head(DEFAULT_TOPN).index.tolist() if not totals.empty else []

    selected = st.multiselect(T(lang, "brands"), options=all_brands, default=defaults)

    metric = st.radio(T(lang, "metric"), [T(lang, "metric_raw"), T(lang, "metric_sov")], index=0, horizontal=True)

    roll_label = T(lang, "rolling_days") if freq == "D" else T(lang, "rolling_weeks")
    roll_win = st.number_input(roll_label, min_value=1, max_value=60, value=7)

    st.markdown("---")
    st.subheader(T(lang, "subs_title"))
    limit_to_brands = st.checkbox(T(lang, "subs_limit_to_brands"), value=True)
    top_n = st.slider(T(lang, "subs_top_n"), min_value=5, max_value=30, value=15, step=1)

st.title(f"{sector.upper()} — {T(lang, 'title')}")

# Guards
if df_win.empty:
    st.info(T(lang, "no_data"))
    st.stop()
if not selected:
    st.info(T(lang, "pick_brands"))
    st.stop()

# -----------------------------
# Metric prep
# -----------------------------
if metric == T(lang, "metric_raw"):
    df_metric = df_win.copy()
    y_label = "Mentions"
else:
    df_metric = compute_sov(df_win)
    y_label = "Share of Voice (%)"

long_sel = to_long(df_metric, selected)
long_smooth = apply_rolling(long_sel, int(roll_win))

# -----------------------------
# Plot
# -----------------------------
fig = px.line(
    long_smooth, x="date", y="smoothed", color="brand",
    title=f"{sector.upper()} — Brand trends · {freq_label} · {metric}",
    labels={"smoothed": y_label, "date": "Date"}
)
fig.update_layout(hovermode="x unified", legend_title_text="Brand")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Top subreddits section
# -----------------------------
try:
    raw = load_raw(raw_path)
    win_mask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
    raw_win = raw.loc[win_mask].copy()
    if limit_to_brands and selected:
        raw_win = raw_win[raw_win["brand"].isin(selected)]

    sub_counts = (raw_win.groupby("subreddit").size().reset_index(name="count")
                  .sort_values("count", ascending=False))
    avail_brands = sorted(raw_win["brand"].dropna().unique().tolist())

    if not sub_counts.empty:
        sub_top = sub_counts.head(int(top_n))
        fig_sub = px.bar(sub_top[::-1], x="count", y="subreddit", orientation="h",
                         labels={"count": T(lang, "subs_axis_count"), "subreddit": T(lang, "subs_axis_sub")},
                         title=T(lang, "subs_title"))
        st.plotly_chart(fig_sub, use_container_width=True)

        distinct_subs = int(sub_counts["subreddit"].nunique())
        top5_share = float(sub_counts.head(5)["count"].sum()) / float(sub_counts["count"].sum()) * 100.0 if len(sub_counts) else 0.0
        col1, col2 = st.columns(2)
        with col1:
            st.metric(T(lang, "subs_stat_distinct"), f"{distinct_subs}")
        with col2:
            st.metric(T(lang, "subs_stat_top5"), f"{top5_share:.1f}%")

        st.markdown("\n")
        st.subheader(T(lang, "subs_brand_table"))
        if avail_brands:
            default_brand = selected[0] if selected and selected[0] in avail_brands else avail_brands[0]
            pick_brand = st.selectbox(T(lang, "subs_brand_picker"), options=avail_brands, index=avail_brands.index(default_brand))
            brand_counts = (raw_win[raw_win["brand"] == pick_brand]
                            .groupby("subreddit").size().reset_index(name="count")
                            .sort_values("count", ascending=False))
            if brand_counts.empty:
                st.info(T(lang, "subs_no_brand_data"))
            else:
                st.dataframe(brand_counts.head(10), use_container_width=True)

            st.markdown("\n")
            st.subheader(T(lang, "subs_examples_title"))
            samples = (raw_win[raw_win["brand"] == pick_brand]
                       .loc[:, ["date", "subreddit", "title", "selftext", "comments_text"]]
                       .copy())
            if not samples.empty:
                samples["date"] = pd.to_datetime(samples["date"]).dt.date.astype(str)
                for col in ["title", "selftext", "comments_text"]:
                    samples[col] = samples[col].astype(str).str.replace("\n", " ").str.slice(0, 280)
                samples = samples.sort_values("date", ascending=False).head(10)
                col_map = {
                    "date": T(lang, "col_date"),
                    "subreddit": T(lang, "subs_axis_sub"),
                    "title": T(lang, "col_title"),
                    "selftext": T(lang, "col_selftext"),
                    "comments_text": T(lang, "col_comments"),
                }
                st.dataframe(samples.rename(columns=col_map), use_container_width=True)
            else:
                st.info(T(lang, "subs_no_brand_data"))
        else:
            st.info(T(lang, "subs_no_brand_data"))

        with st.expander(T(lang, "subs_table")):
            st.dataframe(sub_counts, use_container_width=True)
    else:
        st.info(T(lang, "no_data"))
except Exception as e:
    st.warning(f"Subreddit section unavailable: {e}")

with st.expander(T(lang, "show_data_table")):
    st.dataframe(df_metric[["date"] + selected], use_container_width=True)