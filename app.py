# ============================================================
# Powerball Structural Emulator + Multi-Model Analyzer
# ============================================================
# - Loads historical Powerball draws (NY API)
# - Traditional models: hot/cold, mixed, pattern-balanced, pure random
# - New model: "GSE Emulator" (Generative Structural Emulator style)
# - Advanced analysis + comparison in multiple tabs
#
# DISCLAIMER:
# - This app DOES NOT predict future draws.
# - Lottery draws are random.
# - This is for education, analysis, and simulation only.
# ============================================================

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import random
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# SECTION 2 — APP CONFIG
# ============================================================

st.set_page_config(page_title="Powerball Structural Emulator", layout="wide")
st.title("🎯 Powerball Structural Emulator + Multi-Model Analyzer")
st.caption("Educational analytics & simulation. Not a predictor.")

API_URL = "https://data.ny.gov/resource/d6yy-54nr.json?$limit=5000"
LOW_CUTOFF = 34
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

WHITE_RANGE = np.arange(1, 70)
PB_RANGE = np.arange(1, 27)

# ============================================================
# SECTION 3 — SAFE CHART HELPERS
# ============================================================

def safe_bar(data):
    """Bar chart that never crashes on IntervalIndex from pd.cut()."""
    obj = data.copy()

    # Convert IntervalIndex -> string (pd.cut produces intervals)
    try:
        obj.index = obj.index.astype(str)
    except Exception:
        pass

    # Ensure numeric
    if isinstance(obj, pd.Series):
        obj = pd.to_numeric(obj, errors="coerce").fillna(0)
    else:
        obj = obj.fillna(0)
        for c in obj.columns:
            obj[c] = pd.to_numeric(obj[c], errors="coerce").fillna(0)

    st.bar_chart(obj)


def binned_counts(series: pd.Series, bins):
    """Return counts with stable bins (bins may be int or explicit edges)."""
    return pd.cut(series, bins=bins).value_counts().sort_index()

# ============================================================
# SECTION 4 — LOAD DATA
# ============================================================

@st.cache_data
def load_powerball_data():
    resp = requests.get(API_URL, timeout=15)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        raise RuntimeError("API returned empty response")

    df = pd.DataFrame(raw)
    if "draw_date" not in df.columns or "winning_numbers" not in df.columns:
        raise RuntimeError("API schema changed (missing draw_date or winning_numbers)")

    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df = df.dropna(subset=["draw_date"])

    nums = df["winning_numbers"].str.split(" ", expand=True)
    nums = nums.iloc[:, :6].astype(int)
    nums.columns = ["w1", "w2", "w3", "w4", "w5", "pb"]

    df = pd.concat([df[["draw_date"]], nums], axis=1)
    df = df.sort_values("draw_date").reset_index(drop=True)

    # Validate: 5 unique whites
    df = df[df[["w1", "w2", "w3", "w4", "w5"]].nunique(axis=1) == 5].copy()

    # Validate ranges
    for c in ["w1", "w2", "w3", "w4", "w5"]:
        df = df[(df[c] >= 1) & (df[c] <= 69)]
    df = df[(df["pb"] >= 1) & (df["pb"] <= 26)]

    return df.reset_index(drop=True)

try:
    df_all = load_powerball_data()
    st.success(f"✅ Loaded {len(df_all)} cleaned historical draws")
except Exception as e:
    st.error(f"Failed to load Powerball data: {e}")
    st.stop()

# ============================================================
# SECTION 5 — SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("⚙️ Controls")

lookback = st.sidebar.slider(
    "Lookback draws (use last N)",
    min_value=50,
    max_value=len(df_all),
    value=min(500, len(df_all)),
    step=50
)

hot_k = st.sidebar.slider("Hot pool size (top K)", 5, 30, 15, 1)
cold_k = st.sidebar.slider("Cold pool size (bottom K)", 5, 30, 15, 1)

pattern_tolerance_sum = st.sidebar.slider("Pattern-Balanced: Sum tolerance (±)", 0, 40, 10, 1)
pattern_tolerance_gap = st.sidebar.slider("Pattern-Balanced: Max-gap tolerance (±)", 0, 20, 5, 1)

# Tolerances for Emulator (can be a bit tighter/looser)
emulator_tolerance_sum = st.sidebar.slider("GSE Emulator: Sum tolerance (±)", 0, 50, 15, 1)
emulator_tolerance_gap = st.sidebar.slider("GSE Emulator: Max-gap tolerance (±)", 0, 25, 8, 1)

num_tickets = st.sidebar.slider("Tickets to generate", 1, 50, 10, 1)

selected_model = st.sidebar.selectbox(
    "Generator model (for Generator tab)",
    [
        "Pure Random (baseline)",
        "Hot-heavy",
        "Cold-heavy",
        "Mixed (hot+cold)",
        "Pattern-Balanced ✅",
        "GSE Emulator 🔬"
    ]
)

df = df_all.tail(lookback).reset_index(drop=True)

# ============================================================
# SECTION 6 — FREQUENCY + PATTERN FEATURES
# ============================================================

def white_frequency(df_):
    s = pd.concat([df_.w1, df_.w2, df_.w3, df_.w4, df_.w5])
    return s.value_counts().reindex(WHITE_RANGE, fill_value=0)

def pb_frequency(df_):
    return df_["pb"].value_counts().reindex(PB_RANGE, fill_value=0)

def extract_patterns_from_df(df_):
    rows = []
    for _, r in df_.iterrows():
        whites = np.array([r.w1, r.w2, r.w3, r.w4, r.w5])
        whites.sort()
        gaps = np.diff(whites)
        rows.append({
            "odd_count": int(np.sum(whites % 2)),
            "low_count": int(np.sum(whites <= LOW_CUTOFF)),
            "sum": int(whites.sum()),
            "range": int(whites.max() - whites.min()),
            "max_gap": int(np.max(gaps)),
            "gap1": int(gaps[0]),
            "gap2": int(gaps[1]),
            "gap3": int(gaps[2]),
            "gap4": int(gaps[3])
        })
    return pd.DataFrame(rows)

freq_white = white_frequency(df)
freq_pb = pb_frequency(df)

hot_whites = freq_white.sort_values(ascending=False).head(hot_k).index.tolist()
cold_whites = freq_white.sort_values(ascending=True).head(cold_k).index.tolist()

patterns_real = extract_patterns_from_df(df)

odd_dist = patterns_real["odd_count"].value_counts(normalize=True).sort_index()
low_dist = patterns_real["low_count"].value_counts(normalize=True).sort_index()

def sample_from_empirical(dist_series: pd.Series):
    vals = dist_series.index.to_numpy()
    probs = dist_series.values
    return int(np.random.choice(vals, p=probs))

# ============================================================
# SECTION 7 — WEIGHTED FREQUENCY (FOR EMULATOR)
# ============================================================

def compute_weighted_white_probs(df_full: pd.DataFrame, lookback_window: int = 500, alpha_recent: float = 1.5):
    """
    Build a soft probability distribution over white numbers 1-69:
    - Combines long-term frequency and recent frequency with a weighting factor.
    - Ensures all probabilities are > 0 (no hard exclusions).
    """
    # Long-term frequency
    long_freq = white_frequency(df_full)

    # Recent frequency (on tail)
    df_recent = df_full.tail(lookback_window)
    recent_freq = white_frequency(df_recent)

    # Normalize to probabilities
    long_prob = (long_freq + 1) / (long_freq.sum() + len(long_freq))
    recent_prob = (recent_freq + 1) / (recent_freq.sum() + len(recent_freq))

    # Combine (recent emphasized by alpha_recent)
    combined = long_prob * (recent_prob ** alpha_recent)

    # Normalize again
    combined = combined / combined.sum()

    return combined

weighted_probs_white = compute_weighted_white_probs(df_all, lookback_window=lookback)

# ============================================================
# SECTION 8 — BASE GENERATORS (EXISTING MODELS)
# ============================================================

def generate_whites_pure_random():
    return sorted(random.sample(range(1, 70), 5))

def pick_pb_uniform():
    return random.randint(1, 26)

def generate_whites_hot_heavy():
    if len(hot_whites) >= 5:
        return sorted(random.sample(hot_whites, 5))
    return generate_whites_pure_random()

def generate_whites_cold_heavy():
    if len(cold_whites) >= 5:
        return sorted(random.sample(cold_whites, 5))
    return generate_whites_pure_random()

def generate_whites_mixed():
    # 3 hot + 2 cold
    if len(hot_whites) >= 3 and len(cold_whites) >= 2:
        return sorted(random.sample(hot_whites, 3) + random.sample(cold_whites, 2))
    return generate_whites_pure_random()

def generate_whites_pattern_balanced(max_tries=5000):
    """
    Pattern-Balanced:
    - Sample target odd_count and low_count from real distributions
    - Sample target sum and max_gap from a real pattern row
    - Search for a random white set that matches within tolerances
    """
    target_odd = sample_from_empirical(odd_dist)
    target_low = sample_from_empirical(low_dist)

    target_row = patterns_real.sample(1).iloc[0]
    target_sum = int(target_row["sum"])
    target_gap = int(target_row["max_gap"])

    for _ in range(max_tries):
        w = generate_whites_pure_random()

        odd_count = sum(n % 2 for n in w)
        low_count = sum(n <= LOW_CUTOFF for n in w)
        s = sum(w)
        gap = int(np.max(np.diff(w)))

        if odd_count != target_odd:
            continue
        if low_count != target_low:
            continue
        if abs(s - target_sum) > pattern_tolerance_sum:
            continue
        if abs(gap - target_gap) > pattern_tolerance_gap:
            continue

        return w

    return generate_whites_pure_random()

# ============================================================
# SECTION 9 — GSE EMULATOR GENERATOR (NEW MODEL)
# ============================================================

def sample_pattern_profile():
    """
    Sample a structural pattern profile from historical patterns.
    This approximates a 'mixture model' by just sampling a real row.
    """
    row = patterns_real.sample(1).iloc[0]
    profile = {
        "odd_target": int(row["odd_count"]),
        "low_target": int(row["low_count"]),
        "sum_target": int(row["sum"]),
        "range_target": int(row["range"]),
        "max_gap_target": int(row["max_gap"]),
        "gap1_target": int(row["gap1"]),
        "gap2_target": int(row["gap2"]),
        "gap3_target": int(row["gap3"]),
        "gap4_target": int(row["gap4"]),
    }
    return profile

def generate_whites_emulator(max_tries=8000):
    """
    GSE Emulator:
    - Sample a target pattern profile from real data
    - Generate candidate sets using weighted probabilities over numbers
    - Accept a set if it matches structural constraints within tolerances
    """
    profile = sample_pattern_profile()
    odd_target = profile["odd_target"]
    low_target = profile["low_target"]
    sum_target = profile["sum_target"]
    range_target = profile["range_target"]
    max_gap_target = profile["max_gap_target"]

    # Precompute numeric arrays for choice
    numbers = WHITE_RANGE
    probs = weighted_probs_white.values

    for _ in range(max_tries):
        # Sample 5 distinct numbers (without replacement) using weighted probs
        # We approximate without replacement by simple loop
        chosen = set()
        while len(chosen) < 5:
            n = int(np.random.choice(numbers, p=probs))
            chosen.add(n)
        w = sorted(chosen)

        whites = np.array(w)
        gaps = np.diff(whites)

        odd_count = int(np.sum(whites % 2))
        low_count = int(np.sum(whites <= LOW_CUTOFF))
        s = int(whites.sum())
        r = int(whites.max() - whites.min())
        max_gap = int(np.max(gaps))

        # Basic structural constraints
        if odd_count != odd_target:
            continue
        if low_count != low_target:
            continue
        if abs(s - sum_target) > emulator_tolerance_sum:
            continue
        if abs(max_gap - max_gap_target) > emulator_tolerance_gap:
            continue
        # Range consistency (so shape is similar)
        if abs(r - range_target) > 10:  # small buffer
            continue

        # If it passes all checks, accept
        return w

    # If no match after many tries, fallback to pattern-balanced or random
    return generate_whites_pattern_balanced()

# ============================================================
# SECTION 10 — MODEL DISPATCH
# ============================================================

def generate_ticket(model_name: str):
    if model_name == "Pure Random (baseline)":
        whites = generate_whites_pure_random()
    elif model_name == "Hot-heavy":
        whites = generate_whites_hot_heavy()
    elif model_name == "Cold-heavy":
        whites = generate_whites_cold_heavy()
    elif model_name == "Mixed (hot+cold)":
        whites = generate_whites_mixed()
    elif model_name == "Pattern-Balanced ✅":
        whites = generate_whites_pattern_balanced()
    elif model_name == "GSE Emulator 🔬":
        whites = generate_whites_emulator()
    else:
        whites = generate_whites_pure_random()

    pb = pick_pb_uniform()
    return whites + [pb]

# ============================================================
# SECTION 11 — UI TABS
# ============================================================

tab_analysis, tab_generator, tab_compare, tab_model_lab = st.tabs(
    ["📊 Analysis", "🎟 Generator", "🧪 Compare Models", "🧬 Model Lab"]
)

# ---------------- TAB: ANALYSIS ----------------
with tab_analysis:
    st.subheader("Hot vs Cold Numbers (Lookback window)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔥 Hot Whites")
        st.dataframe(
            pd.DataFrame(
                {
                    "number": hot_whites,
                    "count": [int(freq_white[n]) for n in hot_whites],
                }
            )
        )
        st.markdown("### ❄️ Cold Whites")
        st.dataframe(
            pd.DataFrame(
                {
                    "number": cold_whites,
                    "count": [int(freq_white[n]) for n in cold_whites],
                }
            )
        )

    with c2:
        st.markdown("### Powerball Frequency (all PB values)")
        st.bar_chart(freq_pb)

    st.markdown("---")
    st.subheader("Pattern Distributions (Real Draws)")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Odd Count**")
        st.bar_chart(patterns_real["odd_count"].value_counts().sort_index())
        st.markdown("**Low Count (≤34)**")
        st.bar_chart(patterns_real["low_count"].value_counts().sort_index())

    with cc2:
        st.markdown("**Sum (binned)**")
        safe_bar(binned_counts(patterns_real["sum"], bins=20))
        st.markdown("**Range (binned)**")
        safe_bar(binned_counts(patterns_real["range"], bins=15))

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Max Gap (binned)**")
        safe_bar(binned_counts(patterns_real["max_gap"], bins=15))
    with c4:
        st.markdown("**Gap Vector Components (g1–g4)**")
        gap_cols = ["gap1", "gap2", "gap3", "gap4"]
        st.bar_chart(patterns_real[gap_cols].mean())

# ---------------- TAB: GENERATOR ----------------
with tab_generator:
    st.subheader("Generate Tickets")
    st.write(
        f"**Model:** {selected_model}  |  **Lookback:** last {lookback} draws"
    )

    if st.button("🎟 Generate Now"):
        tickets = [generate_ticket(selected_model) for _ in range(num_tickets)]
        out = pd.DataFrame(tickets, columns=["W1", "W2", "W3", "W4", "W5", "PB"])
        st.success(f"Generated {len(out)} tickets.")
        st.dataframe(out)

        st.download_button(
            "⬇️ Download CSV",
            data=out.to_csv(index=False),
            file_name="powerball_generated_tickets.csv",
            mime="text/csv",
        )

# ---------------- TAB: COMPARE MODELS ----------------
with tab_compare:
    st.subheader("Compare Different Models vs Real Patterns")
    st.caption("Comparison is structural (odd/low/sum/range/gap), not prediction.")

    models_to_compare = [
        "Pure Random (baseline)",
        "Hot-heavy",
        "Cold-heavy",
        "Mixed (hot+cold)",
        "Pattern-Balanced ✅",
        "GSE Emulator 🔬",
    ]

    sample_n = st.slider(
        "Comparison sample size per model", 100, 2000, 500, 100
    )

    # Compute stable bin edges for sum/gap so all models align
    sum_min, sum_max = patterns_real["sum"].min(), patterns_real["sum"].max()
    gap_min, gap_max = patterns_real["max_gap"].min(), patterns_real["max_gap"].max()
    range_min, range_max = patterns_real["range"].min(), patterns_real["range"].max()

    sum_bins_edges = np.linspace(sum_min, sum_max, 21)   # 20 bins
    gap_bins_edges = np.linspace(gap_min, gap_max, 16)   # 15 bins
    range_bins_edges = np.linspace(range_min, range_max, 16)

    # Real reference distributions (aligned bins)
    real_sum_binned = binned_counts(patterns_real["sum"], bins=sum_bins_edges)
    real_gap_binned = binned_counts(patterns_real["max_gap"], bins=gap_bins_edges)
    real_range_binned = binned_counts(patterns_real["range"], bins=range_bins_edges)

    # Compare means table
    compare_rows = []

    st.markdown("### Mean Pattern Metrics (Real vs Models)")
    for m in models_to_compare:
        gen_sample = [generate_ticket(m) for _ in range(sample_n)]
        gen_df = pd.DataFrame(gen_sample, columns=["w1", "w2", "w3", "w4", "w5", "pb"])
        patt = extract_patterns_from_df(gen_df)

        compare_rows.append(
            {
                "model": m,
                "odd_mean": float(patt["odd_count"].mean()),
                "low_mean": float(patt["low_count"].mean()),
                "sum_mean": float(patt["sum"].mean()),
                "range_mean": float(patt["range"].mean()),
                "gap_mean": float(patt["max_gap"].mean()),
            }
        )

    real_means = {
        "model": "REAL (lookback)",
        "odd_mean": float(patterns_real["odd_count"].mean()),
        "low_mean": float(patterns_real["low_count"].mean()),
        "sum_mean": float(patterns_real["sum"].mean()),
        "range_mean": float(patterns_real["range"].mean()),
        "gap_mean": float(patterns_real["max_gap"].mean()),
    }

    means_df = pd.DataFrame([real_means] + compare_rows)
    st.dataframe(means_df)

    st.markdown("---")
    st.markdown("### Distribution Compare (Real vs Selected Model)")

    chosen = st.selectbox(
        "Choose model to compare distributions", models_to_compare, index=0
    )

    gen_sample = [generate_ticket(chosen) for _ in range(sample_n)]
    gen_df = pd.DataFrame(gen_sample, columns=["w1", "w2", "w3", "w4", "w5", "pb"])
    patt = extract_patterns_from_df(gen_df)

    # Aligned binned distributions
    gen_sum_binned = binned_counts(patt["sum"], bins=sum_bins_edges)
    gen_gap_binned = binned_counts(patt["max_gap"], bins=gap_bins_edges)
    gen_range_binned = binned_counts(patt["range"], bins=range_bins_edges)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sum bins: Real vs Model**")
        comp_sum = pd.DataFrame({"real": real_sum_binned, "model": gen_sum_binned}).fillna(0)
        safe_bar(comp_sum)

    with c2:
        st.markdown("**Max gap bins: Real vs Model**")
        comp_gap = pd.DataFrame({"real": real_gap_binned, "model": gen_gap_binned}).fillna(0)
        safe_bar(comp_gap)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Range bins: Real vs Model**")
        comp_range = pd.DataFrame(
            {"real": real_range_binned, "model": gen_range_binned}
        ).fillna(0)
        safe_bar(comp_range)

    with c4:
        st.markdown("**Odd/Low Count: Real vs Model**")
        oc_real = patterns_real["odd_count"].value_counts().sort_index()
        oc_gen = patt["odd_count"].value_counts().sort_index()
        lc_real = patterns_real["low_count"].value_counts().sort_index()
        lc_gen = patt["low_count"].value_counts().sort_index()

        comp_oc = pd.DataFrame({"real": oc_real, "model": oc_gen}).fillna(0)
        comp_lc = pd.DataFrame({"real": lc_real, "model": lc_gen}).fillna(0)

        st.markdown("Odd Count")
        st.bar_chart(comp_oc)
        st.markdown("Low Count (≤34)")
        st.bar_chart(comp_lc)

# ---------------- TAB: MODEL LAB (EMULATOR FOCUS) ----------------
with tab_model_lab:
    st.subheader("Model Lab: GSE Emulator vs Real Draws")
    st.caption("Inspect how the Emulator's structural behavior compares with history.")

    lab_sample_n = st.slider(
        "Emulator sample size (Model Lab)", 200, 5000, 1000, 200
    )

    # Generate emulator sample
    emu_sample = [generate_ticket("GSE Emulator 🔬") for _ in range(lab_sample_n)]
    emu_df = pd.DataFrame(emu_sample, columns=["w1", "w2", "w3", "w4", "w5", "pb"])
    emu_patterns = extract_patterns_from_df(emu_df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sum Distribution: Real vs Emulator**")
        emu_sum_binned = binned_counts(emu_patterns["sum"], bins=sum_bins_edges)
        comp_sum = pd.DataFrame(
            {"real": real_sum_binned, "emulator": emu_sum_binned}
        ).fillna(0)
        safe_bar(comp_sum)

        st.markdown("**Range Distribution: Real vs Emulator**")
        emu_range_binned = binned_counts(emu_patterns["range"], bins=range_bins_edges)
        comp_range = pd.DataFrame(
            {"real": real_range_binned, "emulator": emu_range_binned}
        ).fillna(0)
        safe_bar(comp_range)

    with c2:
        st.markdown("**Max Gap Distribution: Real vs Emulator**")
        emu_gap_binned = binned_counts(emu_patterns["max_gap"], bins=gap_bins_edges)
        comp_gap = pd.DataFrame(
            {"real": real_gap_binned, "emulator": emu_gap_binned}
        ).fillna(0)
        safe_bar(comp_gap)

        st.markdown("**Odd/Low Count: Real vs Emulator**")
        oc_real = patterns_real["odd_count"].value_counts().sort_index()
        oc_emu = emu_patterns["odd_count"].value_counts().sort_index()
        lc_real = patterns_real["low_count"].value_counts().sort_index()
        lc_emu = emu_patterns["low_count"].value_counts().sort_index()

        comp_oc = pd.DataFrame({"real": oc_real, "emulator": oc_emu}).fillna(0)
        comp_lc = pd.DataFrame({"real": lc_real, "emulator": lc_emu}).fillna(0)

        st.markdown("Odd Count")
        st.bar_chart(comp_oc)
        st.markdown("Low Count (≤34)")
        st.bar_chart(comp_lc)

    st.markdown("---")
    st.markdown("### Emulator Mean vs Real Mean")

    emu_means = {
        "model": "GSE Emulator 🔬",
        "odd_mean": float(emu_patterns["odd_count"].mean()),
        "low_mean": float(emu_patterns["low_count"].mean()),
        "sum_mean": float(emu_patterns["sum"].mean()),
        "range_mean": float(emu_patterns["range"].mean()),
        "gap_mean": float(emu_patterns["max_gap"].mean()),
    }

    real_means_lab = {
        "model": "REAL (lookback)",
        "odd_mean": float(patterns_real["odd_count"].mean()),
        "low_mean": float(patterns_real["low_count"].mean()),
        "sum_mean": float(patterns_real["sum"].mean()),
        "range_mean": float(patterns_real["range"].mean()),
        "gap_mean": float(patterns_real["max_gap"].mean()),
    }

    lab_means_df = pd.DataFrame([real_means_lab, emu_means])
    st.dataframe(lab_means_df)

st.markdown("---")
st.caption(
    "Disclaimer: Lottery draws are random. Hot/cold, pattern metrics, and emulator models describe history only and do not predict future results."
)
