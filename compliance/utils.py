import numpy as np

# ---------- global knobs -------------------------------------------------
TOP_K = 20  # shortlist size returned by HNSW
GEO_SIGMA = 50  # km for geo decay
TOL_PCT = 0.10  # ±10 % perfect for numeric fields
DECAY_PCT = 0.50  # ±50 % → zero
DISCOUNT_GOOD = 0.20  # ≤20 % cheaper counts perfect
DISCOUNT_MAX = 0.50  # >50 % cheaper → zero
MARKUP_TOL = 0.10  # ≤10 % dearer still perfect
MARKUP_MAX = 0.50  # >50 % dearer → zero


# ---------- geo helpers --------------------------------------------------
def haversine_np(lat1, lon1, lats, lons):
    R = 6371.0
    lat1, lon1 = map(float, (lat1, lon1))
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def geo_similarity_np(lat1, lon1, lats, lons, σ=GEO_SIGMA):
    return np.exp(-haversine_np(lat1, lon1, lats, lons) / σ)


# ---------- numeric similarity ------------------------------------------
def numeric_similarity_np(values, seed_val):
    values = np.asarray(values, dtype=float)
    seed = float(seed_val)
    diff_pct = np.abs(values - seed) / (seed + 1e-9)
    inside = diff_pct <= TOL_PCT
    decay = (DECAY_PCT - diff_pct) / (DECAY_PCT - TOL_PCT)
    return np.where(inside, 1.0, np.clip(decay, 0.0, 1.0))


def price_similarity_np(values, seed_price):
    v = np.asarray(values, dtype=float)
    sp = float(seed_price)

    cheaper_frac = (sp - v) / sp
    cheaper_good = cheaper_frac <= DISCOUNT_GOOD
    cheaper_bad = cheaper_frac >= DISCOUNT_MAX
    s_cheaper = np.where(
        cheaper_good, 1.0,
        np.where(
            cheaper_bad, 0.0,
            1.0 - (cheaper_frac - DISCOUNT_GOOD) / (DISCOUNT_MAX - DISCOUNT_GOOD)
        )
    )

    dearer_frac = (v - sp) / sp
    dearer_good = dearer_frac <= MARKUP_TOL
    dearer_bad = dearer_frac >= MARKUP_MAX
    s_dearer = np.where(
        dearer_good, 1.0,
        np.where(
            dearer_bad, 0.0,
            1.0 - (dearer_frac - MARKUP_TOL) / (MARKUP_MAX - MARKUP_TOL)
        )
    )

    return np.where(v <= sp, s_cheaper, s_dearer)


# ---------- BUY range helpers -------------------------------------------
def interval_similarity_np(values, lo, hi):
    """
    1.0 inside [lo, hi].
    Outside: decay to 0 with a margin proportional to DECAY_PCT relative to mid-value (percent-based).
    Fallback to absolute-margin if mid <= 0.
    """
    v = np.asarray(values, dtype=float)
    lo = float(lo);
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    width = max(hi - lo, 1e-6)
    mid = (lo + hi) / 2.0

    if mid > 0:
        diff_pct_below = np.clip((lo - v) / (mid + 1e-9), 0, None)
        diff_pct_above = np.clip((v - hi) / (mid + 1e-9), 0, None)
        outside_pct = diff_pct_below + diff_pct_above
        inside = (v >= lo) & (v <= hi)
        decay = (DECAY_PCT - outside_pct) / (DECAY_PCT + 1e-9)
        return np.where(inside, 1.0, np.clip(decay, 0.0, 1.0))

    # degenerate fallback
    margin = max(0.5 * width * DECAY_PCT, 1.0)
    inside = (v >= lo) & (v <= hi)
    dist = np.where(v < lo, lo - v, np.where(v > hi, v - hi, 0.0))
    return np.where(inside, 1.0, np.clip(1.0 - dist / (margin + 1e-9), 0.0, 1.0))


def price_against_range_similarity_np(values, lo, hi):
    """
    For BUY seed: candidate SELL price vs desired [lo, hi].
    - v <= hi: 1.0 (buyers don't mind cheaper; if you want to penalize < lo, switch to interval_similarity_np)
    - v >  hi: decay with 'dearer' curve
    """
    v = np.asarray(values, dtype=float)
    lo = float(lo);
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo

    s_in = np.where(v <= hi, 1.0, 0.0)  # accept cheaper or in-range as perfect

    dearer_frac = np.clip((v - hi) / (hi + 1e-9), 0, None)
    dearer_good = dearer_frac <= MARKUP_TOL
    dearer_bad = dearer_frac >= MARKUP_MAX
    s_dearer = np.where(
        dearer_good, 1.0,
        np.where(dearer_bad, 0.0,
                 1.0 - (dearer_frac - MARKUP_TOL) / (MARKUP_MAX - MARKUP_TOL))
    )
    return np.where(v <= hi, s_in, s_dearer)


# ---------- weights ------------------------------------------------------
BASE_WEIGHTS = {"year": 0.20, "mileage": 0.12, "insurance": 0.08,
                "price": 0.12, "geo": 0.18, "color": 0.10}
VEC_WT = 0.20
WEIGHTS = {**BASE_WEIGHTS, "vec": VEC_WT}
WEIGHTS = {k: v / sum(WEIGHTS.values()) for k, v in WEIGHTS.items()}


# ---------- color id match (IDs, not names/hues) -------------------------
def colour_similarity_id(seed_color_id, cand_color_ids):
    t = str(seed_color_id).strip().lower()
    arr = np.array([str(c).strip().lower() for c in cand_color_ids])
    return np.where(arr == t, 1.0, 0.0)


# ---------- helpers for masking missing fields ---------------------------
def _as_float_array(metas, key):
    """Return array of floats with np.nan where value is missing/blank."""
    arr = []
    for x in metas:
        v = x.get(key)
        try:
            if v is None:
                arr.append(np.nan);
                continue
            s = str(v).strip()
            if s == "":
                arr.append(np.nan);
                continue
            arr.append(float(s))
        except Exception:
            arr.append(np.nan)
    return np.asarray(arr, dtype=float)


def _weighted_average(scores_dict, weights_dict):
    """
    scores_dict: {name: np.array (N,)} with possible np.nan when missing
    weights_dict: {name: scalar}
    Returns: np.array (N,) — weighted average per candidate over available factors only.
    """
    total_w = None
    num = None
    for name, s in scores_dict.items():
        base_w = float(weights_dict.get(name, 0.0))
        if base_w <= 0:
            continue
        w = np.where(np.isnan(s), 0.0, base_w)  # zero weight where the score is missing
        s_safe = np.nan_to_num(s, nan=0.0)

        num = s_safe * w if num is None else num + s_safe * w
        total_w = w if total_w is None else total_w + w

    if total_w is None:  # no factors at all
        return np.zeros(0, dtype=float)
    return np.divide(num, total_w, out=np.zeros_like(num), where=total_w > 0)


# ---------- scorers ------------------------------------------------------
def _cos_sim_01(seed_vec, cand_vecs):
    sv = np.asarray(seed_vec, dtype=float)
    cv = np.asarray(cand_vecs, dtype=float)
    dot = (cv @ sv) / (np.linalg.norm(cv, axis=1) * np.linalg.norm(sv) + 1e-9)
    return 0.5 * (dot + 1.0)  # [-1,1] → [0,1]


def compliance_scores(seed_meta, cand_metas, seed_vec, cand_vecs):
    """
    SELL seed → SELL candidates.
    Missing fields (either side) are excluded from the average (no penalty).
    """
    m = cand_metas

    # candidate arrays (may contain nan)
    year = _as_float_array(m, "year")
    mileage = _as_float_array(m, "mileage")
    ins_mo = _as_float_array(m, "insurance_mo")
    price = _as_float_array(m, "price")
    lat = _as_float_array(m, "lat")
    lon = _as_float_array(m, "lon")
    col_ids = [x.get("color_id", "") for x in m]

    # seed fields (may be missing)
    sy = seed_meta.get("year")
    skm = seed_meta.get("mileage")
    sins = seed_meta.get("insurance_mo")
    sp = seed_meta.get("price")
    slat = seed_meta.get("lat")
    slon = seed_meta.get("lon")
    scol = seed_meta.get("color_id", "")

    # helpers to compute score or nan
    def _num_sim_or_nan(vals, seed_val, fn):
        if seed_val is None or str(seed_val) == "":
            return np.full_like(vals, np.nan, dtype=float)
        mask = ~np.isnan(vals)
        out = np.full_like(vals, np.nan, dtype=float)
        out[mask] = fn(vals[mask], seed_val)
        return out

    s_year = _num_sim_or_nan(year, sy, numeric_similarity_np)
    s_mile = _num_sim_or_nan(mileage, skm, numeric_similarity_np)
    s_ins = _num_sim_or_nan(ins_mo, sins, numeric_similarity_np)
    s_price = _num_sim_or_nan(price, sp, price_similarity_np)

    # geo requires both sides
    if slat is None or slon is None or str(slat) == "" or str(slon) == "":
        s_geo = np.full_like(lat, np.nan, dtype=float)
    else:
        mask = (~np.isnan(lat)) & (~np.isnan(lon))
        s_geo = np.full_like(lat, np.nan, dtype=float)
        s_geo[mask] = geo_similarity_np(float(slat), float(slon), lat[mask], lon[mask])

    # color: if seed color or candidate color missing -> nan (exclude)
    if not str(scol).strip():
        s_col = np.full(len(m), np.nan, dtype=float)
    else:
        s_col = colour_similarity_id(scol, col_ids).astype(float)
        has_cand_col = np.array([bool(str(c).strip()) for c in col_ids])
        s_col = np.where(has_cand_col, s_col, np.nan)

    # vector similarity is always present
    s_vec = _cos_sim_01(seed_vec, cand_vecs)

    scores = _weighted_average(
        {
            "year": s_year,
            "mileage": s_mile,
            "insurance": s_ins,
            "price": s_price,
            "geo": s_geo,
            "color": s_col,
            "vec": s_vec,
        },
        WEIGHTS,
    )
    return scores


def compliance_scores_mixed(seed_meta, cand_metas, seed_vec, cand_vecs):
    """
    BUY seed → SELL candidates (range-aware).
    Missing fields are excluded from the average.
    """
    if seed_meta.get("type") != "buy":
        return compliance_scores(seed_meta, cand_metas, seed_vec, cand_vecs)

    m = cand_metas
    year = _as_float_array(m, "year")
    mileage = _as_float_array(m, "mileage")
    ins_mo = _as_float_array(m, "insurance_mo")
    price = _as_float_array(m, "price")
    lat = _as_float_array(m, "lat")
    lon = _as_float_array(m, "lon")
    col_ids = [x.get("color_id", "") for x in m]

    # BUY ranges (treat 0/0 as "no preference")
    def _rng(lo_key, hi_key):
        lo_v = seed_meta.get(lo_key);
        hi_v = seed_meta.get(hi_key)
        try:
            lo_f = float(lo_v);
            hi_f = float(hi_v)
            if lo_f == 0 and hi_f == 0:
                return None, None
            return lo_f, hi_f
        except Exception:
            return None, None

    y_lo, y_hi = _rng("from_year", "to_year")
    km_lo, km_hi = _rng("from_km", "to_km")
    ins_lo, ins_hi = _rng("from_insurance_mo", "to_insurance_mo")
    p_lo, p_hi = _rng("from_price", "to_price")

    slat = seed_meta.get("lat")
    slon = seed_meta.get("lon")
    scol = seed_meta.get("color_id", "")

    def _rng_sim_or_nan(vals, lo, hi, fn):
        if lo is None or hi is None:
            return np.full_like(vals, np.nan, dtype=float)
        mask = ~np.isnan(vals)
        out = np.full_like(vals, np.nan, dtype=float)
        out[mask] = fn(vals[mask], lo, hi)
        return out

    s_year = _rng_sim_or_nan(year, y_lo, y_hi, interval_similarity_np)
    s_mile = _rng_sim_or_nan(mileage, km_lo, km_hi, interval_similarity_np)
    s_ins = _rng_sim_or_nan(ins_mo, ins_lo, ins_hi, interval_similarity_np)
    s_price = _rng_sim_or_nan(price, p_lo, p_hi, price_against_range_similarity_np)

    if slat is None or slon is None or str(slat) == "" or str(slon) == "":
        s_geo = np.full_like(lat, np.nan, dtype=float)
    else:
        mask = (~np.isnan(lat)) & (~np.isnan(lon))
        s_geo = np.full_like(lat, np.nan, dtype=float)
        s_geo[mask] = geo_similarity_np(float(slat), float(slon), lat[mask], lon[mask])

    if not str(scol).strip():
        s_col = np.full(len(m), np.nan, dtype=float)
    else:
        s_col = colour_similarity_id(scol, col_ids).astype(float)
        has_cand_col = np.array([bool(str(c).strip()) for c in col_ids])
        s_col = np.where(has_cand_col, s_col, np.nan)

    s_vec = _cos_sim_01(seed_vec, cand_vecs)

    scores = _weighted_average(
        {
            "year": s_year,
            "mileage": s_mile,
            "insurance": s_ins,
            "price": s_price,
            "geo": s_geo,
            "color": s_col,
            "vec": s_vec,
        },
        WEIGHTS,
    )
    return scores
