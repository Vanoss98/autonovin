import numpy as np

# ---------- colour utilities --------------------------------------------
HSV_HUES = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "cyan": 180,
            "blue": 240, "purple": 270, "magenta": 300, "black": None,
            "white": None, "gray": None, "silver": None}


def colour_similarity_np(target, colours):
    """
    Vector colour similarity 0-1.
    • Exact string match  → 1.0
    • Otherwise HSV-hue distance scaled to [0,1].
    • If either colour has no defined hue (e.g. black/white/gray), score = 0.
    """
    import numpy as np

    # seed colour
    t = str(target).strip().lower()
    h1 = HSV_HUES.get(t)  # int | None
    h1 = float(h1) if h1 is not None else np.nan

    # candidate hues array (float, NaN where unknown)
    arr = np.array(
        [HSV_HUES.get(str(c).lower()) for c in colours],
        dtype=float
    )
    # mask: rows where BOTH hues are valid numbers
    hue_ok = (~np.isnan(arr)) & (~np.isnan(h1))

    # hue distance (180° wrap-around)
    dh = np.where(
        hue_ok,
        np.minimum(np.abs(arr - h1), 360 - np.abs(arr - h1)),
        0.0
    )
    sim = np.where(hue_ok, 1 - dh / 180, 0.0)

    # exact string match overrides everything
    exact = np.array([str(c).lower() == t for c in colours])
    sim[exact] = 1.0
    return sim


# ---------- geo helpers --------------------------------------------------
def haversine_np(lat1, lon1, lats, lons):
    """
    lat1, lon1  – scalar (seed ad)
    lats, lons  – 1-D arrays (candidates)
    returns 1-D array of great-circle distances in km
    """
    R = 6371.0
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    dlat = np.radians(lats - lat1)  # array
    dlon = np.radians(lons - lon1)
    a = np.sin(dlat / 2) ** 2 + (
            np.cos(np.radians(lat1)) *
            np.cos(np.radians(lats)) *
            np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def geo_similarity_np(lat1, lon1, lats, lons, σ=50):
    return np.exp(-haversine_np(lat1, lon1, lats, lons) / σ)


# ---------- numeric similarity ------------------------------------------
TOL_PCT = 0.10  # ±10 % around seed is perfect        (score = 1)
DECAY_PCT = 0.50  # ±50 % around seed decays to zero    (score = 0)


def numeric_similarity_np(values, seed_val):
    """
    1.0   if |value - seed| <= TOL_PCT * seed
    0..1  linear between ±TOL_PCT and ±DECAY_PCT
    0.0   if difference >= DECAY_PCT * seed
    """
    values = np.asarray(values, dtype=float)
    seed = float(seed_val)

    diff_pct = np.abs(values - seed) / (seed + 1e-9)  # relative error
    inside = diff_pct <= TOL_PCT
    decay = (DECAY_PCT - diff_pct) / (DECAY_PCT - TOL_PCT)
    return np.where(
        inside, 1.0,
        np.clip(decay, 0.0, 1.0)
    )


DISCOUNT_GOOD = 0.20  # ≤ 20 % cheaper counts as perfect
DISCOUNT_MAX = 0.50  # > 50 % cheaper => 0
MARKUP_TOL = 0.10  # ≤ 10 % dearer still perfect
MARKUP_MAX = 0.50  # > 50 % dearer => 0


def price_similarity_np(values, seed_price):
    """
    Asymmetric: cheaper up to 20 % is great; too cheap (>50 %) or too expensive
    (>50 % dearer) scores 0; linear decay in between.
    """
    v = np.asarray(values, dtype=float)
    sp = float(seed_price)

    # cheaper side
    cheaper_frac = (sp - v) / sp  # positive if cheaper
    cheaper_good = cheaper_frac <= DISCOUNT_GOOD
    cheaper_bad = cheaper_frac >= DISCOUNT_MAX
    cheaper_mid = (~cheaper_good & ~cheaper_bad)
    s_cheaper = np.where(
        cheaper_good, 1.0,
        np.where(
            cheaper_bad, 0.0,
            1.0 - (cheaper_frac - DISCOUNT_GOOD) / (DISCOUNT_MAX - DISCOUNT_GOOD)
        )
    )

    # dearer side
    dearer_frac = (v - sp) / sp  # positive if dearer
    dearer_good = dearer_frac <= MARKUP_TOL
    dearer_bad = dearer_frac >= MARKUP_MAX
    dearer_mid = (~dearer_good & ~dearer_bad)
    s_dearer = np.where(
        dearer_good, 1.0,
        np.where(
            dearer_bad, 0.0,
            1.0 - (dearer_frac - MARKUP_TOL) / (MARKUP_MAX - MARKUP_TOL)
        )
    )

    # combine
    return np.where(v <= sp, s_cheaper, s_dearer)


# ---------- weights ------------------------------------------------------
BASE_WEIGHTS = {"year": 0.20, "mileage": 0.12, "insurance": 0.08,
                "price": 0.12, "geo": 0.18, "color": 0.10}
VEC_WT = 0.20  # embedding-cosine component
WEIGHTS = {**BASE_WEIGHTS, "vec": VEC_WT}
WEIGHT_SUM = sum(WEIGHTS.values())
WEIGHTS = {k: v / WEIGHT_SUM for k, v in WEIGHTS.items()}

MAX = {"year": 10, "mileage": 40000, "insurance": 12, "price": 10000}


# ---------- main scorer --------------------------------------------------
def compliance_scores(seed_meta, cand_metas, seed_vec, cand_vecs):
    """
    Returns np.ndarray of similarity scores (0-1) for each cand_meta.
    · seed_meta  = dict
    · cand_metas = list[dict]
    · seed_vec   = 1-D embedding
    · cand_vecs  = 2-D list/array of embeddings
    """
    m = cand_metas
    to_f = lambda k: np.array([float(x[k]) for x in m], dtype=float)

    s_year = numeric_similarity_np(to_f("year"), seed_meta["year"])
    s_mile = numeric_similarity_np(to_f("mileage"), seed_meta["mileage"])
    s_ins = numeric_similarity_np(to_f("insurance_mo"), seed_meta["insurance_mo"])
    s_price = price_similarity_np(to_f("price"), seed_meta["price"])
    s_geo = geo_similarity_np(seed_meta["lat"], seed_meta["lon"], to_f("lat"), to_f("lon"))
    s_col = colour_similarity_np(seed_meta["color"], [x["color"] for x in m])

    # cosine between seed_vec and each cand
    sv = np.asarray(seed_vec, dtype=float)
    cv = np.asarray(cand_vecs, dtype=float)
    dot = (cv @ sv) / (np.linalg.norm(cv, axis=1) * np.linalg.norm(sv) + 1e-9)
    s_vec = 0.5 * (dot + 1)  # map [-1,1] → [0,1]

    return (
            WEIGHTS["year"] * s_year +
            WEIGHTS["mileage"] * s_mile +
            WEIGHTS["insurance"] * s_ins +
            WEIGHTS["price"] * s_price +
            WEIGHTS["geo"] * s_geo +
            WEIGHTS["color"] * s_col +
            WEIGHTS["vec"] * s_vec
    )
