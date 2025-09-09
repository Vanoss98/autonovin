# compliance/vectorise.py
def ad_to_text(ad: dict) -> str:
    """
    Backwards-compat helper (if some code still imports this).
    Prefer the typed versions in tasks.py (ad_to_text_sell / ad_to_text_buy).
    """
    t = (ad.get("type") or "").lower()
    if t == "buy":
        return (
            f"BUY brand:{ad.get('brand_id')} model:{ad.get('model_id')} "
            f"color:{ad.get('color_id')} "
            f"year:{ad.get('from_year')}-{ad.get('to_year')} "
            f"km:{ad.get('from_km')}-{ad.get('to_km')} "
            f"price:{ad.get('from_price')}-{ad.get('to_price')} "
            f"insurance_mo:{ad.get('from_insurance_mo')}-{ad.get('to_insurance_mo')} "
            f"@({ad.get('lat')},{ad.get('lon')})"
        )
    # default SELL
    return (
        f"SELL brand:{ad.get('brand_id')} model:{ad.get('model_id')} "
        f"color:{ad.get('color_id')} "
        f"year:{ad.get('year')} km:{ad.get('mileage')} "
        f"price:{ad.get('price')} insurance_mo:{ad.get('insurance_mo')} "
        f"@({ad.get('lat')},{ad.get('lon')})"
    )
