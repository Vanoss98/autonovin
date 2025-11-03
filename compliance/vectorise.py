# -*- coding: utf-8 -*-
# compliance/vectorise.py
def ad_to_text(ad: dict) -> str:
    t = (ad.get("type") or "").lower()
    color_id = ad.get("color_id")
    color_en = ad.get("color_name_en") or ""

    if t == "buy":
        return (
            f"BUY brand:{ad.get('brand_id')} model:{ad.get('model_id')} "
            f"color_id:{color_id} color_en:{color_en} "
            f"year:{ad.get('from_year')}-{ad.get('to_year')} "
            f"km:{ad.get('from_km')}-{ad.get('to_km')} "
            f"price:{ad.get('from_price')}-{ad.get('to_price')} "
            f"insurance_mo:{ad.get('from_insurance_mo')}-{ad.get('to_insurance_mo')} "
            f"@({ad.get('lat')},{ad.get('lon')})"
        )
    return (
        f"SELL brand:{ad.get('brand_id')} model:{ad.get('model_id')} "
        f"color_id:{color_id} color_en:{color_en} "
        f"year:{ad.get('year')} km:{ad.get('mileage')} "
        f"price:{ad.get('price')} insurance_mo:{ad.get('insurance_mo')} "
        f"@({ad.get('lat')},{ad.get('lon')})"
    )
