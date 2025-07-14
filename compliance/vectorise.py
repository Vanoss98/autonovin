
def ad_to_text(ad: dict) -> str:
    """Deterministic short sentence used for the OpenAI embedding."""
    return (
        f"{ad['year']} {ad['brand']} {ad['model']} — "
        f"{ad['color']}, {ad['mileage']} km, ${ad['price']}, "
        f"{ad['insurance_mo']} mo insurance"
    )
