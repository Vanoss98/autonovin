# serializers.py
from rest_framework import serializers

class BrandModelRetrieveSerializer(serializers.Serializer):
    id = serializers.CharField()  # accept "sell:2001", "buy:3001", or "2001"
    exclude_seed = serializers.BooleanField(required=False, default=True)
    threshold = serializers.FloatField(required=False, default=0.70)
    type = serializers.ChoiceField(choices=["sell", "buy"], required=False)
    nationalCode = serializers.CharField(required=True, allow_blank=False, trim_whitespace=True)

    def validate(self, attrs):
        raw = str(attrs["id"]).strip()
        t = attrs.get("type")

        if ":" in raw:
            # already "sell:2001" / "buy:3001"
            attrs["id"] = raw
            return attrs

        # bare numeric id → add a sensible prefix
        # if client provided type, use it; else default to 'sell'
        prefix = t if t in ("sell", "buy") else "sell"
        attrs["id"] = f"{prefix}:{raw}"
        return attrs
