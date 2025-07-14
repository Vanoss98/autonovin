from rest_framework import serializers


class BrandModelRetrieveSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    exclude_seed = serializers.BooleanField(required=False, default=True)
    threshold = serializers.FloatField(required=False, default=0.70)
