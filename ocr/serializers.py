from rest_framework import serializers


class PlateReplaceInputSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)
