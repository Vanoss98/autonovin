from rest_framework import serializers


class CrawlConfigSerializer(serializers.Serializer):
    start_url = serializers.URLField()
    include_patterns = serializers.ListField(
        child=serializers.CharField(), required=False, default=[]
    )
    exclude_patterns = serializers.ListField(
        child=serializers.CharField(), required=False, default=[]
    )
    max_depth = serializers.IntegerField(min_value=1)
    max_pages = serializers.IntegerField(min_value=1)
    excluded_selector = serializers.CharField(required=False, allow_blank=True, default="")
    excluded_tags = serializers.ListField(
        child=serializers.CharField(), required=False, default=[]
    )
