from django.db import models


class Page(models.Model):
    entity_id = models.IntegerField(null=True)
    source = models.CharField(max_length=255)
    title = models.CharField(max_length=255, blank=True, null=True)
    markdown_file = models.FileField(upload_to='crawled_docs', null=True)
    url = models.URLField(unique=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True)

    def __str__(self):
        return self.title or self.url


class PageImage(models.Model):
    page = models.ForeignKey(Page, on_delete=models.CASCADE, related_name="images")
    url = models.URLField(unique=False)
    file = models.FileField(upload_to="page_images/")     # stores to MEDIA_ROOT/page_images
    downloaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["page", "url"])]
        unique_together = [("page", "url")]