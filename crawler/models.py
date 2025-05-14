from django.db import models


class Page(models.Model):
    source = models.CharField(max_length=255)
    title = models.CharField(max_length=255, blank=True)
    markdown_content = models.TextField()
    url = models.URLField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or self.url


class PageImage(models.Model):
    page = models.ForeignKey(Page, related_name='crawled_images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='crawled_page_images/')
    alt_text = models.CharField(max_length=255, blank=True)
    downloaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image for {self.page}"

