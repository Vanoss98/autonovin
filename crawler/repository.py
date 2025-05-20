from asgiref.sync import sync_to_async
from django.db import transaction
from .models import Page, PageImage
from django.utils import timezone


class BaseRepository:
    def __init__(self, model):
        self.model = model

    def all(self):
        """Return only non-deleted records."""
        return self.model.objects.filter(deleted_at__isnull=True)

    def get_by_id(self, pk):
        """Retrieve a non-deleted record by ID."""
        try:
            return self.model.objects.get(pk=pk, deleted_at__isnull=True)
        except self.model.DoesNotExist:
            return None

    def create(self, data):
        """Create a new record and generate a slug if 'name' exists."""
        return self.model.objects.create(**data)

    def delete(self, obj):
        """Soft delete: set deleted_at timestamp instead of deleting."""
        obj.deleted_at = timezone.now()
        obj.save()
        return True  # Indicate success

    def restore(self, pk):
        """Restore a soft-deleted record."""
        instance = self.get_by_id(pk)
        if instance and instance.deleted_at:
            instance.deleted_at = None
            instance.save()
            return instance
        return None

    def filter(self, data):
        """Filter only non-deleted records based on criteria."""
        return self.model.objects.filter(**data, deleted_at__isnull=True)

    def filter_exclude(self, data, excluded_data):
        return self.model.objects.filter(**data, deleted_at__isnull=True).exclude(**excluded_data)

    def update(self, pk, data):
        """Update a record, ensuring it's not soft-deleted."""
        instance = self.get_by_id(pk)
        if not instance:
            return None  # Prevent updating deleted records

        for key, value in data.items():
            setattr(instance, key, value)
        instance.save()
        return instance

    def get_by_field(self, key, value):
        """Retrieve a record by a specific field, excluding soft-deleted records."""
        return self.model.objects.filter(**{key: value}, deleted_at__isnull=True).first()

    async def acreate(self, data: dict):
        return await sync_to_async(self.create, thread_sensitive=True)(data)

    async def aget_by_id(self, pk):
        return await sync_to_async(self.get_by_id, thread_sensitive=True)(pk)

    async def aexists(self, **lookups):
        exists_fn = lambda: self.model.objects.filter(**lookups).exists()
        return await sync_to_async(exists_fn, thread_sensitive=True)()


class PageRepository(BaseRepository):
    def __init__(self):
        super().__init__(model=Page)

    def create(self, data):
        page = super().create(data)

        def _enqueue():
            # local import breaks the circular chain
            from .tasks import download_images
            download_images.delay(page.id)

        transaction.on_commit(_enqueue)
        return page


class PageImageRepository(BaseRepository):
    def __init__(self):
        super().__init__(model=PageImage)

    def exists(self, page_id, url) -> bool:
        return self.model.objects.filter(page_id=page_id, url=url).exists()
