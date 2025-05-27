from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError
from .serializers import CrawlConfigSerializer
from .tasks import page_crawler_task


class CrawlExecutionView(APIView):
    def post(self, request):
        serializer = CrawlConfigSerializer(data=self.request.data)
        if serializer.is_valid():
            config = serializer.validated_data
            # Trigger the Celery task chain
            task = page_crawler_task.apply_async((config,))

            return Response(
                {"task_id": task.id},
                status=status.HTTP_202_ACCEPTED
            )
        else:
            raise ValidationError(serializer.errors)