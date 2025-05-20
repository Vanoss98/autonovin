from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .tasks import execute_crawling
from .service import CrawlService


class CrawlExecutionView(APIView):
    def post(self, request):
        service = CrawlService()
        serializer = service.serializer(data=request.data)
        if not serializer.is_valid():
            return Response({"errors": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        config = serializer.validated_data
        task = execute_crawling.apply_async(args=[config])
        return Response({"message": "Crawling started", "task_id": task.id}, status=status.HTTP_202_ACCEPTED)
