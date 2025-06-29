from asgiref.sync import async_to_sync
from rest_framework import status, viewsets
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from crawler.infrastructure.models import Page
from crawler.infrastructure.repositories.page_repository import PageRepository
from .application.services.chunker_service import ChunkerService
from .application.usecases.chunker_usecase import PageChunkingUseCase
from .service import index_pages, chroma_client
from rest_framework.views import APIView
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from django.conf import settings

class Retrieve(APIView):
    def post(self, request):

        query = request.data.get("query", "").strip()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
        store = Chroma(
            client=chroma_client,
            collection_name="car_spec",
            embedding_function=embeddings,
            # persist_directory=os.path.join(settings.BASE_DIR, "vectors"),
        )
        docs = store.similarity_search(query, k=4)
        results = [
            {
                "metadata": d.metadata,
                "content": d.page_content
            }
            for d in docs
        ]
        return Response({"data": results})


class List(APIView):
    def get(self, request):
        queryset = Page.objects.all().values('id')
        return Response({'list': queryset})


class Embedder(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        page_id = self.kwargs['pk']
        if not page_id:
            return Response({'error': 'no valid object'}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        page_repo = PageRepository()
        service = ChunkerService()
        usecase = PageChunkingUseCase(chunker_service=service, repository=page_repo)

        # Use async_to_sync to execute the async method in a sync view
        result = async_to_sync(usecase.execute)(page_id)

        return Response(
            {
                'chunks': result
            },
            status=status.HTTP_201_CREATED,
        )