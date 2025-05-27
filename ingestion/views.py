from rest_framework import status, viewsets
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from crawler.infrastructure.models import Page
from .service import index_pages, chroma_client
from rest_framework.views import APIView
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class Retrieve(APIView):
    def post(self, request):

        query = request.data.get("query", "").strip()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
        store = Chroma(
            client=chroma_client,
            collection_name="chroma-test",
            embedding_function=embeddings,
            # add persist_directory="…" if you saved the DB somewhere custom
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
        page = get_object_or_404(Page, pk=page_id)
        chunk_map = index_pages(Page.objects.filter(pk=page.pk))
        chunk_ids = chunk_map.get(page.pk, [])

        return Response(
            {
                "page_id": page.pk,
                "chunks_written": len(chunk_ids),
                "chunk_ids": chunk_ids,
            },
            status=status.HTTP_201_CREATED,
        )

