# plate_api/views.py
from io import BytesIO
import cv2
from django.conf import settings
from django.http import HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt  # ← new
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser  # ← good practice
from .usecases import ReplacePlateUseCase, ReplacePlateCommand
from .services import YoloPlateReplacer
from .serializers import PlateReplaceInputSerializer
from django.contrib.staticfiles import finders

_DET_WEIGHTS = settings.YOLO_WEIGHTS_PATH
_CUSTOM_PLATE = finders.find("autonovin-plate.png")

_replacer = YoloPlateReplacer(_DET_WEIGHTS, _CUSTOM_PLATE)
_usecase = ReplacePlateUseCase(_replacer)


@method_decorator(csrf_exempt, name="dispatch")  # ← add this
class ReplacePlateAPIView(APIView):
    """
    POST an image, receive the processed image (or 204 if none found).
    CSRF disabled because clients are not expected to have a session cookie.
    """
    authentication_classes = []  # disables SessionAuthentication
    permission_classes = []  # open endpoint
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        ser = PlateReplaceInputSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        in_file = ser.validated_data["image"]
        command = ReplacePlateCommand(image_bytes=in_file.read())
        result = _usecase.execute(command)

        if result.image is None:
            return HttpResponse(status=status.HTTP_204_NO_CONTENT)

        _, buf = cv2.imencode(".png", result.image)
        return HttpResponse(buf.tobytes(), content_type="image/png")
