import os
import requests


class ImageUploadService:
    """
    Knows how to send only Title + DesktopFile in a multipart/form-data POST.
    """

    def __init__(self, api_url: str, auth_token: str = None):
        self.api_url = api_url
        self.auth_token = auth_token

    def _guess_content_type(self, image_path: str) -> str:
        _, ext = os.path.splitext(image_path.lower())
        if ext in (".jpg", ".jpeg"):
            return "image/jpeg"
        elif ext == ".png":
            return "image/png"
        elif ext == ".gif":
            return "image/gif"
        else:
            return "application/octet-stream"

    def upload_desktop_file(
        self,
        title: str,
        image_path: str,
        entity_id: int = None,
        timeout: int = 60
    ) -> dict:
        """
        Streams a single image file under the "DesktopFile" form‐field,
        along with the Title (and EntityId if available).
        Omits all other fields.

        Returns parsed JSON or raises on HTTP error.
        """

        # 1) Build the form fields for text values:
        data = {
            "Title": title or "",
            "AttachmentCategoryId": 17,
            "EntityId":None
        }
        # If the API expects EntityId even if it's null, you can include it; otherwise omit:
        if entity_id is not None:
            data["EntityId"] = str(entity_id)

        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        # 2) Open the image file and build the `files` dict.
        content_type = self._guess_content_type(image_path)
        with open(image_path, "rb") as f:
            # key must match "DesktopFile" exactly
            files = {
                "DesktopFile": (os.path.basename(image_path), f, content_type)
            }
            # 3) Let `requests` build the multipart/form-data for us:
            response = requests.post(
                self.api_url,
                data=data,
                files=files,
                headers=headers,
                timeout=timeout
            )

        response.raise_for_status()
        return response.json()

