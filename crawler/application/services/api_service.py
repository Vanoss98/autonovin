import os
from typing import Any, Dict, Optional
from autonovin.common.http_client import HttpClient


class ApiService:
    """
    A service class (application layer) that performs GET/POST and, in particular,
    can upload an Excel file to a predefined endpoint.
    """

    def __init__(self, http_client: HttpClient, base_url: str):
        self._http = http_client
        self._base_url = base_url.rstrip("/")

    def fetch_resource(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        return self._http.get(url, params=params, headers=headers)

    def create_resource(
        self,
        path: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        return self._http.post(url, json=payload, headers=headers)

    def import_excel(
        self,
        path: str,
        excel_path: str,
        product_id: Optional[int] = None,
        type_value: int = 2,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload an Excel file + ProductId + Type (always = 2).
        - path: e.g. "api/services/app/CarsAndBrandService/Import"
        - excel_path: filesystem path to .xlsx
        - product_id: integer (can be None or omitted if the API accepts null)
        - type_value: always 2 per swagger doc
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        # Open the file in binary mode
        with open(excel_path, "rb") as f:
            files = {
                "File": (os.path.basename(excel_path), f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            }
            data = {
                # The API expects exactly "ProductId" and "Type"
                "ProductId": str(product_id) if product_id is not None else "",
                "Type": str(type_value),
            }
            return self._http.post_multipart(url, files=files, data=data, headers=headers)