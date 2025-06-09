import requests
from typing import Any, Dict, Optional
from requests.exceptions import RequestException
from autonovin.common.http_client import HttpClient


class RequestsHttpClient(HttpClient):
    """
    A concrete HttpClient implementation that uses `requests`.
    """

    def __init__(self, default_timeout: float = 10.0):
        # you can tweak session, retries, etc. as needed
        self._session = requests.Session()
        self._default_timeout = default_timeout

    def get(
            self,
            url: str,
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            resp = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout or self._default_timeout
            )
            resp.raise_for_status()
            # attempt to parse JSON; if it fails, we propagate
            return resp.json()
        except RequestException as e:
            # You can repackage or log
            raise RuntimeError(f"HTTP GET failed: {e}") from e

    def post(
            self,
            url: str,
            json: Optional[Dict[str, Any]] = None,
            data: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            resp = self._session.post(
                url,
                json=json,
                data=data,
                headers=headers,
                timeout=timeout or self._default_timeout
            )
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            raise RuntimeError(f"HTTP POST failed: {e}") from e

    def post_multipart(
            self,
            url: str,
            files: Dict[str, Any],
            data: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Example usage of `files`:
            files = {
                "File": open("/path/to/file.xlsx", "rb"),
            }
        Or if you need to specify a filename/content-type tuple:
            files = {
                "File": ("some_name.xlsx", open("/path/to/file.xlsx", "rb"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            }
        """
        try:
            resp = self._session.post(
                url,
                files=files,
                data=data,
                headers=headers,
                timeout=timeout or self._default_timeout
            )
            resp.raise_for_status()
            return resp.json()
        except RequestException as e:
            raise RuntimeError(f"HTTP multipart POST failed: {e}") from e
