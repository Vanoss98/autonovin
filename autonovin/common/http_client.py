from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class HttpClient(ABC):
    """
    A simple “port” (interface) that says:
    any HTTP client implementation needs to provide .get() and .post().
    """

    @abstractmethod
    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform an HTTP GET request.
        Returns a dict (e.g. parsed JSON) or raises on network / HTTP errors.
        """
        raise NotImplementedError

    @abstractmethod
    def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform an HTTP POST request.
        Returns a dict (e.g. parsed JSON) or raises on network / HTTP errors.
        """
        raise NotImplementedError

    @abstractmethod
    def post_multipart(
            self,
            url: str,
            files: Dict[str, Any],
            data: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform an HTTP POST with multipart/form-data (e.g. file upload).
        - files: a dict of { fieldname: file-tuple-or‐file‐object }
        - data: form‐fields (non‐file fields)
        Returns parsed JSON as a dict (or raises on error).
        """
        raise NotImplementedError