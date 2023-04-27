import json
import os
from pathlib import Path
from typing import Any

import requests

from nebullvm.config import VERSION

NEBULLVM_METADATA_PATH = Path.home() / ".nebullvm/collect.json"


class FeedbackCollector:
    def __init__(
        self, url: str, disable_telemetry_environ_var: str, app_version: str
    ):
        self._disable_telemetry_environ_var = disable_telemetry_environ_var
        self._is_active = (
            int(os.getenv(disable_telemetry_environ_var, "0")) == 0
        )
        self._url = url
        self._metadata = {
            "nebullvm_version": VERSION,
            "app_version": app_version,
        }

    def _store_ip_address(self):
        try:
            self._metadata["ip_address"] = requests.get(
                "https://api.ipify.org"
            ).text
        except Exception:
            self._metadata["ip_address"] = "Unknown"

    @property
    def is_active(self):
        return self._is_active

    def _inform_user(self):
        message = (
            f"Nebuly collects anonymous usage statistics to help improve the "
            f"product. You can opt-out by setting the environment variable "
            f"{self._disable_telemetry_environ_var}=1."
        )
        print(message)

    def store_info(self, key: str, value: Any):
        if key in self._metadata and isinstance(value, list):
            self._metadata[key] += value
        else:
            self._metadata[key] = value

    def send_feedback(self, timeout: int = 30):
        if not self.is_active:
            return {}
        self._store_ip_address()
        request_body = self._metadata
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self._url,
            data=json.dumps(request_body),
            headers=headers,
            timeout=timeout,
        )
        return response

    def get(self, key: str, default: Any = None):
        return self._metadata.get(key, default)

    def reset(self, key: str):
        self._metadata.pop(key, None)
