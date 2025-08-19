# tori/utils/component_registry.py
from threading import Lock

class ComponentRegistry:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._status = {}
        return cls._instance

    def mark_ready(self, name: str):
        self._status[name] = True

    def mark_not_ready(self, name: str):
        self._status[name] = False

    def all_ready(self) -> bool:
        return all(self._status.values()) and bool(self._status)

    def component_status(self):
        return dict(self._status)
