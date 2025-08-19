# /safety/constitution.py
import json, jsonschema, hashlib, datetime, subprocess, os

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "toriconstitution.schema.json")

class Constitution:
    def __init__(self, path="constitution.json"):
        with open(SCHEMA_PATH) as f:
            self.schema = json.load(f)
        with open(path) as f:
            self.doc = json.load(f)
        jsonschema.validate(self.doc, self.schema)

    # ---------- invariants ----------
    def assert_resource_budget(self, usage):
        b = self.doc["resource_budget"]
        assert usage.cpu <= b["cpu_core_seconds"]
        assert usage.gpu <= b["gpu_seconds"]
        assert usage.ram <= b["ram_bytes"]

    def assert_action(self, syscall_name):
        if syscall_name in self.doc["safety_rules"]["forbidden_calls"]:
            raise PermissionError(f"Call {syscall_name} blocked by constitution")

    # ---------- rollback ----------
    def rollback(self, commit_hash):
        hist = self.doc["rollback"]["history_path"]
        quorum = self.doc["rollback"]["quorum"]
        # verify critics quorum already stored in audit log
        if not self._quorum_met(commit_hash, quorum):
            raise RuntimeError("Rollback quorum not met")
        subprocess.run(["git", "reset", "--hard", commit_hash], check=True)

    # ---------- helper ----------
    def _quorum_met(self, commit, q):
        # toy implementation: count stamped approvals in audit log
        try:
            with open(".audit/approvals.log") as f:
                approvals = [l for l in f if commit in l]
            return len(approvals) >= q
        except FileNotFoundError:
            return False
