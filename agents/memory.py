import json
import os
import shutil
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
#####################################################################
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
#####################################################################
class JSONMemory:

    def __init__(self, path: str = "agent_memory.json"):
        self.path = path                # file path for memory storage
        self.data: Dict[str, Any] = {
            "datasets": {},
            "notes": [],
        }
        self._load()  # loads existing memory if file exists
#####################################################################
    def _load(self) -> None:                #Loads memory from disk

        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)  # loads JSON memory

        except Exception:
            backup = self.path + ".bak"
            shutil.copy(self.path, backup) # backup corrupted file

            self.data = {
                "datasets": {},
                "notes": [{
                    "ts": now_iso(),
                    "msg": f"Memory reset. backup at {backup}"
                }]
            }
#####################################################################
    def save(self) -> None:   #Writes memory to file

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
#####################################################################
    def getDatasetRecord(self, fingerprint: str) -> Optional[Dict[str, Any]]: #Fetches stored info about a dataset

        return self.data.get("datasets", {}).get(fingerprint)
#####################################################################
    def findSimilarDataset(self, target: str, is_classification: bool) -> Optional[Dict[str, Any]]:

        datasets = self.data.get("datasets", {}) or {}

        for fp, record in datasets.items():
            if record.get("target") == target and record.get("is_classification") == is_classification:
                return record

        return None
#####################################################################
    def upsertDatasetRecord(self, fingerprint: str, record: Dict[str, Any]) -> None: #Inserts or updates dataset memory

        datasets = self.data.setdefault("datasets", {})

        existing = datasets.get(fingerprint)

        if existing:
            merged = dict(existing)

            
            history = existing.get("history", []) # Preserve history
            new_history = record.get("history", [])

            merged.update(record)
            merged["history"] = history + new_history

            datasets[fingerprint] = merged

        else:
            datasets[fingerprint] = record

        self.save()
