"""Persistent lineage archive for the Transgenerational Memory Weave."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


@dataclass
class LineageMetadata:
    """Structured metadata describing a lineage snapshot."""

    species: Optional[str] = None
    stage: Optional[str] = None
    stage_index: Optional[int] = None
    event: str = "unspecified"
    reason: Optional[str] = None
    episode: Optional[int] = None
    total_steps: Optional[int] = None
    environment_seed: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Drop keys that are ``None`` to keep manifest compact.
        return {k: v for k, v in payload.items() if v is not None}


@dataclass
class LineageRecord:
    """In-memory representation of an archived lineage snapshot."""

    record_id: str
    created_at: datetime
    metadata: LineageMetadata
    policy_state: Optional[Dict[str, Any]] = None
    viability_state: Optional[Dict[str, Any]] = None
    safety_state: Optional[Dict[str, Any]] = None
    affect_state: Optional[Dict[str, Any]] = None
    fisher_mask: Optional[Dict[str, torch.Tensor]] = None


class LineageArchive:
    """Stores and retrieves lineage data for agent respawns."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, root: str, max_records: int = 64):
        self.root = Path(root)
        self.max_records = max_records
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / self.MANIFEST_NAME
        self._manifest: Dict[str, Dict[str, Any]] = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------
    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            try:
                raw = json.load(handle)
            except json.JSONDecodeError:
                return {}
        return {key: value for key, value in raw.items() if isinstance(value, dict)}

    def _write_manifest(self) -> None:
        ordered = {
            key: self._manifest[key]
            for key in sorted(
                self._manifest.keys(),
                key=lambda item: self._manifest[item]["created_at"],
                reverse=True,
            )
        }
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(ordered, handle, indent=2)

    def _trim_manifest(self) -> None:
        if len(self._manifest) <= self.max_records:
            return
        # Drop the oldest entries based on the creation timestamp.
        sorted_ids = sorted(
            self._manifest.keys(),
            key=lambda item: self._manifest[item]["created_at"],
        )
        for record_id in sorted_ids[: len(self._manifest) - self.max_records]:
            entry = self._manifest.pop(record_id)
            payload_path = self.root / entry["payload"]
            if payload_path.exists():
                payload_path.unlink()
        self._write_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_snapshot(
        self,
        *,
        metadata: LineageMetadata,
        policy_state: Optional[Dict[str, Any]] = None,
        viability_state: Optional[Dict[str, Any]] = None,
        safety_state: Optional[Dict[str, Any]] = None,
        affect_state: Optional[Dict[str, Any]] = None,
        fisher_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> LineageRecord:
        """Persist a snapshot and update the manifest."""

        record_id = uuid.uuid4().hex
        created_at = datetime.utcnow()
        payload_name = f"{created_at.strftime('%Y%m%dT%H%M%S')}_{record_id}.pt"
        payload_path = self.root / payload_name

        payload = {
            "metadata": metadata.to_dict(),
            "policy_state": policy_state,
            "viability_state": viability_state,
            "safety_state": safety_state,
            "affect_state": affect_state,
            "fisher_mask": {
                name: tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in (fisher_mask or {}).items()
            },
        }
        torch.save(payload, payload_path)

        self._manifest[record_id] = {
            "created_at": created_at.isoformat(),
            "payload": payload_name,
            "metadata": metadata.to_dict(),
        }
        self._write_manifest()
        self._trim_manifest()

        return LineageRecord(
            record_id=record_id,
            created_at=created_at,
            metadata=metadata,
            policy_state=policy_state,
            viability_state=viability_state,
            safety_state=safety_state,
            affect_state=affect_state,
            fisher_mask=fisher_mask,
        )

    def iter_records(
        self,
        *,
        species: Optional[str] = None,
        stage: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterable[LineageRecord]:
        """Yield records filtered by metadata."""

        sorted_items: List[tuple[str, Dict[str, Any]]] = sorted(
            self._manifest.items(),
            key=lambda item: item[1]["created_at"],
            reverse=True,
        )
        count = 0
        for record_id, manifest_entry in sorted_items:
            meta = manifest_entry.get("metadata", {}) or {}
            if species and meta.get("species") != species:
                continue
            if stage and meta.get("stage") != stage:
                continue
            payload_path = self.root / manifest_entry["payload"]
            if not payload_path.exists():
                continue
            payload = torch.load(payload_path, map_location="cpu")
            metadata_obj = LineageMetadata(**(payload.get("metadata", {})))
            fisher_payload = payload.get("fisher_mask") or {}
            fisher = {
                name: (
                    tensor
                    if isinstance(tensor, torch.Tensor)
                    else torch.as_tensor(tensor)
                )
                for name, tensor in fisher_payload.items()
            }
            yield LineageRecord(
                record_id=record_id,
                created_at=datetime.fromisoformat(manifest_entry["created_at"]),
                metadata=metadata_obj,
                policy_state=payload.get("policy_state"),
                viability_state=payload.get("viability_state"),
                safety_state=payload.get("safety_state"),
                affect_state=payload.get("affect_state"),
                fisher_mask=fisher,
            )
            count += 1
            if limit is not None and count >= limit:
                break

    def latest_records(self, limit: int = 5) -> List[LineageRecord]:
        return list(self.iter_records(limit=limit))

    def build_report(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return lightweight report rows for CLI presentation."""

        rows: List[Dict[str, Any]] = []
        for record in self.iter_records(limit=limit):
            rows.append(
                {
                    "record_id": record.record_id,
                    "created_at": record.created_at.isoformat(timespec="seconds"),
                    "event": record.metadata.event,
                    "species": record.metadata.species,
                    "stage": record.metadata.stage,
                    "stage_index": record.metadata.stage_index,
                    "episode": record.metadata.episode,
                    "score": record.metadata.score,
                    "total_steps": record.metadata.total_steps,
                }
            )
        return rows

    def has_records(self) -> bool:
        return bool(self._manifest)

    def archive_path(self) -> str:
        return os.fspath(self.root)
