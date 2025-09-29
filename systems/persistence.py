import os
import torch
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class PersistenceManager:
    """
    Manages atomic checkpointing, roll-forward, and integrity checks for agent state.
    """
    def __init__(self, checkpoint_dir, retain_n=5):
        self.checkpoint_dir = checkpoint_dir
        self.retain_n = retain_n
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self):
        path = Path(self.checkpoint_dir)
        current = Path(path.anchor) if path.is_absolute() else Path()
        for part in path.parts:
            if part in ('', path.anchor, path.root):
                continue
            if current == Path():
                current = Path(part)
            else:
                current = current / part
            try:
                os.mkdir(current)
            except FileExistsError:
                continue

    def _get_checkpoint_path(self, step):
        return os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")

    def _get_manifest_path(self):
        return os.path.join(self.checkpoint_dir, "manifest.json")

    def _calculate_crc32(self, filepath):
        """Calculates the CRC32 checksum of a file."""
        with open(filepath, "rb") as f:
            file_hash = hashlib.crc32(f.read())
        return file_hash

    def save_checkpoint(self, state, step):
        """
        Atomically saves a checkpoint.

        Args:
            state (dict): The state to save (e.g., model weights, optimizer states).
            step (int): The current training step.
        """
        path = self._get_checkpoint_path(step)
        tmp_path = f"{path}.tmp"

        try:
            torch.save(state, tmp_path)
            crc32 = self._calculate_crc32(tmp_path)
            os.rename(tmp_path, path)
            self._update_manifest(step, path, crc32)
            self._cleanup_old_checkpoints()
            logger.info(f"Saved checkpoint at step {step} to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _update_manifest(self, step, path, crc32):
        manifest = self._read_manifest()
        manifest[step] = {
            "path": path,
            "timestamp": datetime.utcnow().isoformat(),
            "crc32": crc32
        }

        # Sort by step to keep the manifest clean
        sorted_manifest = {k: manifest[k] for k in sorted(manifest.keys(), reverse=True)}

        with open(self._get_manifest_path(), "w") as f:
            json.dump(sorted_manifest, f, indent=4)

    def _read_manifest(self):
        manifest_path = self._get_manifest_path()
        if not os.path.exists(manifest_path):
            return {}
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _cleanup_old_checkpoints(self):
        manifest = self._read_manifest()
        if len(manifest) <= self.retain_n:
            return

        steps_to_remove = sorted(manifest.keys())[:-self.retain_n]
        for step in steps_to_remove:
            path = manifest[step]["path"]
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed old checkpoint: {path}")
            del manifest[step]

        with open(self._get_manifest_path(), "w") as f:
            json.dump(manifest, f, indent=4)

    def load_latest_checkpoint(self, verify_integrity=True):
        """
        Loads the latest valid checkpoint.

        Args:
            verify_integrity (bool): If True, verifies the checksum before loading.

        Returns:
            The loaded state dict, or None if no valid checkpoint is found.
        """
        manifest = self._read_manifest()
        if not manifest:
            logger.warning("No checkpoints found.")
            return None

        latest_step = sorted(manifest.keys(), reverse=True)[0]
        checkpoint_info = manifest[latest_step]
        path = checkpoint_info["path"]

        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found: {path}. Trying previous versions.")
            # This could be extended to try older checkpoints
            return None

        if verify_integrity:
            crc32_expected = checkpoint_info.get("crc32")
            if crc32_expected is None:
                logger.warning(f"No CRC32 checksum found for checkpoint {path}. Loading without verification.")
            else:
                crc32_actual = self._calculate_crc32(path)
                if crc32_actual != crc32_expected:
                    logger.error(f"Integrity check failed for {path}. Checksum mismatch. Aborting load.")
                    # Fallback to degraded mode would be handled by the caller
                    return None

        try:
            state = torch.load(path)
            logger.info(f"Successfully loaded checkpoint from {path} at step {latest_step}.")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")
            return None

    def has_checkpoints(self):
        return bool(self._read_manifest())