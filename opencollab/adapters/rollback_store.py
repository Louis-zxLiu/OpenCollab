"""Memory and optional authenticated-encrypted rollback state stores."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class MemoryRollbackCheckpointStore:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def save(self, run_id: str, state: Mapping[str, Any]) -> None:
        self._states[run_id] = json.loads(json.dumps(state))

    def load(self, run_id: str) -> dict[str, Any] | None:
        value = self._states.get(run_id)
        return json.loads(json.dumps(value)) if value is not None else None

    def delete(self, run_id: str) -> None:
        self._states.pop(run_id, None)


class EncryptedRollbackCheckpointStore:
    """Persist state with AES-GCM; construction fails when crypto is unavailable."""

    def __init__(self, directory: str | os.PathLike[str], key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("rollback encryption key must be exactly 32 bytes")
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise RuntimeError("encrypted rollback persistence requires the cryptography extra") from exc
        self._aesgcm = AESGCM(key)
        self._directory = Path(directory).resolve()
        self._directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self._directory, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    def _path(self, run_id: str) -> Path:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not run_id or any(char not in allowed for char in run_id):
            raise ValueError("invalid rollback state identifier")
        return self._directory / f"{run_id}.state"

    def save(self, run_id: str, state: Mapping[str, Any]) -> None:
        nonce = os.urandom(12)
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        blob = nonce + self._aesgcm.encrypt(nonce, payload, run_id.encode())
        target = self._path(run_id)
        fd, temporary = tempfile.mkstemp(prefix=f".{run_id}.", dir=self._directory)
        try:
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, "wb") as stream:
                stream.write(blob)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def load(self, run_id: str) -> dict[str, Any] | None:
        target = self._path(run_id)
        try:
            blob = target.read_bytes()
        except FileNotFoundError:
            return None
        if len(blob) < 13:
            raise ValueError("corrupted rollback state")
        try:
            payload = self._aesgcm.decrypt(blob[:12], blob[12:], run_id.encode())
            value = json.loads(payload)
        except Exception as exc:
            raise ValueError("rollback state authentication failed") from exc
        if not isinstance(value, dict):
            raise ValueError("corrupted rollback state")
        return value

    def delete(self, run_id: str) -> None:
        try:
            self._path(run_id).unlink()
        except FileNotFoundError:
            pass
