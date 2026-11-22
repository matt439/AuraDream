"""A dated, bounded copy of popular Danbooru tags and their active aliases."""
from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import gallery_lock, sidecar_io, tags

PAGE_SIZE = 1000
PAGES = 10
MAX_TAGS = PAGE_SIZE * PAGES
MAX_FILE_BYTES = 8 * 1024 * 1024
# Danbooru allows anonymous clients two requests a second. Leave room for a
# suggestion or Check being made while this user-initiated download runs.
PAGE_INTERVAL = 1.0


class SnapshotError(RuntimeError):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


class SnapshotChange(BaseModel):
    action: Literal["download", "offline", "live"]


class SnapshotStatus(BaseModel):
    snapshot_date: str | None = None
    enabled: bool = False
    tag_count: int = 0
    limit: int = MAX_TAGS
    detail: str | None = None


class _Alias(BaseModel):
    model_config = ConfigDict(strict=True)
    antecedent_name: str = Field(min_length=1, max_length=tags.TAG_NAME_MAX)
    status: str


class _Tag(BaseModel):
    model_config = ConfigDict(strict=True)
    name: str = Field(min_length=1, max_length=tags.TAG_NAME_MAX)
    category: int = Field(ge=0, le=5)
    post_count: int = Field(gt=0)
    created_at: str
    is_deprecated: bool
    consequent_aliases: list[_Alias]

    @field_validator("created_at")
    @classmethod
    def _date_is_valid(cls, value: str) -> str:
        return date.fromisoformat(value[:10]).isoformat()


class _SnapshotData(BaseModel):
    model_config = ConfigDict(strict=True)
    version: Literal[1] = 1
    captured_on: date
    enabled: bool
    tags: list[_Tag] = Field(min_length=1, max_length=MAX_TAGS)


class OfflineTagVocabulary:
    """A lookup reads one complete snapshot even if a refresh publishes another."""
    def __init__(self, data: _SnapshotData):
        self.data = data
        self.snapshot_date = data.captured_on.isoformat()
        ordered = sorted(data.tags, key=lambda row: (-row.post_count, row.name))
        self._records = {tags.normalize_tag(row.name): row.model_dump(exclude={"consequent_aliases"}) for row in ordered}
        self._aliases: dict[str, list[str]] = {}
        self._canonical: dict[str, str] = {}
        for row in ordered:
            name = tags.normalize_tag(row.name)
            for alias in row.consequent_aliases:
                if alias.status == "active":
                    antecedent = tags.normalize_tag(alias.antecedent_name)
                    self._aliases.setdefault(name, []).append(antecedent)
                    self._canonical.setdefault(antecedent, name)

    def suggest(
        self, query: str, cutoff: str | None, limit: int = tags.SUGGESTION_LIMIT, *, artist_only: bool = False,
    ) -> list[dict]:
        wanted = tags.normalize_tag(query).replace("*", "").replace("?", "")
        if not wanted:
            return []
        # Include an exact name or alias even where another tag has more posts.
        exact = self._canonical.get(wanted, wanted)
        names = dict.fromkeys((wanted, exact, *self._records))
        found = []
        for name in names:
            row = self._records.get(name)
            if row is None or not tags.tag_eligible(row, cutoff):
                continue
            if artist_only and row["category"] != 1:
                continue
            if wanted in name or any(wanted in alias for alias in self._aliases.get(name, ())):
                found.append(tags.tag_suggestion(row))
                if len(found) == limit:
                    break
        return found

    def validate(
        self, requested: list[str], special: set[str], cutoff: str | None, allow_year_tag: bool = True,
    ) -> list[dict]:
        verdicts = []
        for tag in requested:
            if tags.is_model_tag(tag, special, allow_year_tag):
                verdicts.append(tags.model_tag_result(tag))
                continue
            name = tags.normalize_tag(tag)
            canonical = self._canonical.get(name)
            target = self._records.get(canonical) if canonical else None
            verdict = tags.tag_verdict(tag, self._records.get(name), target, cutoff)
            if verdict["code"] == "not_found":
                if target is not None:
                    verdict = tags.tag_verdict(tag, target, None, cutoff)
                else:
                    verdict["reason"] = f"absent from the Danbooru snapshot dated {self.snapshot_date}"
            verdicts.append(verdict)
        return verdicts

    def status(self) -> SnapshotStatus:
        return SnapshotStatus(
            snapshot_date=self.snapshot_date, enabled=self.data.enabled, tag_count=len(self._records),
        )


# Only one server-side path is in use at a time. The atomic sidecar's stat also
# lets another process publish a replacement without leaving an old cache live.
_cache_lock = threading.Lock()
_cache: tuple[Path, tuple[int, int, int], OfflineTagVocabulary] | None = None
_update_lock = threading.Lock()


def path_for(database: Path) -> Path:
    return database.parent / "auradream.tags.json"


def _load(path: Path) -> OfflineTagVocabulary | None:
    global _cache
    with _cache_lock:
        try:
            stat = path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if _cache is not None and _cache[:2] == (path, signature):
                return _cache[2]
            with sidecar_io.open_read(path) as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES:
                raise ValueError("snapshot exceeds 8 MiB")
            data = _SnapshotData.model_validate_json(raw)
            index = OfflineTagVocabulary(data)
            _cache = (path, signature, index)
            return index
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise SnapshotError("The offline tag index could not be read. Download a fresh copy.") from exc


def enabled(database: Path) -> OfflineTagVocabulary | None:
    snapshot = _load(path_for(database))
    return snapshot if snapshot is not None and snapshot.data.enabled else None


def status(database: Path) -> SnapshotStatus:
    try:
        snapshot = _load(path_for(database))
    except SnapshotError as exc:
        return SnapshotStatus(detail=str(exc))
    return snapshot.status() if snapshot is not None else SnapshotStatus()


def _download(fetch: Callable[[str, dict[str, str]], list[dict]]) -> _SnapshotData:
    rows: dict[str, _Tag] = {}
    for page in range(1, PAGES + 1):
        if page > 1:
            time.sleep(PAGE_INTERVAL)
        response = fetch("/tags.json", {
            "page": str(page),
            "limit": str(PAGE_SIZE),
            "search[hide_empty]": "true",
            "search[order]": "count",
            "only": "name,category,post_count,created_at,is_deprecated,consequent_aliases[antecedent_name,status]",
        })
        if len(response) != PAGE_SIZE:
            raise ValueError(f"Danbooru returned an incomplete page ({len(response)} of {PAGE_SIZE} tags)")
        count_before = len(rows)
        for raw in response:
            tag = _Tag.model_validate(raw)
            rows.setdefault(tags.normalize_tag(tag.name), tag)
        if len(rows) == count_before:
            raise ValueError("Danbooru repeated a page")
    return _SnapshotData(captured_on=datetime.now(timezone.utc).date(), enabled=True, tags=list(rows.values()))


def _publish(path: Path, data: _SnapshotData) -> None:
    encoded = data.model_dump_json().encode("utf-8")
    if len(encoded) > MAX_FILE_BYTES:
        raise ValueError("snapshot exceeds 8 MiB")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".auradream.tags.", suffix=".part", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        sidecar_io.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def change(database: Path, request: SnapshotChange) -> SnapshotStatus:
    path = path_for(database)
    if not _update_lock.acquire(blocking=False):
        raise SnapshotError("The tag index is already being updated. Try again when it finishes.", 409)
    try:
        with gallery_lock.hold(path):
            if request.action == "download":
                data = _download(tags.request_json)
            else:
                previous = _load(path)
                if previous is None:
                    raise SnapshotError("Download an offline tag index first.", 409)
                data = previous.data.model_copy(update={"enabled": request.action == "offline"})
            _publish(path, data)
            return OfflineTagVocabulary(data).status()
    except gallery_lock.GalleryBusy as exc:
        raise SnapshotError("Another AuraDream process is updating the tag index.", 409) from exc
    except (tags.TagLookupUnavailable, OSError, ValueError) as exc:
        raise SnapshotError("The tag index update failed. Any previous snapshot is still available.") from exc
    finally:
        _update_lock.release()
