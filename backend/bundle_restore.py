"""Select one image and its replay dependencies from a verified browse view."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from PIL import Image

from . import bundle_crypto, bundle_images, bundles, storage
from .bundle_archive import ArchiveSelection
from .bundle_common import BrowseUnavailable, BundleError, Progress, _rows
from .metadata import detailer_from_json, edit_from_json, tiled_refine_from_json
from .refusal_codes import CodedError


class RestoreCatalogue:
    def __init__(self, path: Path, archive: zipfile.ZipFile, rows: list[dict], manifest: dict, inspected: dict):
        self.path = path
        self.archive = archive
        self.rows = {row["id"]: row for row in rows}
        self.manifest = manifest
        self.files = manifest["files"]
        self.inspected = inspected

    def _pixel_facts(self, row: dict) -> bundle_images._Pixels:
        return self.inspected["images/" + row["filename"]]

    def _source_proven(self, row: dict, role: str, source: dict) -> bool:
        if role in {"refine", "reference"}:
            return True
        if row["id"] == source["id"]:
            return False
        tiled = None
        if role == "edit":
            edit = edit_from_json(row.get("edit"))
            assert edit is not None
            proof = edit.edit_source
        elif role == "detailer":
            detail = detailer_from_json(row.get("detailer"))
            assert detail is not None
            proof = detail.plan.source
        else:
            tiled = tiled_refine_from_json(row.get("tiled_refine"))
            assert tiled is not None
            proof = tiled.root_source
        pixels = self._pixel_facts(source)
        return (source["filename"] == proof.filename and (pixels.width, pixels.height) == (proof.width, proof.height)
                and pixels.rgb_hash == proof.hash
                and (tiled is None or pixels.prepared_rgb_hash == tiled.prepared_rgb_sha256))

    def _proven_source(self, row: dict, role: str, target: int | None) -> dict | None:
        source = self.rows.get(target) if target is not None else None
        if source is None or not self._source_proven(row, role, source):
            return None
        return source

    def _closure(self, image_id: int):
        if image_id not in self.rows:
            raise KeyError(image_id)
        pending = [image_id]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            row = self.rows[current]
            links = [(role, target, self._proven_source(row, role, target))
                     for role, target in bundle_images._image_links(row)]
            pending.extend(source["id"] for _, _, source in reversed(links) if source is not None)
            yield row, links

    def _live_rows(self) -> dict[str, dict]:
        return {row["filename"].casefold(): row for row in _rows(storage.settings.db_path)}

    def _present(self, row: dict, live: dict[str, dict]) -> dict | None:
        name = row["filename"]
        known = live.get(name.casefold())
        if known is None or known["filename"] != name:
            return None
        target = storage.settings.outputs_dir / name
        if bundles._is_link(target) or not target.is_file():
            return None
        pixels = self._pixel_facts(row)
        try:
            bundle_images._check_known_record(known, row, pixels.execution, pixels.edit)
            if bundle_images._validate_source_row(known, target).identity == pixels.identity:
                return known
        except BundleError:
            # A filename collision is still an available archived image. If
            # selected, the normal merge reports the conflict without writing.
            pass
        return None

    def _archived_mask(self, row: dict, digest: str) -> bool:
        if f"masks/{digest}.png" in self.files:
            return True
        detail = detailer_from_json(row.get("detailer"))
        if detail is not None and digest in detail.plan.masks:
            # The initial archive verification already checked each frozen mask.
            return True
        member = f"images/{row['filename']}"
        try:
            data = self.archive.read(member)
        except (OSError, zipfile.BadZipFile, bundle_crypto.CryptoError) as exc:
            raise BrowseUnavailable("the saved bundle changed since it was opened") from exc
        if hashlib.sha256(data).hexdigest() != self.files[member]["sha256"]:
            raise BrowseUnavailable("the saved bundle changed since it was opened")
        with Image.open(io.BytesIO(data)) as image:
            mask = storage.embedded_mask_picture(image, digest)
            if mask is not None:
                with mask:
                    if mask.size == (row["width"], row["height"]):
                        return True
            source = self.rows.get(row.get("refine"))
            spec = None
            if row.get("outpaint") is not None:
                spec = storage.OutpaintSpec.parse_key(row["outpaint"])
            elif source is not None:
                spec = storage.recovered_outpaint(image.info.get("parameters"), row["width"], row["height"],
                                                  digest, (source["width"], source["height"]))
        if spec is None:
            return False
        mask = storage.rebuilt_outpaint_mask(row["width"], row["height"], spec, digest)
        if mask is None:
            return False
        mask.close()
        return True

    def preview(self, image_id: int) -> dict:
        live = self._live_rows()
        present = {}
        dependencies = []
        for row, links in self._closure(image_id):
            for role, target, source in links:
                item = {"kind": "image", "id": target, "used_by": row["id"], "role": role, "status": "missing"}
                if source is not None:
                    if target not in present:
                        present[target] = self._present(source, live)
                    known = present[target]
                    item.update(status="present" if known is not None else "available", prompt=source["prompt"],
                                binned=bool((known if known is not None else source).get("binned_at")))
                dependencies.append(item)
            for digest in sorted(bundle_images._mask_hashes(row)):
                try:
                    path = storage.mask_path(digest)
                except CodedError:
                    mask = None
                else:
                    mask = None if bundles._is_link(path) else storage.open_mask(path, digest)
                if mask is not None:
                    mask.close()
                status = "present" if mask is not None else "available" if self._archived_mask(row, digest) else "missing"
                dependencies.append({"kind": "mask", "id": digest, "used_by": row["id"], "role": "mask", "status": status})
        return {"image_id": image_id, "dependencies": dependencies}

    def restore(self, image_id: int, include_ids: list[int], progress: Progress) -> dict:
        live = self._live_rows()
        related = {row["id"]: (row, links) for row, links in self._closure(image_id)}
        chosen = {image_id, *include_ids}
        if not chosen <= related.keys():
            raise BrowseUnavailable("only replay dependencies of this archived image can be restored with it")
        # Present dependencies participate in the ordinary import mapping but
        # are kept in place. Follow only links from rows taking part in this
        # restore, so an omitted branch does not add its own dependencies.
        pending = list(chosen)
        members = set()
        while pending:
            row, links = related[pending.pop()]
            for _, _, source in links:
                if source is not None and source["id"] not in chosen and self._present(source, live) is not None:
                    chosen.add(source["id"])
                    pending.append(source["id"])

        for archive_id in chosen:
            row, _ = related[archive_id]
            filename = row["filename"]
            thumb = Path(filename).with_suffix(".webp").name
            members.update({f"images/{filename}", f"images/{thumb}", f"images/thumbnails/{thumb}"})
            if row.get("refine") in chosen or row.get("detailer") is not None:
                members.update(f"masks/{digest}.png" for digest in bundle_images._mask_hashes(row))
        selection = ArchiveSelection(image_id, frozenset(chosen), frozenset(members & self.files.keys()), self.manifest)
        return bundles.restore_selection(self.path, self.archive, selection, progress)
