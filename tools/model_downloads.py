"""Check backend/model-downloads.json offline, or refresh it from the Hub.

    python tools/model_downloads.py --check               # offline: shape and catalogue
    python tools/model_downloads.py [MODEL_ID ...]        # refresh from the Hub, a dry run
    python tools/model_downloads.py --write [MODEL_ID ...]   # ... and save the file

The file is what `model_info` reads for Models' download size and cache
state, and what `hub_access` asks the Hub about: per catalogue id, the Hub
sources its first load reads (`sources`, each a repository at a revision,
with the observed commit, every file's size and the small JSON descriptors
kept as `indexes`) and the files the loader selects from each.

`--check` reaches no network and imports no diffusers: every catalogue id
(each official repo and each lite-infer id the app writes, never a renamed
alias) has a recipe, every selected file is in its source with a size, a
pipeline's shard maps have all their shards selected, a selection that is
not yet known says why (`unknown`), and FLUX 3's follow its loader's
patterns. It exits 1 on a problem and lists the ids whose size stays
unknown.

A refresh asks the Hub for metadata only, with the token huggingface_hub
already finds (never printed): each source's listing at its revision (a
pinned commit must still be the commit), and the JSON descriptors (at most
2 MB each, never a weight) the selections need. A source whose repository
moved (lite-infer's renames, followed through the Hub's 307) keeps the id
the app asks for and records `moved_to`. A selection that is unknown is
derived: FLUX 3's from `flux3_loading`'s snapshot patterns, any other
pipeline's from diffusers' own `DiffusionPipeline.download` allowlist over
its `model_index.json`. A recorded selection is kept, and a file it names
that is gone is an error for a person to resolve. Converted outputs are
local inventories and are never touched.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# config before anything that could import diffusers (AGENTS.md).
from backend import config as _config  # noqa: E402,F401
from backend.backends import flux3_loading  # noqa: E402
from backend.launch import CHECKPOINTS, FORMATS  # noqa: E402

DATA = ROOT / "backend" / "model-downloads.json"
API = "https://huggingface.co/api/models/"
# The largest descriptor read; model_info refuses to compare a bigger one.
DESCRIPTOR_LIMIT = 2_000_000
TOP_KEYS = {"models", "observed_at", "selection_basis", "sources"}
KINDS = {"pipeline", "converted"}
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ROOT_DESCRIPTORS = ("model_index.json", flux3_loading.INDEX)


def catalogue_ids() -> set[str]:
    """Every id the app writes: the official repos and lite-infer's ids."""
    return {entry.repo for entry in CHECKPOINTS} | {
        repo for entry in CHECKPOINTS for fmt, _ in FORMATS if (repo := entry.lite_repo(fmt))}


def relative(name: str) -> bool:
    path = PurePosixPath(name)
    return (bool(path.parts) and not path.is_absolute() and "\\" not in name and ":" not in name
            and all(part not in {".", ".."} for part in path.parts))


def is_descriptor(name: str) -> bool:
    """A JSON file a selection keeps the contents of: the pipeline index or a shard map."""
    return PurePosixPath(name).name in ROOT_DESCRIPTORS or name.endswith(".index.json")


def flux3_selection(filenames: list[str]) -> list[str]:
    """The files `flux3_loading.resolve_model`'s two snapshot passes fetch."""
    patterns = [*flux3_loading.CONFIG_PATTERNS, *flux3_loading.WEIGHT_PATTERNS]
    return sorted(name for name in filenames if any(fnmatch.fnmatch(name, p) for p in patterns))


def diffusers_selection(repo_id: str, filenames: list[str], index: dict) -> list[str]:
    """The files `DiffusionPipeline.download` fetches for a plain load (no
    variant, safetensors, no components passed), from the installed diffusers'
    own helpers, as its `download` computes `expected_files`."""
    from diffusers.pipelines import pipeline_loading_utils as utils
    from diffusers.pipelines.pipeline_loading_utils import CUSTOM_PIPELINE_FILE_NAME
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
    from diffusers.utils.constants import CONFIG_NAME, TRANSFORMERS_COMPONENT_AUX_FILES

    config = dict(index)
    names = set(filenames) - set(config.pop("_ignore_files", []))
    custom, folders = utils._get_custom_components_and_folders(repo_id, config, sorted(names))
    model_folders = {os.path.split(f)[0] for f in utils.filter_model_files(names) if os.path.split(f)[0] in folders}
    # download's own arguments: no components passed, safetensors without
    # pickles, onnx unset (the pipeline class's False), no flashpack, no variant.
    use_onnx: Any = None
    ignore = utils._get_ignore_patterns([], list(model_folders), list(names), True, False, use_onnx, False, False, None)
    # Annotated as a list or str, it returns (model files, variant files).
    siblings: Any = utils.variant_compatible_siblings(names, variant=None, ignore_patterns=ignore)
    allow = [str(name) for name in siblings[0]]
    allow += [f"{k}/*" for k in folders if k not in model_folders]
    allow += [f"{k}/{f}.py" for k, f in custom.items()]
    allow += [f"{k}/config.json" for k in model_folders]
    allow += [SCHEDULER_CONFIG_NAME, CONFIG_NAME, DiffusionPipeline.config_name, CUSTOM_PIPELINE_FILE_NAME]
    allow += TRANSFORMERS_COMPONENT_AUX_FILES
    ignore = ignore + [f"{i}.index.*json" for i in ignore]
    denied = [re.compile(fnmatch.translate(p)) for p in ignore]
    allowed = [re.compile(fnmatch.translate(p)) for p in allow]
    return sorted(f for f in names if not any(p.match(f) for p in denied) and any(p.match(f) for p in allowed))


# --- the offline check -------------------------------------------------------


def check(data: Any) -> list[str]:
    """Every problem with the file's shape and its agreement with the catalogue."""
    problems: list[str] = []
    if not isinstance(data, dict) or set(data) != TOP_KEYS:
        return [f"top-level keys must be exactly {sorted(TOP_KEYS)}"]
    if not isinstance(data["observed_at"], str) or not DATE.match(data["observed_at"]):
        problems.append("observed_at is not a YYYY-MM-DD date")
    if not isinstance(data["selection_basis"], list) or not all(isinstance(s, str) and s for s in data["selection_basis"]):
        problems.append("selection_basis must be a list of sentences")
    sources, models = data["sources"], data["models"]
    expected = catalogue_ids()
    for missing in sorted(expected - set(models)):
        problems.append(f"{missing}: catalogue id without a recipe")
    for extra in sorted(set(models) - expected):
        problems.append(f"{extra}: recipe for an id the catalogue does not write")
    for key, source in sources.items():
        problems += [f"source {key}: {p}" for p in _source_problems(key, source)]
    used: set[str] = set()
    for model_id, recipe in models.items():
        for problem in _recipe_problems(model_id, recipe, sources, used):
            problems.append(f"{model_id}: {problem}")
    for orphan in sorted(set(sources) - used):
        problems.append(f"source {orphan}: no recipe reads it")
    return problems


def _source_problems(key: str, source: Any) -> list[str]:
    if not isinstance(source, dict):
        return ["not an object"]
    allowed = {"repo_id", "revision", "commit", "files", "indexes", "url", "moved_to"}
    if not {"repo_id", "revision", "commit", "files", "indexes", "url"} <= set(source) <= allowed:
        return [f"keys must be {sorted(allowed)} (moved_to optional)"]
    problems = []
    if key != f"{source['repo_id']}@{source['revision']}":
        problems.append("key is not repo_id@revision")
    if not isinstance(source["commit"], str) or not COMMIT.match(source["commit"]):
        problems.append("commit is not a 40-hex sha")
    elif COMMIT.match(str(source["revision"])) and source["revision"] != source["commit"]:
        problems.append("a pinned revision's commit differs from the pin")
    if not isinstance(source["url"], str) or not source["url"].startswith(API):
        problems.append(f"url does not start with {API}")
    if "moved_to" in source and (not isinstance(source["moved_to"], str) or source["moved_to"] == source["repo_id"]):
        problems.append("moved_to must name another repository")
    files, indexes = source["files"], source["indexes"]
    if not isinstance(files, dict) or not files:
        return problems + ["files must map names to sizes"]
    for name, size in files.items():
        if not relative(name) or type(size) is not int or size < 0:
            problems.append(f"bad file entry {name!r}")
    if not isinstance(indexes, dict):
        return problems + ["indexes must be an object"]
    for name, value in indexes.items():
        if name not in files:
            problems.append(f"index {name} is not one of its files")
        elif not isinstance(value, dict) or files[name] > DESCRIPTOR_LIMIT:
            problems.append(f"index {name} is not a small JSON object")
    return problems


def _recipe_problems(model_id: str, recipe: Any, sources: dict, used: set[str]) -> list[str]:
    if not isinstance(recipe, dict) or recipe.get("kind") not in KINDS:
        return [f"kind must be one of {sorted(KINDS)}"]
    allowed = {"kind", "sources", "observed_at", "unknown", "converted"}
    if not set(recipe) <= allowed:
        return [f"unexpected keys {sorted(set(recipe) - allowed)}"]
    if (recipe["kind"] == "converted") != ("converted" in recipe):
        return ["a converted recipe, and only one, has a converted inventory"]
    if "observed_at" in recipe and not (isinstance(recipe["observed_at"], str) and DATE.match(recipe["observed_at"])):
        return ["observed_at is not a YYYY-MM-DD date"]
    items = recipe.get("sources")
    if not isinstance(items, list) or not items:
        return ["sources must be a nonempty list"]
    problems = []
    unknown = False
    for item in items:
        if not isinstance(item, dict) or set(item) != {"source", "files"}:
            problems.append("each source is {source, files}")
            continue
        key = item["source"]
        used.add(key)
        source = sources.get(key)
        if source is None or not isinstance(source.get("files"), dict):
            problems.append(f"source {key} is not recorded")
            continue
        names = item["files"]
        if names is None:
            unknown = True
            continue
        if not isinstance(names, list) or not names or len(set(names)) != len(names):
            problems.append(f"{key}: files must be a nonempty list without repeats")
            continue
        for name in names:
            if name not in source["files"] or not source["files"][name]:
                problems.append(f"{key}: selected {name} is not one of its sized files")
            elif recipe["kind"] == "pipeline" and is_descriptor(name) and name not in source["indexes"]:
                problems.append(f"{key}: selected descriptor {name} has no retained contents")
        if recipe["kind"] == "pipeline":
            for name in names:
                if name.endswith(".safetensors.index.json") and isinstance(source["indexes"].get(name), dict):
                    shards = source["indexes"][name].get("weight_map", {}).values()
                    folder = PurePosixPath(name).parent
                    missing = {str(folder / s) for s in shards} - set(names)
                    if missing:
                        problems.append(f"{key}: {name} names unselected shards {sorted(missing)}")
            if flux3_loading.matches(model_id) and sorted(names) != flux3_selection(list(source["files"])):
                problems.append(f"{key}: selection differs from flux3_loading's snapshot patterns")
    note = recipe.get("unknown")
    if unknown and not (isinstance(note, str) and note):
        problems.append("a selection is unknown without an `unknown` note saying why")
    if not unknown and "unknown" in recipe:
        problems.append("every selection is known, so its `unknown` note is stale")
    return problems


def unknown_sizes(data: dict) -> list[str]:
    """The ids Models shows an unknown download size for."""
    return sorted(model_id for model_id, recipe in data["models"].items()
                  if any(item["files"] is None for item in recipe["sources"]))


# --- the refresh from the Hub ------------------------------------------------


@dataclass(frozen=True)
class Listing:
    repo_id: str  # the repository's name now (a renamed one's new name)
    commit: str
    files: dict[str, int]


class Hub:
    """huggingface_hub metadata calls, with the token it finds itself."""

    def __init__(self) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi()

    def listing(self, repo_id: str, revision: str) -> Listing:
        info = self.api.model_info(repo_id, revision=revision, files_metadata=True)
        files = {s.rfilename: int(s.size or 0) for s in info.siblings or []}
        return Listing(repo_id=str(info.id), commit=str(info.sha), files=files)

    def descriptor(self, repo_id: str, commit: str, name: str) -> Any:
        """One small JSON file's contents, read into memory: nothing is cached."""
        from huggingface_hub import get_session, hf_hub_url
        from huggingface_hub.utils._headers import build_hf_headers
        from huggingface_hub.utils._http import hf_raise_for_status

        response = get_session().get(hf_hub_url(repo_id, name, revision=commit), headers=build_hf_headers())
        hf_raise_for_status(response)
        if len(response.content) > DESCRIPTOR_LIMIT:
            raise ValueError(f"{repo_id}/{name} is larger than a descriptor")
        return response.json()


def _source_url(repo_id: str, revision: str) -> str:
    return f"{API}{repo_id}/revision/{revision}?blobs=true"


def _select(model_id: str, key: str, source: dict, read: Callable[[str], Any]) -> list[str]:
    files = list(source["files"])
    if flux3_loading.matches(model_id):
        return flux3_selection(files)
    if "model_index.json" not in source["files"]:
        raise ValueError(f"{key} has no model_index.json to derive a selection from")
    return diffusers_selection(source["repo_id"], files, read("model_index.json"))


def refresh(data: dict, hub: Any, *, only: list[str] | None = None,
            today: str | None = None) -> tuple[dict, list[str]]:
    """A refreshed copy of `data` and what changed. `hub` answers `listing`
    and `descriptor` (`Hub`, or a test's stand-in)."""
    from huggingface_hub.errors import HfHubHTTPError

    today = today or datetime.date.today().isoformat()
    data = copy.deepcopy(data)
    report: list[str] = []
    model_ids = only or sorted(data["models"])
    for model_id in model_ids:
        if model_id not in data["models"]:
            raise SystemExit(f"{model_id} has no recipe in {DATA.name}")
    keys = list(dict.fromkeys(item["source"] for m in model_ids for item in data["models"][m]["sources"]))
    changed: set[str] = set()
    unreadable: dict[str, str] = {}
    for key in keys:
        source = data["sources"][key]
        try:
            listing = hub.listing(source["repo_id"], source["revision"])
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            unreadable[key] = f"{type(exc).__name__}{f' {status}' if status else ''}"
            report.append(f"{key}: not readable with this sign-in ({unreadable[key]}); left as recorded")
            continue
        if COMMIT.match(source["revision"]) and listing.commit != source["revision"]:
            raise SystemExit(f"{key}: the Hub answered commit {listing.commit} for a pinned revision")
        moved = listing.repo_id if listing.repo_id != source["repo_id"] else None
        # One commit is one content: a record that kept only some of its
        # files (Kontext's, from the cache) stays as it is while they agree.
        same = listing.commit == source["commit"] and all(
            listing.files.get(name) == size for name, size in source["files"].items())
        if same and moved == source.get("moved_to"):
            continue
        changed.add(key)
        old_commit = source["commit"]
        source.update(commit=listing.commit, files=listing.files, url=_source_url(source["repo_id"], source["revision"]))
        source.pop("moved_to", None)
        if moved:
            source["moved_to"] = moved
        if old_commit != listing.commit:
            source["indexes"] = {}  # read again below, at the new commit
        report.append(f"{key}: {'commit ' + listing.commit[:12] if old_commit != listing.commit else 'sizes'} "
                      f"refreshed{f', moved to {moved}' if moved else ''}")

    def reader(key: str) -> Callable[[str], Any]:
        source = data["sources"][key]

        def read(name: str) -> Any:
            if name not in source["indexes"]:
                size = source["files"].get(name)
                if size is None or not name.endswith(".json") or size > DESCRIPTOR_LIMIT:
                    raise ValueError(f"{key}: {name} is not a small JSON descriptor")
                source["indexes"][name] = hub.descriptor(source["repo_id"], source["commit"], name)
            return source["indexes"][name]
        return read

    for model_id in model_ids:
        recipe = data["models"][model_id]
        filled = False
        for item in recipe["sources"]:
            key = item["source"]
            if key in unreadable:
                continue
            source = data["sources"][key]
            if item["files"] is None and recipe["kind"] == "pipeline":
                item["files"] = _select(model_id, key, source, reader(key))
                filled = True
                report.append(f"{model_id}: {len(item['files'])} files selected from {key}")
            for name in item["files"] or []:
                if name not in source["files"]:
                    raise SystemExit(f"{model_id}: {key} no longer has selected {name}; resolve by hand")
                if recipe["kind"] == "pipeline" and is_descriptor(name):
                    reader(key)(name)
            if (key in changed and not filled and recipe["kind"] == "pipeline" and len(recipe["sources"]) == 1
                    and "model_index.json" in source["indexes"]):
                # A kept selection is a person's; say where diffusers would now differ.
                derived = diffusers_selection(source["repo_id"], list(source["files"]), source["indexes"]["model_index.json"])
                if derived != sorted(item["files"]):
                    report.append(f"{model_id}: note, diffusers' own selection now differs "
                                  f"(+{sorted(set(derived) - set(item['files']))} -{sorted(set(item['files']) - set(derived))})")
        for key in [item["source"] for item in recipe["sources"]]:
            # A kept descriptor nothing selects any more stays only while it is a file.
            indexes = data["sources"][key]["indexes"]
            for name in [n for n in indexes if n not in data["sources"][key]["files"]]:
                del indexes[name]
        if all(item["files"] is not None for item in recipe["sources"]):
            recipe.pop("unknown", None)
        elif any(item["source"] in unreadable for item in recipe["sources"]):
            reasons = ", ".join(sorted({unreadable[i["source"]] for i in recipe["sources"] if i["source"] in unreadable}))
            recipe["unknown"] = f"Not readable with the local sign-in on {today} ({reasons}); the size stays unknown."
        if filled or any(item["source"] in changed for item in recipe["sources"]):
            recipe["observed_at"] = today
    data["sources"] = dict(sorted(data["sources"].items()))
    return data, report


def dumps(data: dict) -> str:
    """The file's own layout: sorted keys, two-space indent, no final newline."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Check backend/model-downloads.json, or refresh it from the Hub.")
    parser.add_argument("--check", action="store_true", help="offline: validate the file, reach no network")
    parser.add_argument("--write", action="store_true", help="save the refreshed file (default: a dry run)")
    parser.add_argument("models", nargs="*", metavar="MODEL_ID", help="refresh only these recipes")
    args = parser.parse_args(argv)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    if args.check:
        if args.write or args.models:
            parser.error("--check takes no other arguments")
        problems = check(data)
        for problem in problems:
            print(f"FAIL {problem}")
        for model_id in unknown_sizes(data):
            print(f"unknown size: {model_id} ({data['models'][model_id].get('unknown', '')})")
        print(f"{DATA.name}: {len(problems)} problem(s), {len(unknown_sizes(data))} unknown size(s)")
        return 1 if problems else 0
    fresh, report = refresh(data, Hub(), only=args.models or None)
    for line in report:
        print(line)
    problems = check(fresh)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        return 1
    if args.write:
        DATA.write_text(dumps(fresh), encoding="utf-8", newline="\n")
        print(f"wrote {DATA.name}")
    else:
        print("dry run; --write saves it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
