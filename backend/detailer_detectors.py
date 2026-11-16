"""Lazy, pinned CPU detectors for the explicit gallery detailer review.

Status never imports OpenCV, creates a cache, or contacts the network. Only
prepare() downloads assets; detection refuses missing or changed bytes.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import math
import os
import tempfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .backends.base import Cancelled

if TYPE_CHECKING:
    from PIL.Image import Image

    from .detailer import DetectorFacts, Region

RUNTIME = "opencv-python-headless"
RUNTIME_VERSION = "4.13.0.92"
ZOO_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
ANIME_REVISION = "4433ab1ae1166ea75acfe99eb0f18709dac329a0"
MAX_SOURCE_PIXELS = 16_777_216
MAX_SOURCE_SIDE = 8192
MAX_DETECTION_SIDE = 1536
MAX_CANDIDATES = 128
DOWNLOAD_TIMEOUT = 20
DOWNLOAD_BUDGET = 120


class DetailerDetectorError(RuntimeError):
    def __init__(self, detail: str, code: str = "detailer_detector_failed") -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str
    revision: str


@dataclass(frozen=True)
class Detector:
    id: str
    label: str
    kind: Literal["face", "hand"]
    assets: tuple[Asset, ...]
    note: str
    preprocessing: str


def _zoo_asset(folder: str, name: str, size: int, sha256: str) -> Asset:
    return Asset(name, f"https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                 f"{ZOO_REVISION}/models/{folder}/{name}", size, sha256, ZOO_REVISION)


DETECTORS = (
    Detector(
        "anime-face-lbp-v1", "Anime faces (LBP)", "face",
        (Asset("lbpcascade_animeface.xml", "https://raw.githubusercontent.com/nagadomi/"
               f"lbpcascade_animeface/{ANIME_REVISION}/lbpcascade_animeface.xml", 246945,
               "9376d30ac38db6bda2a68b88b3b76bbd7e6aa33af47f7f5c76bc88ca75f1ce30", ANIME_REVISION),),
        "Anime-trained face finder. Review every box; small, profile or obscured faces may be missed.",
        "rgb-bgr;max-side1536-linear;gray-equalize;scale1.1;neighbors5;min24;confidence1-unscored;box-floor-ceil-v1",
    ),
    Detector(
        "photo-face-yunet-v1", "Photographic faces (YuNet)", "face",
        (_zoo_asset("face_detection_yunet", "face_detection_yunet_2023mar.onnx", 232589,
                    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),),
        "Photo-trained face finder. Detection on illustrated faces is unqualified; review every box.",
        "rgb-bgr;max-side1536-linear;yunet-score0.5;nms0.3;top128;box-floor-ceil-v1",
    ),
    Detector(
        "photo-hands-opencv-v1", "Photographic hands (MediaPipe ONNX)", "hand",
        (_zoo_asset("palm_detection_mediapipe", "palm_detection_mediapipe_2023feb.onnx", 3905734,
                    "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"),
         _zoo_asset("handpose_estimation_mediapipe", "handpose_estimation_mediapipe_2023feb.onnx", 4099621,
                    "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c")),
        "Photo-trained hand finder. Illustrations, occlusion and malformed hands may be missed; review every box.",
        "rgb-bgr;max-side1536-linear;opencv-hands-v1-anchors64-inverse-affine;"
        "palm192-score0.5-nms0.3-top16;pose224-conf0.8;landmark-extents;box-floor-ceil-v1",
    ),
)


def _cancel(should_cancel: Callable[[], bool]) -> None:
    if should_cancel():
        raise Cancelled()


def _runtime_problem() -> str | None:
    try:
        version = importlib.metadata.version(RUNTIME)
        installed = [name for name in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless")
                     if _installed(name)]
        if installed:
            return "Detailing needs one OpenCV distribution; remove the other cv2 package before enabling it."
        if version != RUNTIME_VERSION or importlib.util.find_spec("cv2") is None:
            return f"Detailing needs {RUNTIME} {RUNTIME_VERSION}. Run uv sync, then restart AuraDream."
        return None
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        return "Restore the CPU detector runtime with uv sync, then restart AuraDream."


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _detector(detector_id: str) -> Detector:
    for detector in DETECTORS:
        if detector.id == detector_id:
            return detector
    raise DetailerDetectorError("Choose one of the listed face or hand detectors.", "detailer_unavailable")


class DetailerDetectors:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def status(self) -> list[dict[str, Any]]:
        """Presence/size only; detect and prepare verify complete hashes off-loop."""
        problem = _runtime_problem()
        answer = []
        for detector in DETECTORS:
            missing = []
            for asset in detector.assets:
                try:
                    if (self.cache_dir / asset.name).stat().st_size != asset.size:
                        missing.append(asset)
                except OSError:
                    missing.append(asset)
            answer.append({
                "id": detector.id, "label": detector.label, "classes": [detector.kind],
                "available": problem is None and not missing,
                # Size is only a cold hint. Explicit preparation must remain
                # available to repair same-size files that fail SHA-256.
                "prepare_available": problem is None,
                "reason": problem or ("Download this detector before finding regions." if missing else None),
                "note": detector.note, "download_bytes": sum(asset.size for asset in missing),
                "asset_bytes": sum(asset.size for asset in detector.assets),
                "assets_cached": not missing, "runtime": f"{RUNTIME} {RUNTIME_VERSION}",
            })
        return answer

    def prepare(self, detector_id: str, should_cancel: Callable[[], bool] = lambda: False) -> None:
        detector = _detector(detector_id)
        problem = _runtime_problem()
        if problem:
            raise DetailerDetectorError(problem, "detailer_unavailable")
        _cancel(should_cancel)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for asset in detector.assets:
            _cancel(should_cancel)
            path = self.cache_dir / asset.name
            if not _verified(path, asset, should_cancel):
                self._download(path, asset, should_cancel)

    @staticmethod
    def _download(path: Path, asset: Asset, should_cancel: Callable[[], bool]) -> None:
        part: Path | None = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=asset.name + ".", suffix=".part", dir=path.parent)
            part = Path(temporary)
            started = time.monotonic()
            digest, size = hashlib.sha256(), 0
            with os.fdopen(fd, "wb") as output:
                request = urllib.request.Request(asset.url, headers={"User-Agent": "AuraDream-detailer/1"})
                with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                    while True:
                        _cancel(should_cancel)
                        if time.monotonic() - started > DOWNLOAD_BUDGET:
                            raise DetailerDetectorError("The detector download timed out. Try Download again.")
                        block = response.read(min(65536, asset.size + 1 - size))
                        if not block:
                            break
                        size += len(block)
                        if size > asset.size:
                            raise DetailerDetectorError("The detector download is larger than its pinned asset.")
                        digest.update(block)
                        output.write(block)
            _cancel(should_cancel)
            if size != asset.size or digest.hexdigest() != asset.sha256:
                raise DetailerDetectorError("The detector download did not match its pinned size and SHA-256.")
            os.replace(part, path)
        except (OSError, ValueError) as exc:
            raise DetailerDetectorError(f"The detector download failed: {exc}") from exc
        finally:
            if part is not None:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass

    def detect(self, detector_id: str, image: Image,
               should_cancel: Callable[[], bool] = lambda: False) -> tuple[list[Region], list[DetectorFacts]]:
        """Return canvas boxes, not masks. Never redetect at queue/replay time."""
        from .detailer import AssetFacts, DetectorFacts, Region

        detector = _detector(detector_id)
        problem = _runtime_problem()
        if problem:
            raise DetailerDetectorError(problem, "detailer_unavailable")
        width, height = image.size
        if min(width, height) < 24 or max(width, height) > MAX_SOURCE_SIDE or width * height > MAX_SOURCE_PIXELS:
            raise DetailerDetectorError("The source is outside this CPU detector's image-size limits.")
        paths = [self.cache_dir / asset.name for asset in detector.assets]
        for path, asset in zip(paths, detector.assets):
            _cancel(should_cancel)
            if not _verified(path, asset, should_cancel):
                raise DetailerDetectorError("The detector asset is missing or changed. Download it again before finding regions.",
                                            "detailer_unavailable")
        try:
            # Import only on the admitted worker, after the explicit setup.
            cv: Any = importlib.import_module("cv2")
            import numpy as np

            _cancel(should_cancel)
            bgr = cv.cvtColor(np.asarray(image.convert("RGB")), cv.COLOR_RGB2BGR)
            if max(width, height) > MAX_DETECTION_SIDE:
                scale = MAX_DETECTION_SIDE / max(width, height)
                bgr = cv.resize(bgr, (max(24, round(width * scale)), max(24, round(height * scale))),
                                interpolation=cv.INTER_LINEAR)
            xscale, yscale = width / bgr.shape[1], height / bgr.shape[0]
            _cancel(should_cancel)
            if detector.id == "anime-face-lbp-v1":
                model = cv.CascadeClassifier(str(paths[0]))
                if model.empty():
                    raise DetailerDetectorError("The anime face cascade could not be loaded.")
                gray = cv.equalizeHist(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY))
                boxes = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
                raw = [(float(x), float(y), float(x + w), float(y + h), 1.0) for x, y, w, h in boxes]
            elif detector.id == "photo-face-yunet-v1":
                model = cv.FaceDetectorYN.create(str(paths[0]), "", (bgr.shape[1], bgr.shape[0]),
                                                0.5, 0.3, MAX_CANDIDATES, cv.dnn.DNN_BACKEND_OPENCV,
                                                cv.dnn.DNN_TARGET_CPU)
                _, faces = model.detect(bgr)
                raw = [] if faces is None else [(float(row[0]), float(row[1]), float(row[0] + row[2]),
                                                float(row[1] + row[3]), float(row[-1])) for row in faces]
            else:
                from .detailer_hands import detect_hands

                raw = detect_hands(cv, np, bgr, paths[0], paths[1], should_cancel)
            _cancel(should_cancel)
            if len(raw) > MAX_CANDIDATES:
                raise DetailerDetectorError("The detector returned too many regions; use a smaller source.")
            for candidate in raw:
                if len(candidate) != 5 or not all(math.isfinite(value) for value in candidate):
                    raise DetailerDetectorError("The detector returned non-finite coordinates or confidence.")
            raw.sort(key=lambda box: (box[1], box[0], box[3], box[2], -box[4]))
            regions = []
            for index, (left, top, right, bottom, confidence) in enumerate(raw):
                if not all(math.isfinite(value) for value in (left, top, right, bottom, confidence)):
                    raise DetailerDetectorError("The detector returned non-finite coordinates or confidence.")
                if right <= left or bottom <= top or not 0 <= confidence <= 1:
                    raise DetailerDetectorError("The detector returned an invalid box or confidence.")
                box = (max(0, math.floor(left * xscale)), max(0, math.floor(top * yscale)),
                       min(width, math.ceil(right * xscale)), min(height, math.ceil(bottom * yscale)))
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise DetailerDetectorError("The detector returned a region outside the image.")
                regions.append(Region(id=f"{detector.id}:{index}", kind=detector.kind, box=box,
                                      confidence=confidence, detector=detector.id))
            facts = DetectorFacts(
                id=detector.id, runtime=RUNTIME, version=RUNTIME_VERSION,
                asset_revision=detector.assets[0].revision, asset_sha256=detector.assets[0].sha256,
                preprocessing=f"{detector.preprocessing};cv={cv.__version__};numpy={np.__version__}",
                additional_assets=[AssetFacts(id=asset.name, revision=asset.revision, sha256=asset.sha256)
                                   for asset in detector.assets[1:]],
            )
            return regions, [facts]
        except Cancelled:
            raise
        except DetailerDetectorError:
            raise
        except Exception as exc:
            raise DetailerDetectorError(f"The {detector.label} detector failed: {exc}") from exc


def _verified(path: Path, asset: Asset, should_cancel: Callable[[], bool]) -> bool:
    try:
        if path.stat().st_size != asset.size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(65536):
                _cancel(should_cancel)
                digest.update(block)
        return digest.hexdigest() == asset.sha256
    except OSError:
        return False
