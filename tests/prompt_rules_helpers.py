"""Shared fixtures and request helpers for the prompt rules tests."""
from __future__ import annotations

from backend import guides, launch, prompt_rules
from backend.model_defaults import defaults_for
from backend.schemas import GenerateRequest


FLUX_DEV = "black-forest-labs/FLUX.1-dev"


KREA_DEV = "black-forest-labs/FLUX.1-Krea-dev"


SCHNELL = "black-forest-labs/FLUX.1-schnell"


KLEIN = "black-forest-labs/FLUX.2-klein-4B"


KREA2 = "krea/Krea-2-Turbo"


QWEN = "Qwen/Qwen-Image"


ERNIE = "baidu/ERNIE-Image-Turbo"


ANIMAGINE = "cagliostrolab/animagine-xl-4.0"


KOHAKU = "KBlueLeaf/Kohaku-XL-Zeta"


ILLUSTRIOUS = "OnomaAIResearch/Illustrious-XL-v2.0"


WAI = "LyliaEngine/waiIllustriousSDXL_v170"


Z_ANIME = "SeeSee21/Z-Anime"


ANIMA = "circlestone-labs/Anima"


ANIMA_AESTHETIC = "circlestone-labs/Anima@aesthetic-v1.1"


NETAYUME = "duongve/NetaYume-Lumina-Image-2.0"


NEWBIE = "NewBie-AI/NewBie-image-Exp0.1"


def _catalogue_repos() -> list[str]:
    return [c.repo for c in launch.CHECKPOINTS] + [
        c.lite_repo(fmt) for c in launch.CHECKPOINTS for fmt, _ in launch.FORMATS if c.lite
    ]


def _all_guides() -> dict[str, dict]:
    # Every guide constant, the two undistilled ones outside the catalogue too.
    return {name: getattr(guides, name) for name in dir(guides) if name.endswith("_GUIDE")}


def run(model_id: str, prompt: str, categories: dict[str, str] | None = None, **fields):
    """`check` as the route will call it: the request, and applied() on the
    checkpoint's own profile answers."""
    request = GenerateRequest(prompt=prompt, **fields)
    profile = defaults_for(model_id)
    applied = request.applied(
        negative_prompt_supported=profile["negative_prompt_supported"],
        cfg_threshold=profile["cfg_threshold"],
        mu_supported=profile["mu_supported"],
        true_cfg_supported=profile["true_cfg_supported"],
        reference_supported=profile["reference_supported"],
    )
    findings = prompt_rules.check(model_id, request, applied, categories)
    for finding in findings:
        text = prompt if finding.field == "prompt" else (request.negative_prompt or "")
        if finding.start is not None:
            assert 0 <= finding.start <= finding.end <= len(text), finding
        if finding.source in ("guide", "author"):
            assert finding.cite and finding.url and finding.url.startswith("https://"), finding
        else:
            assert finding.source == "heuristic" and finding.cite is None, finding
    return findings


def only(findings, rule):
    return [finding for finding in findings if finding.rule == rule]


def silent(model_id: str, prompt: str, rule: str, categories=None, **fields) -> bool:
    """`rule` has nothing to say about this prompt on this checkpoint."""
    return only(run(model_id, prompt, categories, **fields), rule) == []


def spanned(prompt: str, finding) -> str:
    return prompt[finding.start:finding.end]


CATEGORIES = {
    "smile": "general",
    "honkai: star rail": "copyright",
    "firefly": "character",
    "outdoors": "general",
}


OVER_CAP = "masterpiece, " + ", ".join(f"t{i}" for i in range(300)) + ", 1girl"
