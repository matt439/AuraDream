"""Cold catalogue details from the app's existing sourced model guides."""
from .launch import CHECKPOINTS, FORMATS
from .model_defaults import defaults_for, hints_for


def details_for(model_id: str) -> dict[str, str]:
    checkpoint = next((
        entry for entry in CHECKPOINTS
        if model_id == entry.repo or any(model_id == entry.lite_repo(fmt) for fmt, _ in FORMATS)
    ), None)
    if checkpoint is None:
        return {
            "description": "A custom checkpoint supplied at launch.",
            "access": "Check its model card for access requirements.",
            "sampling": "The app uses its matching profile, or general defaults where none is known.",
            "prompt": "", "size": "", "url": "",
        }
    profile = defaults_for(model_id)
    hints = hints_for(model_id)
    capabilities = ["text-to-image"]
    if profile["reference_supported"]:
        capabilities.append("reference images")
    if profile["refine_supported"]:
        capabilities.append("Refine")
    source_id = checkpoint.repo.split("@", 1)[0]
    return {
        "description": "App profile: " + ", ".join(capabilities) + ".",
        "access": checkpoint.gate if model_id == checkpoint.repo else "Nunchaku Lite build; Linux/WSL only.",
        "sampling": f"App defaults: {profile['steps']} steps, CFG {profile['guidance_scale']:g}.",
        "prompt": hints["prompt"],
        "size": hints["size"],
        "url": f"https://huggingface.co/{source_id}",
    }
