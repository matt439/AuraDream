"""An install made under the old name, Local Diffusion, keeps working as it is."""
from __future__ import annotations

import huggingface_hub.constants
import pytest

from backend import critic_runtime, legacy_names, refine, snippets, startup_preferences as preferences
from backend.backends import single_file
from backend.schemas_snippets import SnippetFields


@pytest.fixture
def hf_home(tmp_path, monkeypatch):
    home = tmp_path / "hf"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HOME", str(home))
    return home


def test_new_name_wins_and_old_one_is_used_only_while_alone(tmp_path):
    new, legacy = tmp_path / "auradream.db", tmp_path / "local-diffusion.db"
    assert legacy_names.default_db_path(tmp_path) == new
    legacy.write_bytes(b"")
    assert legacy_names.default_db_path(tmp_path) == legacy
    new.write_bytes(b"")
    assert legacy_names.default_db_path(tmp_path) == new
    assert legacy.exists(), "nothing is renamed or removed"


def test_sidecar_follows_whichever_name_holds_it(tmp_path):
    database = tmp_path / "gallery.db"
    assert legacy_names.sidecar(database, "settings.json") == tmp_path / "auradream.settings.json"
    (tmp_path / "local-diffusion.settings.json").write_text("{}")
    assert legacy_names.sidecar(database, "settings.json") == tmp_path / "local-diffusion.settings.json"
    name = f"backup-policy-{database.name}.json"
    assert legacy_names.sidecar(database, name) == tmp_path / f"auradream.{name}"


def test_saved_startup_choice_under_the_old_name_is_read_and_rewritten_in_place(tmp_path, monkeypatch):
    legacy_db = tmp_path / "local-diffusion.db"
    legacy_db.write_bytes(b"")
    choice = preferences.StartupPreferences("krea/Krea-2-Turbo", "int8", "sequential", True)
    old = tmp_path / "local-diffusion.settings.json"
    # Saved under the new name, then given the old one an install has.
    preferences.save(legacy_db, choice)
    (tmp_path / "auradream.settings.json").rename(old)
    assert preferences.path_for(legacy_db) == old
    assert preferences.load_result(legacy_db).preferences == choice

    # The launcher's default database is the old gallery while it is alone,
    # and its saved choice reaches the environment.
    monkeypatch.setattr(preferences, "_ROOT", tmp_path)
    env: dict[str, str] = {}
    assert preferences.apply_saved_environment(env=env) == choice
    assert env["AURADREAM_MODEL_ID"] == choice.model_id

    other = preferences.StartupPreferences("krea/Krea-2-Turbo", "fp8", "model", False)
    preferences.save(legacy_db, other)
    assert preferences.load_result(legacy_db).preferences == other
    assert not (tmp_path / "auradream.settings.json").exists()


def test_snippets_under_the_old_name_are_kept(tmp_path):
    database = tmp_path / "local-diffusion.db"
    first = snippets.change(database, SnippetFields(name="n", text="t", target="prompt"))[0]
    (tmp_path / "auradream.snippets.json").rename(tmp_path / "local-diffusion.snippets.json")
    assert snippets.read(database) == [first]
    snippets.change(database, SnippetFields(name="m", text="u", target="negative"))
    assert len(snippets.read(database)) == 2
    assert not (tmp_path / "auradream.snippets.json").exists()


def test_a_tree_converted_under_the_old_folder_is_a_cache_hit(hf_home, monkeypatch):
    model_id = single_file.ILLUSTRIOUS_ID
    monkeypatch.setattr(single_file, "_convert", lambda *_: pytest.fail("an existing conversion was redone"))
    target, marker, recipe = single_file._target(model_id)
    assert target.parent == hf_home / "auradream" / "converted"
    old = hf_home / "local-diffusion" / "converted" / target.name
    old.mkdir(parents=True)
    (old / "model_index.json").write_text("{}")
    (old / "local-diffusion-conversion.json").write_text(recipe, encoding="utf-8")
    assert single_file.converted_tree(model_id) == str(old)
    assert single_file.resolve_model(model_id, "fp8") == str(old)
    assert not (hf_home / "auradream").exists()


def test_critic_and_upscaler_downloads_under_the_old_folder_are_found(hf_home):
    assert critic_runtime.install_root() == hf_home / "auradream" / "llama.cpp"
    assert critic_runtime._INSTALL_MARKER == "local-diffusion-install.json"
    (hf_home / "local-diffusion" / "llama.cpp").mkdir(parents=True)
    (hf_home / "local-diffusion" / "upscalers").mkdir(parents=True)
    assert critic_runtime.install_root() == hf_home / "local-diffusion" / "llama.cpp"
    assert refine._anime6b_path().is_relative_to(hf_home / "local-diffusion" / "upscalers")
