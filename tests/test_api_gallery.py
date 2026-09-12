"""The gallery's own routes: Delete unstarred, the inverse of the per-card delete."""
from __future__ import annotations

from tests.api_helpers import run_grid


def _star(client, records):
    for record in records:
        client.post(f"/api/images/{record['id']}/favorite?value=true")


def test_delete_unstarred_keeps_the_stars_and_names_the_rest(client, fake, store):
    # Hundreds made, a few starred: the gesture is "keep these", not "not
    # this one" two hundred times. The preview carries the confirmation's
    # numbers and a sample of what goes; the delete is pinned to that count.
    _, events = run_grid(client, steps_axis="6,8,10", cfg_axis="0,1", seed=5)
    images = sorted(events[-1]["images"], key=lambda r: r["id"])
    _star(client, images[:2])

    preview = client.get("/api/gallery/unstarred").json()
    assert (preview["records"], preview["favorites"], preview["unstarred"]) == (6, 2, 4)
    assert preview["prompts"] == ["a lighthouse in fog"]

    response = client.delete("/api/gallery/unstarred?expected=4")
    assert response.status_code == 200, response.text
    assert sorted(response.json()["deleted"]) == [r["id"] for r in images[2:]]
    assert response.json()["files"] == 4
    page = client.get("/api/gallery").json()
    assert [r["id"] for r in page["items"]] == [images[1]["id"], images[0]["id"]]
    outputs = store.settings.outputs_dir
    assert (outputs / images[0]["filename"]).exists()
    assert not (outputs / images[5]["filename"]).exists()


def test_delete_unstarred_is_scoped_by_the_search_box(client, fake):
    first, _ = run_grid(client, steps_axis="6,8", seed=1)
    second, events = run_grid(client, steps_axis="6,8", seed=2)
    _star(client, events[-1]["images"][:1])

    preview = client.get(f"/api/gallery/unstarred?search=job:{second}").json()
    assert (preview["unstarred"], preview["favorites"], preview["jobs"]) == (1, 1, [second])
    client.delete(f"/api/gallery/unstarred?search=job:{second}&expected=1")
    # The other run is untouched: an untriaged night is not the one being judged.
    assert client.get(f"/api/gallery?search=job:{first}").json()["total"] == 2
    assert client.get(f"/api/gallery?search=job:{second}").json()["total"] == 1


def test_delete_unstarred_refuses_a_changed_selection(client, fake):
    # The dialog said 2; a sweep landed two more while it was open. The count
    # is checked under the write lock and nothing goes.
    run_grid(client, steps_axis="6,8", seed=1)
    run_grid(client, steps_axis="6,8", seed=2)
    response = client.delete("/api/gallery/unstarred?expected=2")
    assert response.status_code == 409
    assert "4 unstarred images match now, not the 2 confirmed" in response.json()["detail"]
    assert client.get("/api/gallery").json()["total"] == 4
    # `expected` is not optional: the gate is the contract.
    assert client.delete("/api/gallery/unstarred").status_code == 422


def test_delete_unstarred_refuses_free_text(client, fake):
    # The gallery matched "lighthouse" through FTS and its LIKE fallback; a
    # destructive scope is not resolved through either.
    run_grid(client, steps_axis="6,8", seed=1)
    for method in ("get", "delete"):
        response = getattr(client, method)("/api/gallery/unstarred?search=lighthouse&expected=2")
        assert response.status_code == 422
        assert "free text" in response.json()["detail"]
    assert client.get("/api/gallery").json()["total"] == 2
