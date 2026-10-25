# Contributing

Local Diffusion is a personal project, built around one workstation (an
RTX 5080 on Windows 11, with WSL 2 for Nunchaku). Issues and pull requests are
welcome, but changes are judged by whether they fit that use, and GPU
behaviour can only be confirmed on the maintainer's card.

Before changing an area, read:

- [AGENTS.md](AGENTS.md): the short version of the rules that break things
  silently, the layout, and how to run the tests and browser smokes. It is
  written for humans and coding agents alike.
- That area's part of [docs/rules.md](docs/rules.md) and
  [docs/design-notes.md](docs/design-notes.md): the full rules, and why.

A pull request should:

- pass `uv run --group dev pytest` and `uv run --group dev ruff check .`
  (CI runs both; no GPU or weights needed);
- for a `web/` change, pass the relevant smokes
  (`uv run python tools/run_smoke.py <smoke>`), since there is no JS test
  suite;
- not claim GPU timings, memory or image quality it has not measured; those
  belong in [docs/verified.md](docs/verified.md) once measured on a card;
- keep prompting guidance sourced from each checkpoint's authors, quoted
  verbatim.

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE).
