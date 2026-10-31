"""The command line's --via-server client: the same flags, queued on the
configured running server through its public API, and waited on.

Split from `generate.py`, whose `main()` hands over here once the
arguments are read; it shares no state with the local run loop. Each call
is one HTTP request, never retried: a lost submission may already be queued.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

from backend.generate_args import _grid_flags

_SERVER_POLL_SECONDS = 1.0


class _ServerError(Exception):
    def __init__(self, detail: str, status: int = 0) -> None:
        super().__init__(detail)
        self.status = status


def _server_json(base: str, path: str, *, body=None, data=None, headers=None, timeout=30.0) -> dict:
    """One HTTP call, never retried: a lost submission may already be queued."""
    import json
    import urllib.error
    import urllib.request

    outgoing = dict(headers or {})
    if body is not None:
        data = json.dumps(body, ensure_ascii=True, allow_nan=False).encode("utf-8")
        outgoing["Content-Type"] = "application/json"
    request = urllib.request.Request(base + path, data=data, headers=outgoing)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            with exc:
                detail = json.load(exc).get("detail", str(exc))
        except (OSError, ValueError, AttributeError):
            detail = str(exc)
        if isinstance(detail, list):
            detail = "; ".join(
                f"{'.'.join(str(part) for part in item.get('loc', [])[1:])}: {item.get('msg', 'invalid request')}"
                for item in detail if isinstance(item, dict)
            )
        raise _ServerError(str(detail), exc.code) from None
    except (OSError, ValueError) as exc:
        raise _ServerError(f"no usable answer from {base}: {exc}") from None
    if not isinstance(answer, dict):
        raise _ServerError(f"{base}{path} did not answer with a JSON object")
    return answer


def _via_server(args, parser, source, carried, mask_image, width, height, sized, outpaint_sides, *,
                check_carried_source, mask_from_record, resolved_execution) -> int:
    """Script the public API; all model defaults and admission stay server-owned.

    The three helpers it shares with the local run are `generate`'s
    `_check_carried_source`, `_mask_from_record` and `_resolved_cli_execution`,
    passed in by `main()` rather than imported back: `generate` imports
    this module, and run as `python -m backend.generate` it is `__main__`,
    so an import would be circular or load a second copy. Passed in, a
    patch of `generate`'s name reaches both paths.
    """
    import io
    import os

    if args.ignore_server:
        parser.error("--via-server and --ignore-server are alternatives")
    if args.dry_run:
        parser.error("--via-server and --dry-run are alternatives: a dry run makes no network requests")
    if carried is not None:
        parser.error(
            "--via-server cannot carry a PNG's grid/comparison labels through the public generation API; "
            "run without --via-server, change the prompt, or define new variables with --grid-prompt"
        )
    prompt_axis, gridding = _grid_flags(args, parser)
    host = os.getenv("AURADREAM_HOST", "127.0.0.1")
    host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = int(os.getenv("AURADREAM_PORT", "7860"))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        parser.error("AURADREAM_PORT must be a port from 1 to 65535")
    base = f"http://{host}:{port}"
    headers = {}
    job_id = None
    try:
        status = _server_json(base, "/api/status", timeout=5.0)
        if not all(isinstance(status.get(key), str) for key in ("model_id", "quantization", "offload")):
            raise _ServerError("the server's status has no checkpoint/loading settings")
        if status.get("switching"):
            raise _ServerError("the checkpoint is switching; wait for the switch to finish", 409)
        from backend.execution import configured_loader_unavailable_reason

        reason = configured_loader_unavailable_reason(SimpleNamespace(
            loader_execution=status.get("execution"),
            loader_execution_error=(True if status.get("execution_available") is False and not status.get("execution") else None),
        ))
        if reason is not None:
            raise _ServerError(reason, 422)
        try:
            effective_execution = resolved_execution(args, status.get("execution"), source)
        except ValueError as exc:
            raise _ServerError(str(exc), 422) from exc
        for value, key, flag, env in (
            (args.model, "model_id", "--model", "AURADREAM_MODEL_ID"),
            (args.quant, "quantization", "--quant", "AURADREAM_QUANT"),
            (args.offload, "offload", "--offload", "AURADREAM_OFFLOAD"),
        ):
            wanted = value if value is not None else os.getenv(env)
            if wanted is not None and wanted != status[key]:
                raise _ServerError(
                    f"{flag} asks for {wanted}, but the server uses {status[key]}; "
                    "select those settings in Models first, or use matching flags/environment", 422,
                )
        revision = status.get("model_revision")
        if not isinstance(revision, int):
            raise _ServerError("the server's status has no model revision; update the server before using --via-server")
        headers["X-AuraDream-Model-Revision"] = str(revision)
        made_with = source.get("quantization") if source is not None else None
        if made_with and made_with != status["quantization"]:
            print(f"note: the PNG was made with quant={made_with}; the server uses {status['quantization']}, so the pixels will differ", file=sys.stderr)
        records = {}

        def record(image_id, include_binned=False):
            if image_id not in records:
                records[image_id] = _server_json(base, f"/api/images/{image_id}")
            return SimpleNamespace(**records[image_id])

        # Reuse the CLI's provenance guard, against server records rather
        # than a similarly numbered image in this process's local gallery.
        remote = SimpleNamespace(storage=SimpleNamespace(get_image=record),
                                 settings=SimpleNamespace(db_path=SimpleNamespace(exists=lambda: True)))
        dropped = check_carried_source(args, remote, parser)
        if dropped:
            outpaint_sides = None
        from backend import grid, refine

        references = grid.parse_axis(args.reference, int, label="reference")
        if source is not None and not args.reference_given:
            known = source.get("reference_sources") or []
            for at, image_id in enumerate(references):
                saved = known[at] if at < len(known) and isinstance(known[at], dict) else {}
                held = record(image_id)
                if not saved or (getattr(held, "content_hash", None), held.width, held.height) != (
                    saved.get("hash"), saved.get("width"), saved.get("height")
                ):
                    raise _ServerError(
                        f"--from-image: reference image {image_id} is not verified as the PNG's reference "
                        "in the server gallery; name server images explicitly with --reference", 422,
                    )
        inpainting = args.mask is not None or args.mask_hash is not None
        if args.upscaler not in (None, "lanczos"):
            reason = refine.upscaler_reason(SimpleNamespace(
                upscaler=args.upscaler, refine=args.refine, mask=args.mask or args.mask_hash,
                outpaint=outpaint_sides,
            ), status["model_id"])
            if reason:
                raise _ServerError(reason, 422)
            held = record(args.refine)
            try:
                refine.check_upscaler_size(held.width, held.height)
            except ValueError as exc:
                raise _ServerError(str(exc), 422) from exc
        body = {"prompt": args.prompt, "negative_prompt": args.negative,
                "width": width, "height": height, "num_images": args.batch, "runs": args.runs,
                "reference": references, "lora": args.lora, "lora_scale": args.lora_scale}
        body["execution"] = effective_execution.model_dump(exclude_none=True) if effective_execution is not None else None
        if args.crop_anchor is not None:
            body["crop_anchor"] = args.crop_anchor
        if args.upscaler not in (None, "lanczos"):
            body["upscaler"] = args.upscaler
        if args.then_refine is not None:
            body["then_refine"] = args.then_refine
        for name, value in (("steps", args.steps), ("guidance_scale", args.cfg), ("true_cfg", args.true_cfg),
                            ("seed", args.seed), ("strength", args.strength)):
            if value is not None:
                body[name] = value
        if args.no_mu or args.mu is not None:
            body["mu"] = None if args.no_mu else args.mu
        if args.refine is not None:
            held = record(args.refine)
            body["refine"] = args.refine
            if outpaint_sides is not None:
                presets = _server_json(base, f"/api/images/{args.refine}/outpaint-presets")
                if presets.get("off_grid"):
                    raise _ServerError(presets["off_grid"], 422)
                if isinstance(outpaint_sides, str):
                    chosen = next((item for item in presets["presets"] if item["name"] == outpaint_sides), None)
                    if chosen is None:
                        raise _ServerError(f"no server outpaint preset {outpaint_sides!r}", 422)
                    if chosen.get("held"):
                        raise _ServerError(chosen["held"], 422)
                    outpaint_sides = tuple(chosen[key] for key in ("left", "top", "right", "bottom"))
                left, top, right, bottom = outpaint_sides
                body.update(width=held.width + left + right, height=held.height + top + bottom)
                body["outpaint"] = {"left": left, "top": top, "right": right, "bottom": bottom,
                                    "fill": args.outpaint_fill or presets["fill"],
                                    "band": args.band if args.band is not None else presets["band"]}
                if args.carried_outpaint_mask is not None:
                    from backend.schemas import Outpaint
                    from backend.storage import outpaint_mask_hash

                    if outpaint_mask_hash(body["width"], body["height"], Outpaint(**body["outpaint"])) != args.carried_outpaint_mask:
                        raise _ServerError("--from-image: this build builds an outpaint's mask differently from the one that made this image", 422)
            elif inpainting:
                if not sized:
                    body.update(width=held.width, height=held.height)
            elif args.scale is not None or not sized:
                body["width"], body["height"] = refine.scaled(held.width, held.height, args.scale or refine.DEFAULT_SCALE)
        if inpainting:
            if mask_image is None and args.mask_hash is not None:
                mask_image = mask_from_record(args.from_image, args.mask_hash)
            if mask_image is not None:
                data = io.BytesIO()
                mask_image.save(data, format="PNG")
                uploaded = _server_json(base, f"/api/masks?source={args.refine}", data=data.getvalue(),
                                        headers={**headers, "Content-Type": "image/png"})
                if args.mask_hash is not None and uploaded["hash"] != args.mask_hash:
                    raise _ServerError("the server stored a different mask from the PNG's record", 422)
                body["mask"] = uploaded["hash"]
            else:
                body["mask"] = args.mask_hash
        path = "/api/generate"
        if gridding or args.resume or args.budget is not None:
            path = "/api/grid"
            body.update(runs=1, prompt_axis=prompt_axis, prompt_mode=args.grid_prompt_mode,
                        size_axis=args.grid_size,
                        steps_axis=args.grid_steps, cfg_axis=args.grid_cfg,
                        mu_axis=args.grid_mu, lora_scale_axis=args.grid_lora_scale, strength_axis=args.grid_strength,
                        seeds=args.grid_seeds if gridding else args.runs, seed_axis=args.seed_list,
                        resume=args.resume, budget_minutes=args.budget)
        if not args.quiet:
            print(f"server   : {base}\nmodel    : {status['model_id']} (quant={status['quantization']}, offload={status['offload']})")
        try:
            accepted = _server_json(base, path, body=body, headers=headers)
        except _ServerError as exc:
            if args.resume and exc.status == 409 and str(exc) == "every cell is already rendered":
                print("Nothing to do: every cell is already in the gallery.", file=sys.stderr)
                return 0
            if not exc.status:
                print(f"Submission was not confirmed; check {base}/api/jobs before retrying.", file=sys.stderr)
            raise
        job_id = accepted["job_id"]
        print(f"Queued job {job_id}; run {accepted.get('run_id')}; seed {accepted.get('seed')}.", file=sys.stderr)
        seen = set()
        cancelling = False
        while True:
            try:
                job = _server_json(base, f"/api/jobs/{job_id}", timeout=10.0)
                for image in job.get("images", []):
                    if image["id"] not in seen:
                        seen.add(image["id"])
                        print(f"{base}{image['url']}  seed {image['seed']}", flush=True)
                if job["status"] in {"done", "error", "cancelled"}:
                    if job.get("error"):
                        print(job["error"], file=sys.stderr)
                    if job.get("stopped_by"):
                        print(f"Job {job_id} stopped by {job['stopped_by']}.", file=sys.stderr)
                    return 1 if job["status"] == "error" or job.get("failures") else 130 if job["status"] == "cancelled" else 0
                time.sleep(_SERVER_POLL_SECONDS)
            except KeyboardInterrupt:
                if cancelling:
                    raise
                cancelling = True
                _server_json(base, f"/api/jobs/{job_id}/cancel", body={})
                print(f"Cancelling job {job_id}; waiting for its current step (Ctrl-C again stops waiting).", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"Stopped waiting; check {base}/api/jobs" + (f"/{job_id}" if job_id else " before submitting again") + ".", file=sys.stderr)
        return 130
    except _ServerError as exc:
        print(str(exc) + (f" (job {job_id}; it may still be running)" if job_id else ""), file=sys.stderr)
        return 2 if job_id is None and 400 <= exc.status < 500 else 1
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        print(f"--via-server: {exc}", file=sys.stderr)
        return 2 if job_id is None else 1
