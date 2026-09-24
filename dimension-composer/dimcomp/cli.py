"""dimcomp CLI: config | fit | run | batch."""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml

from .config import parse_set, resolve
from .geometry.fit import Evidence
from .pipeline import evidence_path, fit_path, font_report, load_context, write_layout

app = typer.Typer(add_completion=False, no_args_is_help=True)

STYLE = typer.Option("igloo_default", "--style", help="styles/<name>.yaml")
SET = typer.Option([], "--set", help="session override, e.g. --set search.samples=100 (repeatable)")


def _cfg(spec: Path, style: str, sets: list[str]):
    return resolve(spec, style, parse_set(sets))


@app.command()
def config(spec: Path, style: str = STYLE, set_: list[str] = SET):
    """Print the fully resolved config (style <- profile <- sku <- session)."""
    typer.echo(yaml.safe_dump(_cfg(spec, style, set_).dump(), sort_keys=False, allow_unicode=True))


@app.command()
def fit(spec: Path, style: str = STYLE, set_: list[str] = SET,
        evidence: Path = typer.Option(None, help="lines/corners JSON (default fits/<sku>_evidence.json)"),
        out: Path = typer.Option(Path("out"), help="output root")):
    """Fit the box pose, write fits/<sku>.json and a debug overlay."""
    cfg = _cfg(spec, style, set_)
    if evidence is not None:
        target = evidence_path(cfg)
        if evidence.resolve() != target.resolve():
            Evidence.load(evidence)  # validate
            target.write_text(evidence.read_text())
    fp = fit_path(cfg)
    if fp.exists():
        fp.unlink()
    ctx = load_context(cfg, refit=True, debug_dir=out / cfg.spec.sku)
    typer.echo(json.dumps(ctx.fit.qa(), indent=2, default=float))
    typer.echo(f"debug: {out / cfg.spec.sku / 'debug_fit.png'}")
    if ctx.fit.status != "ok":
        raise typer.Exit(2)


def run_one(spec: Path, style: str, sets: list[str], out: Path, refit: bool, png: bool) -> dict:
    from .layout.search import search

    t0 = time.perf_counter()
    cfg = _cfg(spec, style, sets)
    out_dir = out / cfg.spec.sku
    ctx = load_context(cfg, refit=refit, debug_dir=out_dir)
    t_fit = time.perf_counter()
    top, stats = search(ctx)
    t_search = time.perf_counter()
    cands = []
    for i, c in enumerate(top, 1):
        files = write_layout(ctx, c.layout, out_dir, f"candidate_{i:02d}", png=png)
        cands.append({"rank": i, "cost": round(c.cost, 5), "params": c.params.as_dict(),
                      "breakdown": c.breakdown, "files": files})
    fonts = font_report(ctx)
    report = {
        "sku": cfg.spec.sku, "photo": vars(ctx.photo_info), "fonts": fonts,
        "font_substitutions": sorted(k for k, v in fonts.items() if v["substituted"]),
        "fit": ctx.fit.qa(), "search": stats,
        "timing_s": {"fit_and_load": round(t_fit - t0, 2), "search": round(t_search - t_fit, 2),
                     "render": round(time.perf_counter() - t_search, 2)},
        "candidates": cands, "resolved_config": cfg.dump(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float, ensure_ascii=False))
    return report


@app.command()
def run(spec: Path, style: str = STYLE, set_: list[str] = SET, out: Path = typer.Option(Path("out")),
        refit: bool = typer.Option(False, help="ignore cached fits/<sku>.json"),
        png: bool = typer.Option(True, help="also render PNGs")):
    """Fit (or reuse fit), search layouts, write top-N candidates + report.json."""
    r = run_one(spec, style, set_, out, refit, png)
    if r["font_substitutions"]:
        typer.echo(f"WARNING stand-in fonts for {r['font_substitutions']}: add licensed files to fonts/", err=True)
    for n in r["photo"]["notes"]:
        typer.echo(f"photo: {n}", err=True)
    if r["fit"]["status"] != "ok":
        typer.echo(f"WARNING fit status {r['fit']['status']}: {r['fit']['notes']}", err=True)
    for c in r["candidates"]:
        hard = sum(v["weighted"] for k, v in c["breakdown"].items() if k.startswith("hard_"))
        typer.echo(f"#{c['rank']} cost={c['cost']:.4f} hard={hard:.4f} side={c['params']['height_side']} "
                   f"offset={c['params']['offset_ratio']:.3f} -> {out / r['sku'] / c['files'].get('png', c['files']['svg'])}")
    typer.echo(f"timing {r['timing_s']}  evaluated {r['search']['evaluated']}")


@app.command()
def batch(specs_dir: Path, style: str = STYLE, set_: list[str] = SET, out: Path = typer.Option(Path("out")),
          png: bool = typer.Option(True)):
    """Run every spec in a folder; failures are reported, not fatal."""
    rows = []
    for spec in sorted(specs_dir.glob("*.json")):
        try:
            r = run_one(spec, style, set_, out, False, png)
            rows.append({"sku": r["sku"], "fit": r["fit"]["status"], "best_cost": r["candidates"][0]["cost"]})
        except Exception as e:  # noqa: BLE001 - batch keeps going
            rows.append({"spec": spec.name, "error": f"{type(e).__name__}: {e}"})
    typer.echo(json.dumps(rows, indent=1))


if __name__ == "__main__":
    app()
