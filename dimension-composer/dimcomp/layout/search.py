"""Latin-hypercube sample + coordinate-descent refine over layout params."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import HiddenEdgeError
from .cost import score
from .scene import Context, Layout, Params, build_layout

CONT = ["offset_ratio", "t_L", "t_W", "t_H", "group_scale", "nudge_x", "nudge_y"]


@dataclass
class Candidate:
    params: Params
    cost: float
    breakdown: dict
    layout: Layout


class Space:
    def __init__(self, ctx: Context):
        st = ctx.cfg.style
        r, pinned = st.search.ranges, dict(st.search.pinned)
        self.bounds = {
            "offset_ratio": r.offset_ratio, "t_L": r.label_t, "t_W": r.label_t, "t_H": r.label_t,
            "group_scale": r.group_scale, "nudge_x": r.group_nudge, "nudge_y": r.group_nudge,
        }
        allowed = {c.split("_")[0] for c in ctx.cfg.profile.height_edge_candidates}
        self.sides = [s for s in r.height_side if s in allowed] or sorted(allowed)
        ext = st.annotations.extension_lines
        self.ext = r.extension_lines if ext == "auto" else [ext == "on"]
        # pins: any param name above, plus height_side / extension_lines / label_t
        self.pinned = pinned
        for k in ("L", "W", "H"):
            if "label_t" in pinned:
                pinned.setdefault(f"t_{k}", pinned["label_t"])
        if "group_nudge" in pinned:
            pinned.setdefault("nudge_x", pinned["group_nudge"][0])
            pinned.setdefault("nudge_y", pinned["group_nudge"][1])
        if "height_side" in pinned:
            self.sides = [pinned["height_side"]]
        if "extension_lines" in pinned:
            self.ext = [bool(pinned["extension_lines"])]
        self.free = [k for k in CONT if k not in pinned]
        self.n_evals = 0

    def params(self, v: dict, side: str, ext: bool) -> Params:
        v = {**v, **{k: self.pinned[k] for k in CONT if k in self.pinned}}
        return Params(offset_ratio=v["offset_ratio"],
                      label_t={"L": v["t_L"], "W": v["t_W"], "H": v["t_H"]},
                      height_side=side, group_scale=v["group_scale"],
                      nudge=(v["nudge_x"], v["nudge_y"]), extension_lines=ext)

    def clip(self, k, x):
        lo, hi = self.bounds[k]
        return float(np.clip(x, lo, hi))

    def unit(self, v: dict) -> np.ndarray:
        return np.array([(v[k] - self.bounds[k][0]) / max(self.bounds[k][1] - self.bounds[k][0], 1e-9) for k in CONT])


def lhs(n: int, keys: list[str], bounds: dict, rng) -> list[dict]:
    out = [dict() for _ in range(n)]
    for k in keys:
        lo, hi = bounds[k]
        u = (rng.permutation(n) + rng.random(n)) / n
        for i in range(n):
            out[i][k] = lo + u[i] * (hi - lo)
    return out


def evaluate(ctx: Context, space: Space, v: dict, side: str, ext: bool) -> Candidate | None:
    p = space.params(v, side, ext)
    space.n_evals += 1
    try:
        lay = build_layout(ctx, p)
    except HiddenEdgeError:
        raise
    except ValueError:
        return None
    cost, br = score(ctx, lay)
    return Candidate(p, cost, br, lay)


def _vec(c: Candidate) -> dict:
    p = c.params
    return {"offset_ratio": p.offset_ratio, "t_L": p.label_t["L"], "t_W": p.label_t["W"], "t_H": p.label_t["H"],
            "group_scale": p.group_scale, "nudge_x": p.nudge[0], "nudge_y": p.nudge[1]}


def search(ctx: Context) -> tuple[list[Candidate], dict]:
    st = ctx.cfg.style.search
    space = Space(ctx)
    rng = np.random.default_rng(st.seed)
    defaults = {k: float(np.mean(space.bounds[k])) for k in CONT}
    samples = lhs(st.samples, space.free, space.bounds, rng)
    combos = [(s, e) for s in space.sides for e in space.ext]
    pool: list[Candidate] = []
    for i, smp in enumerate(samples):
        side, ext = combos[i % len(combos)]
        c = evaluate(ctx, space, {**defaults, **smp}, side, ext)
        if c:
            pool.append(c)
    n_sampled = len(pool)

    pool.sort(key=lambda c: c.cost)
    refined = []
    for c in pool[: st.refine_top]:
        best, v = c, _vec(c)
        steps = {k: 0.15 * (space.bounds[k][1] - space.bounds[k][0]) for k in space.free}
        for _ in range(st.refine_rounds):
            improved = False
            for k in space.free:
                for sgn in (1, -1):
                    trial = dict(v)
                    trial[k] = space.clip(k, v[k] + sgn * steps[k])
                    if trial[k] == v[k]:
                        continue
                    cand = evaluate(ctx, space, trial, best.params.height_side, best.params.extension_lines)
                    if cand and cand.cost < best.cost:
                        best, v, improved = cand, trial, True
            for side, ext in combos:
                if (side, ext) != (best.params.height_side, best.params.extension_lines):
                    cand = evaluate(ctx, space, v, side, ext)
                    if cand and cand.cost < best.cost:
                        best, improved = cand, True
            if not improved:
                steps = {k: s / 2 for k, s in steps.items()}
        refined.append(best)

    # a candidate with any hard violation never outranks a clean one
    hard_keys = [k for k in pool[0].breakdown if k.startswith("hard_")] if pool else []
    def is_clean(c):
        return all(c.breakdown[k]["raw"] == 0 for k in hard_keys)
    everything = sorted(pool + refined, key=lambda c: (not is_clean(c), c.cost))
    top: list[Candidate] = []
    for c in everything:
        u = space.unit(_vec(c))
        distinct = all(
            np.linalg.norm(u - space.unit(_vec(o))) / np.sqrt(len(CONT)) > st.min_param_distance
            or (c.params.height_side, c.params.extension_lines) != (o.params.height_side, o.params.extension_lines)
            for o in top)
        if distinct:
            top.append(c)
        if len(top) == st.top_n:
            break
    stats = {"evaluated": space.n_evals, "sampled": n_sampled, "refined": len(refined),
             "free_params": space.free, "height_sides": space.sides, "extension_options": space.ext}
    return top, stats
