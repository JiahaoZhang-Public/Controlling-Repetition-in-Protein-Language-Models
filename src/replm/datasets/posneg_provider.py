# src/replm/datasets/posneg_provider.py
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from replm.utils.io import read_fasta as _read_fasta_simple  # Update import path if needed.

from .base import DatasetProvider

# ===================== Configuration ===================== #


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds applied before balancing the positive/negative pools."""

    ent_hi: float = 0.90
    ent_lo: float = 0.80
    plddt_hi: float = 85.0
    length_bins: tuple[int, ...] = (50, 129, 257, 513, 1025)
    plddt_bins: tuple[float, ...] = (70.0, 80.0, 90.0, 101.0)
    min_len: int = 50
    max_len: int = 1024
    max_per_bucket: int | None = None
    safety_key: str | None = None  # metrics flag that must equal 1
    truth_key: str | None = None  # metrics flag that must equal 1


@dataclass(frozen=True)
class OptimizeConfig:
    """Configuration for balancing/selection heuristics."""

    method: str = "random"  # 'simple' | 'random' | 'composite' | 'pareto'
    target_per_side: int | None = None
    rep_key: str = "rep_metric"
    bio_key: str = "utility_metric"
    bins_in_bucket: int = 6
    max_iter: int = 3
    alpha: float = 1.0
    beta: float = 1.0
    random_seed: int = 42


@dataclass(frozen=True)
class MetricKeyConfig:
    """
    Column aliases for metrics stored in the CSV. Every field can be
    overridden with a single string or a list of strings via config.
    """

    entropy: tuple[str, ...] = ("entropy_norm", "H_norm")
    repetition: tuple[str, ...] = "entropy_norm"
    utility: tuple[str, ...] = "ptm"
    plddt: tuple[str, ...] = ("plddt_mean_0_100", "plddt", "plddt_mean_01")


# ===================== I/O helpers ===================== #


def _first_token(h: str) -> str:
    h = h.strip()
    return h.split()[0] if h else ""


def _read_fasta_with_ids(path: Path) -> list[tuple[str, str]]:
    """Read a FASTA file and ensure every entry has a usable seq_id."""
    out: list[tuple[str, str]] = []
    cur_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None and chunks:
                    out.append((cur_id, "".join(chunks)))
                    chunks.clear()
                cur_id = _first_token(line[1:])
            else:
                chunks.append(line)
    if cur_id is not None and chunks:
        out.append((cur_id, "".join(chunks)))

    if not out:
        # Fallback when headers are missing: reuse the simple FASTA reader.
        seqs = _read_fasta_simple(path)
        return [(f"{path.stem}_{i:06d}", s) for i, s in enumerate(seqs) if s]

    # Normalize blank IDs.
    return [(sid if sid else f"{path.stem}_{i:06d}", s) for i, (sid, s) in enumerate(out) if s]


def _load_metrics(path: Path) -> dict[str, dict]:
    """Load metrics from jsonl/csv and return a seq_id-indexed mapping."""
    if path.suffix.lower() == ".jsonl":
        d: dict[str, dict] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = str(obj.get("seq_id"))
                if sid:
                    d[sid] = obj
        return d

    if path.suffix.lower() == ".csv":
        import csv

        d: dict[str, dict] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "seq_id" not in (reader.fieldnames or []):
                raise ValueError("metrics table must contain 'seq_id'")
            for r in reader:
                sid = str(r["seq_id"])
                # Keep seq_id/sequence as strings and coerce everything else to float.
                entry: dict[str, object] = {}
                for k, v in r.items():
                    if k in ("seq_id", "sequence"):
                        entry[k] = v
                    else:
                        try:
                            entry[k] = float(v)
                        except Exception:
                            entry[k] = v
                d[sid] = entry
        return d

    raise ValueError(f"Unsupported metrics format: {path.suffix}")


def _bucketize(val: float, edges: Sequence[float | int]) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] <= val < edges[i + 1]:
            return i
    return None


# ===================== Metric adapters ===================== #


def _get_metric(
    m: dict,
    keys: Sequence[str],
    *,
    transform=None,
    default: float = np.nan,
) -> float:
    """Generic helper for reading metrics with optional transforms."""
    for k in keys:
        if k in m:
            v = m.get(k)
            try:
                v_f = float(v)
            except Exception:
                continue
            if transform is not None:
                try:
                    v_f = float(transform(v_f))
                except Exception:
                    pass
            return v_f
    return float(default)


def _get_entropy(m: dict, keys: Sequence[str] = ("entropy_norm", "H_norm")) -> float:
    # repetition.py exposes 'H_norm'; legacy tables may use 'entropy_norm'.
    return _get_metric(m, keys)


def _get_repetition(
    m: dict,
    keys: Sequence[str] = ("repetition", "entropy_norm", "H_norm"),
) -> float:
    # Primary signal is 'repetition'; entropy-based proxies serve as fallback.
    return _get_metric(m, keys)


def _get_plddt(
    m: dict,
    keys: Sequence[str] = ("plddt_mean_0_100", "plddt", "plddt_mean_01"),
) -> float:
    def to_0_100(x: float) -> float:
        # Convert 0-1 values to the 0-100 scale.
        if 0.0 <= x <= 1.0:
            return x * 100.0
        return x

    return _get_metric(m, keys, transform=to_0_100)


def _get_length(m: dict, seq: str) -> int:
    val = _get_metric(m, ["length"], default=len(seq))
    try:
        return int(val)
    except Exception:
        return len(seq)


# ===================== Dataset implementation ===================== #


class PosNegDataset:
    """
    Build a balanced positive/negative sample set that matches distributions
    over (length_bin, pLDDT_bin) while separating repetition statistics.

    Metrics are expected to be precomputed. This class only reads them using
    helper utilities that normalize field names, defaults, and transforms.
    """

    def __init__(
        self,
        pos_fasta: Path,
        pos_metrics: Path | None,
        neg_fasta: Path,
        neg_metrics: Path | None,
        seed: int = 42,
        filter_cfg: FilterConfig = FilterConfig(),
        *,
        cache_manifest: Path | None = None,
        opt_cfg: OptimizeConfig = OptimizeConfig(),
        metric_keys: MetricKeyConfig = MetricKeyConfig(),
    ) -> None:
        self.seed = seed
        self.filter_cfg = filter_cfg
        self.opt_cfg = opt_cfg
        self.metric_keys = metric_keys
        self.pos_pairs = _read_fasta_with_ids(Path(pos_fasta))
        self.neg_pairs = _read_fasta_with_ids(Path(neg_fasta))
        self.pos_m = _load_metrics(Path(pos_metrics)) if pos_metrics else {}
        self.neg_m = _load_metrics(Path(neg_metrics)) if neg_metrics else {}

        self.items: list[dict] = []
        self.stats: dict = {}
        self._build(cache_manifest)

    # ---------- Filtering positives/negatives ---------- #
    def _filter_side(
        self,
        pairs: list[tuple[str, str]],
        metrics: dict[str, dict],
        mode: str,
    ) -> list[dict]:
        """Apply length/metric filters for either the positive or negative side."""
        out: list[dict] = []
        for sid, seq in pairs:
            m = metrics.get(sid)
            if (m is None) or (not seq):
                # Skip entries without sequences or metrics.
                continue

            L = _get_length(m, seq)
            if L < self.filter_cfg.min_len or L > self.filter_cfg.max_len:
                continue

            ent = _get_entropy(m, self.metric_keys.entropy)
            repetition = _get_repetition(m, self.metric_keys.repetition)
            plddt = _get_plddt(m, self.metric_keys.plddt)
            utility = _get_metric(m, self.metric_keys.utility)
            if not np.isfinite(utility):
                utility = plddt

            # Optional safety/truth constraints.
            if self.filter_cfg.safety_key:
                v = m.get(self.filter_cfg.safety_key)
                if v is not None:
                    try:
                        if int(float(v)) != 1:
                            continue
                    except Exception:
                        continue
            if self.filter_cfg.truth_key:
                v = m.get(self.filter_cfg.truth_key)
                if v is not None:
                    try:
                        if int(float(v)) != 1:
                            continue
                    except Exception:
                        continue

            if not np.isfinite(ent) or not np.isfinite(plddt):
                continue

            record = {
                "seq_id": sid,
                "sequence": seq,
                "length": L,
                "entropy": ent,
                "rep_metric": repetition,
                "utility_metric": utility,
                "plddt": plddt,
                "source": mode,
            }

            if mode == "pos":
                if plddt >= self.filter_cfg.plddt_hi and ent >= self.filter_cfg.ent_hi:
                    out.append(record)
            elif mode == "neg":
                if plddt >= self.filter_cfg.plddt_hi and ent <= self.filter_cfg.ent_lo:
                    out.append(record)
            else:
                raise ValueError(mode)
        return out

    # ---------- Numeric helpers ---------- #

    @staticmethod
    def _robust_scale(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        med = np.median(x)
        q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
        iqr = max(q3 - q1, eps)
        return (x - med) / iqr

    @staticmethod
    def _hist_quota(values: np.ndarray, bins: int, total: int) -> tuple[list[int], np.ndarray]:
        # Use quantiles to derive bucket edges.
        edges = np.quantile(values, q=np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([values.min() - 1e-6, values.max() + 1e-6])
        hist, edges = np.histogram(values, bins=edges)
        prop = hist / max(hist.sum(), 1)
        raw = prop * total
        quota = np.floor(raw).astype(int)
        remain = total - quota.sum()
        frac = raw - quota
        idxs = np.argsort(-frac)
        for i in range(remain):
            quota[idxs[i]] += 1
        return quota.tolist(), edges

    # ---------- Bucket-level selection ---------- #

    def _select_bucket_composite(
        self,
        pos_items: list[dict],
        neg_items: list[dict],
        k_pos: int,
        k_neg: int,
    ) -> tuple[list[dict], list[dict]]:
        """Composite scoring: maximize delta repetition while minimizing delta bio."""
        rep_key = "rep_metric"
        bio_key = "utility_metric"

        pos_df = np.array(
            [(float(r.get(rep_key, np.nan)), float(r.get(bio_key, np.nan))) for r in pos_items],
            dtype=float,
        )
        neg_df = np.array(
            [(float(r.get(rep_key, np.nan)), float(r.get(bio_key, np.nan))) for r in neg_items],
            dtype=float,
        )

        pos_ref_u = np.nanmean(neg_df[:, 1]) if len(neg_df) else np.nanmean(pos_df[:, 1])
        neg_ref_u = np.nanmean(pos_df[:, 1]) if len(pos_df) else np.nanmean(neg_df[:, 1])
        if not np.isfinite(pos_ref_u):
            pos_ref_u = 0.0
        if not np.isfinite(neg_ref_u):
            neg_ref_u = 0.0

        def score_block(arr: np.ndarray, u_ref: float, rep_is_pos_low: bool) -> np.ndarray:
            r = arr[:, 0]
            u = arr[:, 1]
            rs = self._robust_scale(r)
            # Direction: positive bucket prefers low repetition, negative high.
            s_r = -rs if rep_is_pos_low else +rs
            du = np.abs(u - u_ref)
            dus = -self._robust_scale(du)  # Closer to reference utility is better.
            return self.opt_cfg.alpha * s_r + self.opt_cfg.beta * dus

        rep_is_pos_low = True

        def pick_side(items: list[dict], other_u_mean: float, k: int, rep_low: bool) -> list[dict]:
            if k <= 0 or not items:
                return []
            vals = np.array([float(x.get(bio_key, np.nan)) for x in items], dtype=float)
            vals = np.where(
                np.isfinite(vals),
                vals,
                np.nanmean(vals) if np.isfinite(np.nanmean(vals)) else 0.0,
            )
            quota, edges = self._hist_quota(vals, self.opt_cfg.bins_in_bucket, k)

            arr = np.array(
                [(float(x.get(rep_key, np.nan)), float(x.get(bio_key, np.nan))) for x in items],
                dtype=float,
            )
            arr = np.where(np.isfinite(arr), arr, 0.0)
            scores = score_block(arr, other_u_mean, rep_low)

            chosen: list[dict] = []
            for b in range(len(quota)):
                if quota[b] <= 0:
                    continue
                left, right = edges[b], edges[b + 1]
                mask = (vals >= left) & (vals <= right if b == len(quota) - 1 else vals < right)
                idxs = np.flatnonzero(mask)
                if len(idxs) == 0:
                    continue
                order = idxs[np.argsort(-scores[idxs])]
                picked = order[: quota[b]]
                chosen.extend([items[i] for i in picked])

            if len(chosen) < k:
                # Global top-up if buckets could not fill the quota.
                rest_idx = [i for i in range(len(items)) if items[i] not in chosen]
                if rest_idx:
                    order = np.argsort(-scores[rest_idx])
                    need = k - len(chosen)
                    chosen.extend([items[rest_idx[i]] for i in order[:need]])
            return chosen[:k]

        pos_sel: list[dict] = []
        neg_sel: list[dict] = []
        for _ in range(self.opt_cfg.max_iter):
            pos_sel = pick_side(pos_items, pos_ref_u, k_pos, rep_is_pos_low)
            if pos_sel:
                neg_ref_u = float(np.mean([float(x.get(bio_key, 0.0)) for x in pos_sel]))
            neg_sel = pick_side(neg_items, neg_ref_u, k_neg, rep_low=False)
            if neg_sel:
                pos_ref_u = float(np.mean([float(x.get(bio_key, 0.0)) for x in neg_sel]))

        return pos_sel, neg_sel

    def _select_bucket_pareto(
        self,
        pos_items: list[dict],
        neg_items: list[dict],
        k_pos: int,
        k_neg: int,
    ) -> tuple[list[dict], list[dict]]:
        """Pareto-style selection using a non-dominated front."""
        rep_key = "rep_metric"
        bio_key = "utility_metric"

        def with_objectives(
            items: list[dict],
            u_ref: float,
            rep_low: bool,
        ) -> list[tuple[dict, float, float]]:
            if not items:
                return []
            r = np.array([float(x.get(rep_key, np.nan)) for x in items], dtype=float)
            u = np.array([float(x.get(bio_key, np.nan)) for x in items], dtype=float)
            r = np.where(np.isfinite(r), r, 0.0)
            u = np.where(np.isfinite(u), u, 0.0)
            rs = self._robust_scale(r)
            g1 = -rs if rep_low else +rs
            du = np.abs(u - u_ref)
            g2 = -np.abs(self._robust_scale(du))
            return list(zip(items, g1.tolist(), g2.tolist()))

        def nondom_pick(cands: list[tuple[dict, float, float]], k: int) -> list[dict]:
            if k <= 0 or not cands:
                return []
            cands = sorted(cands, key=lambda x: (x[1], x[2]), reverse=True)
            front: list[tuple[dict, float, float]] = []
            max_g2 = -1e9
            for it in cands:
                if it[2] >= max_g2 - 1e-12:
                    front.append(it)
                    if it[2] > max_g2:
                        max_g2 = it[2]
            if len(front) >= k:
                front.sort(key=lambda x: (x[1] + x[2]), reverse=True)
                return [x[0] for x in front[:k]]
            rest = [x for x in cands if x not in front]
            need = k - len(front)
            return [x[0] for x in front] + [x[0] for x in rest[:need]]

        def safe_mean(arr: list[dict], key: str) -> float:
            vals = [float(r.get(key, np.nan)) for r in arr]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else 0.0

        pos_ref_u = safe_mean(neg_items, bio_key) if neg_items else safe_mean(pos_items, bio_key)
        neg_ref_u = safe_mean(pos_items, bio_key) if pos_items else safe_mean(neg_items, bio_key)

        rep_low_pos = True  # Positive side prefers low repetition.

        def pick_side(items: list[dict], u_ref: float, k: int, rep_low: bool) -> list[dict]:
            if k <= 0 or not items:
                return []
            vals = np.array([float(r.get(bio_key, np.nan)) for r in items], dtype=float)
            vals = np.where(
                np.isfinite(vals),
                vals,
                np.nanmean(vals) if np.isfinite(np.nanmean(vals)) else 0.0,
            )
            quota, edges = self._hist_quota(vals, self.opt_cfg.bins_in_bucket, k)
            chosen: list[dict] = []
            for b in range(len(quota)):
                if quota[b] <= 0:
                    continue
                left, right = edges[b], edges[b + 1]
                mask = (vals >= left) & (vals <= right if b == len(quota) - 1 else vals < right)
                sub = [items[i] for i in np.flatnonzero(mask)]
                if not sub:
                    continue
                objs = with_objectives(sub, u_ref, rep_low)
                chosen.extend(nondom_pick(objs, quota[b]))
            if len(chosen) < k:
                objs = with_objectives(items, u_ref, rep_low)
                remain = k - len(chosen)
                extra = nondom_pick(objs, remain)
                chosen.extend([x for x in extra if x not in chosen])
            return chosen[:k]

        pos_sel = pick_side(pos_items, pos_ref_u, k_pos, rep_low_pos)
        if pos_sel:
            neg_ref_u = float(np.mean([float(x.get(bio_key, 0.0)) for x in pos_sel]))
        neg_sel = pick_side(neg_items, neg_ref_u, k_neg, rep_low=False)
        if neg_sel:
            pos_ref_u = float(np.mean([float(x.get(bio_key, 0.0)) for x in neg_sel]))

        return pos_sel, neg_sel

    # ---------- Bucket balancing ---------- #

    def _balance(self, pos: list[dict], neg: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Balance on (length_bin, plddt_bin) so pos/neg distributions match.

        Modes:
        - simple   : global random sampling
        - random   : per-bucket random sampling with size=min(pos, neg)
        - composite: composite scoring described in _select_bucket_composite
        - pareto   : Pareto-style selection described in _select_bucket_pareto
        """
        rng = np.random.default_rng(self.seed)

        # Simple: global sampling without bucket constraints.
        if self.opt_cfg.method == "simple":
            total_target = self.opt_cfg.target_per_side
            n = total_target if total_target is not None else min(len(pos), len(neg))
            n_pos = min(n, len(pos))
            n_neg = min(n, len(neg))
            pos_bal = rng.choice(pos, size=n_pos, replace=False).tolist() if n_pos > 0 else []
            neg_bal = rng.choice(neg, size=n_neg, replace=False).tolist() if n_neg > 0 else []
            return pos_bal, neg_bal

        def group(items: list[dict]) -> dict[tuple[int, int], list[dict]]:
            g: dict[tuple[int, int], list[dict]] = {}
            for r in items:
                lb = _bucketize(r["length"], self.filter_cfg.length_bins)
                pb = _bucketize(r["plddt"], self.filter_cfg.plddt_bins)
                if lb is None or pb is None:
                    continue
                g.setdefault((lb, pb), []).append(r)
            return g

        g_pos = group(pos)
        g_neg = group(neg)
        keys = sorted(set(g_pos) & set(g_neg))

        pos_bal: list[dict] = []
        neg_bal: list[dict] = []

        total_target = self.opt_cfg.target_per_side
        if self.opt_cfg.method == "random" or total_target is None:
            # Legacy behavior: random sampling per bucket using min sizes.
            for k in keys:
                a = g_pos[k]
                b = g_neg[k]
                n = min(len(a), len(b))
                if self.filter_cfg.max_per_bucket is not None:
                    n = min(n, int(self.filter_cfg.max_per_bucket))
                if n <= 0:
                    continue
                pos_sel = rng.choice(a, size=n, replace=False).tolist()
                neg_sel = rng.choice(b, size=n, replace=False).tolist()
                pos_bal.extend(pos_sel)
                neg_bal.extend(neg_sel)
            return pos_bal, neg_bal

        # For composite/pareto modes assign per-bucket quotas.
        capacities = []
        for k in keys:
            cap = min(len(g_pos[k]), len(g_neg[k]))
            if self.filter_cfg.max_per_bucket is not None:
                cap = min(cap, int(self.filter_cfg.max_per_bucket))
            capacities.append(cap)
        capacities = np.array(capacities, dtype=int)
        total_cap = int(capacities.sum())
        if total_cap == 0:
            return [], []

        raw = capacities / total_cap * total_target
        quota = np.floor(raw).astype(int)
        remain = total_target - int(quota.sum())
        frac = raw - quota
        order = np.argsort(-frac)
        for i in range(remain):
            quota[order[i]] += 1

        for k, q in zip(keys, quota.tolist()):
            if q <= 0:
                continue
            a = g_pos[k]
            b = g_neg[k]
            q = min(q, len(a), len(b))
            if q <= 0:
                continue

            if self.opt_cfg.method == "composite":
                pos_sel, neg_sel = self._select_bucket_composite(a, b, q, q)
            elif self.opt_cfg.method == "pareto":
                pos_sel, neg_sel = self._select_bucket_pareto(a, b, q, q)
            else:
                pos_sel = rng.choice(a, size=q, replace=False).tolist()
                neg_sel = rng.choice(b, size=q, replace=False).tolist()

            pos_bal.extend(pos_sel)
            neg_bal.extend(neg_sel)

        return pos_bal, neg_bal

    # ---------- Build & summary stats ---------- #

    def _build(self, cache_manifest: Path | None) -> None:
        # 1) filter
        pos_f = self._filter_side(self.pos_pairs, self.pos_m, "pos")
        neg_f = self._filter_side(self.neg_pairs, self.neg_m, "neg")

        # 2) balance / optimize
        pos_b, neg_b = self._balance(pos_f, neg_f)

        # 3) combine + stats
        self.items = pos_b + neg_b

        def _stats(arr: list[dict]) -> dict:
            if not arr:
                return {}
            lens = np.array([r["length"] for r in arr], dtype=float)
            ents = np.array([r.get("entropy", np.nan) for r in arr], dtype=float)
            reps = np.array([r.get("rep_metric", np.nan) for r in arr], dtype=float)
            plddts = np.array([r.get("plddt", np.nan) for r in arr], dtype=float)
            utilities = np.array([r.get("utility_metric", np.nan) for r in arr], dtype=float)

            def fmean(x: np.ndarray) -> float:
                x = x[np.isfinite(x)]
                return float(x.mean()) if len(x) else float("nan")

            return {
                "n": len(arr),
                "len_mean": fmean(lens),
                "ent_mean": fmean(ents),
                "repetition_mean": fmean(reps),
                "plddt_mean": fmean(plddts),
                "ptm_mean": fmean(utilities),
            }

        pos_stats = _stats(pos_b)
        neg_stats = _stats(neg_b)

        # Delta repetition and delta utility.
        def mean_key(arr: list[dict], key: str) -> float:
            vals = [float(r.get(key, np.nan)) for r in arr]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        pos_rep_mean = mean_key(pos_b, "rep_metric")
        neg_rep_mean = mean_key(neg_b, "rep_metric")
        pos_bio_mean = mean_key(pos_b, "utility_metric")
        neg_bio_mean = mean_key(neg_b, "utility_metric")

        delta_rep = (
            float(abs(pos_rep_mean - neg_rep_mean))
            if np.isfinite(pos_rep_mean) and np.isfinite(neg_rep_mean)
            else float("nan")
        )
        delta_bio = (
            float(abs(pos_bio_mean - neg_bio_mean))
            if np.isfinite(pos_bio_mean) and np.isfinite(neg_bio_mean)
            else float("nan")
        )

        self.stats = {
            "cfg": asdict(self.filter_cfg),
            "opt": asdict(self.opt_cfg),
            "metric_keys": asdict(self.metric_keys),
            "pos_raw": len(self.pos_pairs),
            "neg_raw": len(self.neg_pairs),
            "pos_after_filter": len(pos_f),
            "neg_after_filter": len(neg_f),
            "pos_after_balance": len(pos_b),
            "neg_after_balance": len(neg_b),
            "pos_stats": pos_stats,
            "neg_stats": neg_stats,
            "delta_rep": delta_rep,
            "delta_bio": delta_bio,
            "rep_key": "rep_metric",
            "bio_key": "utility_metric",
            "pos_rep_mean": pos_rep_mean,
            "neg_rep_mean": neg_rep_mean,
            "pos_bio_mean": pos_bio_mean,
            "neg_bio_mean": neg_bio_mean,
        }

        # 4) optional manifest
        if cache_manifest is not None:
            cache_manifest.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stats": self.stats,
                "items": [
                    {
                        "seq_id": r["seq_id"],
                        "sequence": r["sequence"],
                        "length": r["length"],
                        "entropy": r["entropy"],
                        "rep_metric": r["rep_metric"],
                        "utility_metric": r["utility_metric"],
                        "plddt": r["plddt"],
                        "source": r["source"],
                    }
                    for r in self.items
                ],
            }
            cache_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # PyTorch-style conveniences
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

    def to_fasta(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in self.items:
                f.write(
                    f">{r['seq_id']}|src={r['source']}|len={r['length']}"
                    f"|ent={r['entropy']:.4f}|plddt={r['plddt']:.1f}\n"
                )
                s = r["sequence"]
                for i in range(0, len(s), 60):
                    f.write(s[i : i + 60] + "\n")


# ===================== Provider wrapper ===================== #


class PosNegProvider(DatasetProvider):
    """Thin wrapper that builds/caches PosNegDataset instances and artifacts."""

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self._last_ds: PosNegDataset | None = None
        self._last_out: Path | None = None

    # ---------- private helpers ---------- #
    @staticmethod
    def _manifest_path(out_dir: Path) -> Path:
        return Path(out_dir) / "dataset_manifest.json"

    def _try_load_from_manifest(self, out_dir: Path) -> PosNegDataset | None:
        """Rehydrate a lightweight PosNegDataset from a cached manifest."""
        manifest = self._manifest_path(out_dir)
        if not manifest.exists():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            items = payload.get("items", [])
            stats = payload.get("stats", {})
            for rec in items:
                if "entropy" not in rec and "entropy_norm" in rec:
                    rec["entropy"] = rec["entropy_norm"]
                if "rep_metric" not in rec and "repetition" in rec:
                    rec["rep_metric"] = rec["repetition"]
                if "utility_metric" not in rec:
                    if "ptm" in rec and np.isfinite(rec.get("ptm", np.nan)):
                        rec["utility_metric"] = rec["ptm"]
                    elif "plddt" in rec:
                        rec["utility_metric"] = rec["plddt"]
            # Create an instance without invoking __init__.
            dummy_filter_cfg = FilterConfig(**(stats.get("cfg", {}) or {}))
            dummy_opt_cfg = OptimizeConfig(**(stats.get("opt", {}) or {}))
            dummy_metric_keys = MetricKeyConfig(**(stats.get("metric_keys", {}) or {}))
            ds = PosNegDataset.__new__(PosNegDataset)  # bypass __init__
            ds.filter_cfg = dummy_filter_cfg
            ds.opt_cfg = dummy_opt_cfg
            ds.metric_keys = dummy_metric_keys
            ds.pos_pairs = []
            ds.neg_pairs = []
            ds.pos_m = {}
            ds.neg_m = {}
            ds.items = items
            ds.stats = stats
            return ds
        except Exception:
            # Manifest is missing/invalid/old format: rebuild from scratch.
            return None

    @staticmethod
    def _to_plain(d):
        if isinstance(d, (DictConfig, ListConfig)):
            return OmegaConf.to_container(d, resolve=True)
        return dict(d) if isinstance(d, dict) else {}

    @staticmethod
    def _maybe_path(v) -> Path | None:
        # Allow various "empty" sentinel values.
        if v is None:
            return None
        if isinstance(v, (DictConfig, ListConfig)):
            v = OmegaConf.to_container(v, resolve=True)
        if isinstance(v, str):
            vs = v.strip().lower()
            if vs in ("", "none", "null", "nil"):
                return None
            return Path(v)
        return Path(v)

    @staticmethod
    def _to_str_tuple(val) -> tuple[str, ...] | None:
        if val is None:
            return tuple()
        if isinstance(val, (DictConfig, ListConfig)):
            val = OmegaConf.to_container(val, resolve=True)
        if isinstance(val, str):
            vs = val.strip()
            return (vs,) if vs else tuple()
        if isinstance(val, Iterable):
            out: list[str] = []
            for item in val:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
            return tuple(out)
        s = str(val).strip()
        return (s,) if s else tuple()

    def _metric_key_cfg(self) -> MetricKeyConfig:
        raw = self._to_plain(self.cfg.get("metrics", {}))
        if not raw:
            return MetricKeyConfig()
        normalized: dict[str, tuple[str, ...]] = {}

        # Allow short-hands like `rep` / `utility` for the common case.
        rep_alias = raw.get("rep") or raw.get("repetition_metric")
        util_alias = raw.get("utility") or raw.get("utility_metric")

        for field in MetricKeyConfig.__dataclass_fields__.keys():
            val = raw.get(field)
            if val is None:
                continue
            tup = self._to_str_tuple(val)
            if tup:
                normalized[field] = tup

        if rep_alias is not None:
            tup = self._to_str_tuple(rep_alias)
            if tup:
                normalized.setdefault("repetition", tup)
                normalized.setdefault("entropy", tup)
        if util_alias is not None:
            tup = self._to_str_tuple(util_alias)
            if tup:
                normalized.setdefault("utility", tup)

        return MetricKeyConfig(**normalized)

    # ---------- public API ---------- #
    def build(self, out_dir: Path):
        """Cache-first build that persists manifests/FASTA outputs."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._last_out = out_dir

        # Try cache first.
        ds = self._try_load_from_manifest(out_dir)
        if ds is None:
            fcfg = self._to_plain(self.cfg.get("filter", {}))
            ocfg = self._to_plain(self.cfg.get("opt", {}))
            mcfg = self._metric_key_cfg()

            filter_cfg = FilterConfig(**fcfg) if fcfg else FilterConfig()
            opt_cfg = OptimizeConfig(**ocfg) if ocfg else OptimizeConfig()

            manifest_path = self._manifest_path(out_dir)

            ds = PosNegDataset(
                pos_fasta=Path(self.cfg["pos_fasta"]),
                pos_metrics=self._maybe_path(self.cfg.get("pos_metrics", None)),
                neg_fasta=Path(self.cfg["neg_fasta"]),
                neg_metrics=self._maybe_path(self.cfg.get("neg_metrics", None)),
                seed=self.cfg["seed"],
                filter_cfg=filter_cfg,
                cache_manifest=manifest_path,
                opt_cfg=opt_cfg,
                metric_keys=mcfg,
            )

            # Persist original pos/neg FASTA for reference.
            try:
                (out_dir / "pos.fasta").write_text(
                    Path(self.cfg["pos_fasta"]).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                (out_dir / "neg.fasta").write_text(
                    Path(self.cfg["neg_fasta"]).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except Exception:
                pass

            # Write the balanced combined FASTA.
            try:
                ds.to_fasta(out_dir / "balanced.fasta")
            except Exception:
                pass

        self._last_ds = ds
        return ds

    # ---- accessors for downstream scripts ---- #
    def items(self) -> list[dict]:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        return self._last_ds.items

    def stats(self) -> dict:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        return self._last_ds.stats

    def iter_pos(self) -> Iterable[tuple[str, str]]:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        for r in self._last_ds.items:
            if r.get("source") == "pos":
                yield (r["seq_id"], r["sequence"])

    def iter_neg(self) -> Iterable[tuple[str, str]]:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        for r in self._last_ds.items:
            if r.get("source") == "neg":
                yield (r["seq_id"], r["sequence"])
