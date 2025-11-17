# src/replm/datasets/posneg_provider.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from .base import DatasetProvider
from replm.utils.io import read_fasta as _read_fasta_simple  # 如果路径/函数名不同，按实际项目改这一行即可


# ===================== 配置 ===================== #

@dataclass(frozen=True)
class FilterConfig:
    ent_hi: float = 0.90
    ent_lo: float = 0.80
    plddt_hi: float = 85.0
    length_bins: Tuple[int, ...] = (50, 129, 257, 513, 1025)
    plddt_bins: Tuple[float, ...] = (70.0, 80.0, 90.0, 101.0)
    min_len: int = 50
    max_len: int = 1024
    max_per_bucket: Optional[int] = None
    # 可选：若 metrics 里有安全/真实性标记，这里可以启用硬约束过滤
    safety_key: Optional[str] = None          # 例如 'safety_flag'（==1 通过）
    truth_key: Optional[str] = None           # 例如 'truth_flag'（==1 通过）


@dataclass(frozen=True)
class OptimizeConfig:
    """
    对比式选样配置：
    - method: 'simple'（全局纯随机）、'random'（桶内随机；保持旧逻辑）、'composite'、'pareto'
    - target_per_side: 每侧最终样本数（例如 100）；None 表示保持旧逻辑（每桶取 min 可用）
    - rep_key: 重复/重复度相关指标列名（默认 'entropy_norm'；也可用 'rep_2'/'rep_3'/'repetition'）
    - bio_key: 生物效用指标列名（默认 'ptm'，若缺失会回退到 'plddt'）
    - bins_in_bucket: 桶内基于 bio_key 的分层分箱数
    - max_iter: P/N 交替优化轮数
    - alpha/beta: 复合评分中 repetition 与 bio 匹配的权重
    - random_seed: 随机种子
    """
    method: str = "random"               # 'simple' | 'random' | 'composite' | 'pareto'
    target_per_side: Optional[int] = None
    rep_key: str = "entropy_norm"
    bio_key: str = "ptm"
    bins_in_bucket: int = 6
    max_iter: int = 3
    alpha: float = 1.0
    beta: float = 1.0
    random_seed: int = 42


# ===================== I/O 辅助 ===================== #

def _first_token(h: str) -> str:
    h = h.strip()
    return h.split()[0] if h else ""


def _read_fasta_with_ids(path: Path) -> List[Tuple[str, str]]:
    """读取 FASTA，支持有 / 无 header 的情况，给每条序列一个 seq_id。"""
    out: List[Tuple[str, str]] = []
    cur_id: Optional[str] = None
    chunks: List[str] = []
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
        # 没有 header：fallback 到简单读法，并自动生成 ID
        seqs = _read_fasta_simple(path)
        return [(f"{path.stem}_{i:06d}", s) for i, s in enumerate(seqs) if s]

    # 规范化空 ID
    return [
        (sid if sid else f"{path.stem}_{i:06d}", s)
        for i, (sid, s) in enumerate(out)
        if s
    ]


def _load_metrics(path: Path) -> Dict[str, dict]:
    """从 .jsonl 或 .csv 加载 metrics，按 seq_id 索引。

    这里只负责“读表 + 按 seq_id 分组”，不做任何指标公式计算，
    指标计算都由外部的 metrics 脚本完成。
    """
    if path.suffix.lower() == ".jsonl":
        d: Dict[str, dict] = {}
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

        d: Dict[str, dict] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "seq_id" not in (reader.fieldnames or []):
                raise ValueError("metrics table must contain 'seq_id'")
            for r in reader:
                sid = str(r["seq_id"])
                # 保留 'sequence' 字段（如果有）为字符串，其他字段尽量转为 float
                entry: Dict[str, object] = {}
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


def _bucketize(val: float, edges: Sequence[float | int]) -> Optional[int]:
    for i in range(len(edges) - 1):
        if edges[i] <= val < edges[i + 1]:
            return i
    return None


# ===================== Metric 适配 ===================== #

def _get_metric(
    m: dict,
    keys: Sequence[str],
    *,
    transform=None,
    default: float = np.nan,
) -> float:
    """通用的 metrics 读取函数。

    - keys: 依次尝试的一组字段名（兼容不同 metrics 脚本）
    - transform: 若提供，则对取出的值做一次变换，例如 0–1 -> 0–100
    """
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


def _get_entropy(m: dict) -> float:
    # repetition.py: 'H_norm'; 旧表可能是 'entropy_norm'
    return _get_metric(m, ["entropy_norm", "H_norm"])


def _get_repetition(m: dict) -> float:
    # 三个脚本合起来可能给：
    # - 'repetition'：聚合分数（可由 repetition_score 导出）
    # - 回退到 'entropy_norm'/'H_norm' 也可以凑合做 rep proxy
    return _get_metric(m, ["repetition", "entropy_norm", "H_norm"])


def _get_homopolymer(m: dict) -> float:
    # repetition.py: 'R_hpoly'
    return _get_metric(m, ["homopolymer", "R_hpoly"])


def _get_rep2(m: dict) -> float:
    # repetition.py: 'Distinct-2'
    return _get_metric(m, ["rep_2", "Distinct-2"])


def _get_rep3(m: dict) -> float:
    # repetition.py: 'Distinct-3'
    return _get_metric(m, ["rep_3", "Distinct-3"])


def _get_plddt(m: dict) -> float:
    # structure.py: 'plddt_mean_01' (0–1)；旧表：'plddt_mean_0_100' 或 'plddt'
    def to_0_100(x: float) -> float:
        # 如果看起来像 0–1 区间，就乘 100
        if 0.0 <= x <= 1.0:
            return x * 100.0
        return x

    return _get_metric(
        m,
        ["plddt_mean_0_100", "plddt", "plddt_mean_01"],
        transform=to_0_100,
    )


def _get_ptm(m: dict) -> float:
    # structure.py: 'ptm'
    return _get_metric(m, ["ptm"])


def _get_length(m: dict, seq: str) -> int:
    val = _get_metric(m, ["length"], default=len(seq))
    try:
        return int(val)
    except Exception:
        return len(seq)


# ===================== Dataset 核心实现 ===================== #

class PosNegDataset:
    """
    构造一个平衡的正/负样本集合，使两侧在 (length_bin, pLDDT_bin) 上分布匹配，
    而主要在「重复度/熵」上有显著差异。

    注意：这里不再计算任何 metrics，只是：
    - 从 metrics 表里读取（由 diversity / repetition / structure 脚本事先算好）
    - 用统一的 _get_* 辅助函数做字段名兼容与缺省处理
    """

    def __init__(
        self,
        pos_fasta: Path,
        pos_metrics: Optional[Path],
        neg_fasta: Path,
        neg_metrics: Optional[Path],
        seed: int = 42,
        filter_cfg: FilterConfig = FilterConfig(),
        *,
        cache_manifest: Optional[Path] = None,
        opt_cfg: OptimizeConfig = OptimizeConfig(),
    ) -> None:
        self.seed = seed
        self.filter_cfg = filter_cfg
        self.opt_cfg = opt_cfg
        self.pos_pairs = _read_fasta_with_ids(Path(pos_fasta))
        self.neg_pairs = _read_fasta_with_ids(Path(neg_fasta))
        self.pos_m = _load_metrics(Path(pos_metrics)) if pos_metrics else {}
        self.neg_m = _load_metrics(Path(neg_metrics)) if neg_metrics else {}

        self.items: List[dict] = []
        self.stats: dict = {}
        self._build(cache_manifest)

    # ---------- 过滤正/负样本 ---------- #
    def _filter_side(
        self,
        pairs: List[Tuple[str, str]],
        metrics: Dict[str, dict],
        mode: str,
    ) -> List[dict]:
        """
        mode='pos': 保留 pLDDT >= hi & entropy >= ent_hi
        mode='neg': 保留 pLDDT >= hi & entropy <= ent_lo
        同时约束长度范围。
        """
        out: List[dict] = []
        for sid, seq in pairs:
            m = metrics.get(sid)
            if (m is None) or (not seq):
                # 没有 metrics 的条目（无 ent/plddt）直接跳过
                continue

            L = _get_length(m, seq)
            if L < self.filter_cfg.min_len or L > self.filter_cfg.max_len:
                continue

            ent = _get_entropy(m)
            repetition = _get_repetition(m)
            homopolymer = _get_homopolymer(m)
            rep_2 = _get_rep2(m)
            rep_3 = _get_rep3(m)
            plddt = _get_plddt(m)
            ptm = _get_ptm(m)

            # 可选：安全/真实性硬约束
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
                "entropy_norm": ent,
                "repetition": repetition,
                "homopolymer": homopolymer,
                "rep_2": rep_2,
                "rep_3": rep_3,
                "plddt": plddt,
                "ptm": ptm,
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

    # ---------- 一些数值辅助 ---------- #

    @staticmethod
    def _robust_scale(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        med = np.median(x)
        q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
        iqr = max(q3 - q1, eps)
        return (x - med) / iqr

    @staticmethod
    def _hist_quota(values: np.ndarray, bins: int, total: int) -> Tuple[List[int], np.ndarray]:
        # 按分位数构造边界
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

    # ---------- 自动选择“重复度指标键名” ---------- #

    def _resolve_rep_key(
        self,
        pos_items: List[dict],
        neg_items: List[dict],
    ) -> str:
        """
        优先使用 repetition；若不可用则回退到 opt_cfg.rep_key（默认 entropy）；
        再不行则尝试 rep_2/rep_3；最终兜底 entropy。
        判定“可用”= 至少存在一个有限值。
        """
        candidates = [
            "repetition",
            (self.opt_cfg.rep_key or "entropy_norm"),
            "rep_2",
            "rep_3",
            "entropy_norm",
        ]
        seen = set()

        def has_finite(items: List[dict], key: str) -> bool:
            for r in items:
                v = r.get(key, None)
                try:
                    if v is not None and np.isfinite(float(v)):
                        return True
                except Exception:
                    pass
            return False

        for k in candidates:
            if k in seen:
                continue
            seen.add(k)
            if has_finite(pos_items, k) or has_finite(neg_items, k):
                return k
        return "entropy_norm"

    # ---------- composite / pareto 的桶内选样 ---------- #

    def _select_bucket_composite(
        self,
        pos_items: List[dict],
        neg_items: List[dict],
        k_pos: int,
        k_neg: int,
    ) -> Tuple[List[dict], List[dict]]:
        """
        复合评分法：ΔRep 最大化，ΔBio 最小化；正集低重复、负集高重复。
        """
        rep_key = self._resolve_rep_key(pos_items, neg_items)
        bio_key = self.opt_cfg.bio_key

        # bio_key 若缺失，回退到 plddt
        for arr in (pos_items, neg_items):
            if arr and (bio_key not in arr[0] or not np.isfinite(arr[0].get(bio_key, np.nan))):
                bio_key = "plddt"
                break

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
            # 方向：正集低重复、负集高重复
            s_r = -rs if rep_is_pos_low else +rs
            du = np.abs(u - u_ref)
            dus = -self._robust_scale(du)  # 距离小分高
            return self.opt_cfg.alpha * s_r + self.opt_cfg.beta * dus

        rep_is_pos_low = True

        def pick_side(items: List[dict], other_u_mean: float, k: int, rep_low: bool) -> List[dict]:
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

            chosen: List[dict] = []
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
                # 全局补齐
                rest_idx = [i for i in range(len(items)) if items[i] not in chosen]
                if rest_idx:
                    order = np.argsort(-scores[rest_idx])
                    need = k - len(chosen)
                    chosen.extend([items[rest_idx[i]] for i in order[:need]])
            return chosen[:k]

        pos_sel: List[dict] = []
        neg_sel: List[dict] = []
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
        pos_items: List[dict],
        neg_items: List[dict],
        k_pos: int,
        k_neg: int,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Pareto 近似：非支配前沿 + 分层；正集低重复、负集高重复。
        """
        rep_key = self._resolve_rep_key(pos_items, neg_items)
        bio_key = self.opt_cfg.bio_key
        # 回退 bio_key
        for arr in (pos_items, neg_items):
            if arr and (bio_key not in arr[0] or not np.isfinite(arr[0].get(bio_key, np.nan))):
                bio_key = "plddt"
                break

        def with_objectives(items: List[dict], u_ref: float, rep_low: bool) -> List[Tuple[dict, float, float]]:
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

        def nondom_pick(cands: List[Tuple[dict, float, float]], k: int) -> List[dict]:
            if k <= 0 or not cands:
                return []
            cands = sorted(cands, key=lambda x: (x[1], x[2]), reverse=True)
            front: List[Tuple[dict, float, float]] = []
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

        def safe_mean(arr: List[dict], key: str) -> float:
            vals = [float(r.get(key, np.nan)) for r in arr]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else 0.0

        pos_ref_u = safe_mean(neg_items, bio_key) if neg_items else safe_mean(pos_items, bio_key)
        neg_ref_u = safe_mean(pos_items, bio_key) if pos_items else safe_mean(neg_items, bio_key)

        rep_low_pos = True  # 正集低重复，负集高重复

        def pick_side(items: List[dict], u_ref: float, k: int, rep_low: bool) -> List[dict]:
            if k <= 0 or not items:
                return []
            vals = np.array([float(r.get(bio_key, np.nan)) for r in items], dtype=float)
            vals = np.where(
                np.isfinite(vals),
                vals,
                np.nanmean(vals) if np.isfinite(np.nanmean(vals)) else 0.0,
            )
            quota, edges = self._hist_quota(vals, self.opt_cfg.bins_in_bucket, k)
            chosen: List[dict] = []
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

    # ---------- 桶间配平（balance） ---------- #

    def _balance(self, pos: List[dict], neg: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Balance on (length_bin, plddt_bin) so pos/neg distributions match.
        支持四种模式：
        - simple   : 纯随机（忽略桶），全局从 pos/neg 各自集合中随机抽样
        - random   : 旧逻辑（每桶 min(len(pos), len(neg)) 随机抽）
        - composite: 复合评分法（ΔRep 最大化，ΔBio 最小化）
        - pareto   : Pareto 近似（非支配前沿 + 分层）
        """
        rng = np.random.default_rng(self.seed)

        # simple：纯随机，不做桶配平
        if self.opt_cfg.method == "simple":
            total_target = self.opt_cfg.target_per_side
            n = total_target if total_target is not None else min(len(pos), len(neg))
            n_pos = min(n, len(pos))
            n_neg = min(n, len(neg))
            pos_bal = rng.choice(pos, size=n_pos, replace=False).tolist() if n_pos > 0 else []
            neg_bal = rng.choice(neg, size=n_neg, replace=False).tolist() if n_neg > 0 else []
            return pos_bal, neg_bal

        def group(items: List[dict]) -> Dict[Tuple[int, int], List[dict]]:
            g: Dict[Tuple[int, int], List[dict]] = {}
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

        pos_bal: List[dict] = []
        neg_bal: List[dict] = []

        total_target = self.opt_cfg.target_per_side
        if self.opt_cfg.method == "random" or total_target is None:
            # 旧逻辑：每个桶内随机取 min(len(pos_bucket), len(neg_bucket))
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

        # 对比式算法：给每桶一个 quota
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

    # ---------- 构建 & 统计 ---------- #

    def _build(self, cache_manifest: Optional[Path]) -> None:
        # 1) filter
        pos_f = self._filter_side(self.pos_pairs, self.pos_m, "pos")
        neg_f = self._filter_side(self.neg_pairs, self.neg_m, "neg")

        # 2) balance / optimize
        pos_b, neg_b = self._balance(pos_f, neg_f)

        # 3) combine + stats
        self.items = pos_b + neg_b

        def _stats(arr: List[dict]) -> dict:
            if not arr:
                return {}
            lens = np.array([r["length"] for r in arr], dtype=float)
            ents = np.array([r.get("entropy_norm", np.nan) for r in arr], dtype=float)
            reps = np.array([r.get("repetition", np.nan) for r in arr], dtype=float)
            plddts = np.array([r.get("plddt", np.nan) for r in arr], dtype=float)
            homopolymers = np.array([r.get("homopolymer", np.nan) for r in arr], dtype=float)
            rep_2s = np.array([r.get("rep_2", np.nan) for r in arr], dtype=float)
            rep_3s = np.array([r.get("rep_3", np.nan) for r in arr], dtype=float)
            ptms = np.array([r.get("ptm", np.nan) for r in arr], dtype=float)

            def fmean(x: np.ndarray) -> float:
                x = x[np.isfinite(x)]
                return float(x.mean()) if len(x) else float("nan")

            return {
                "n": len(arr),
                "len_mean": fmean(lens),
                "ent_mean": fmean(ents),
                "repetition_mean": fmean(reps),
                "plddt_mean": fmean(plddts),
                "homopolymer_mean": fmean(homopolymers),
                "rep_2_mean": fmean(rep_2s),
                "rep_3_mean": fmean(rep_3s),
                "ptm_mean": fmean(ptms),
            }

        pos_stats = _stats(pos_b)
        neg_stats = _stats(neg_b)

        # ΔRepetition 与 ΔBio
        rep_key = self._resolve_rep_key(pos_b, neg_b) or "entropy_norm"
        bio_key = self.opt_cfg.bio_key or "ptm"

        def mean_key(arr: List[dict], key: str, fallback: Optional[str] = None) -> float:
            vals = [float(r.get(key, np.nan)) for r in arr]
            vals = [v for v in vals if np.isfinite(v)]
            if (not vals) and fallback:
                vals = [float(r.get(fallback, np.nan)) for r in arr]
                vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        pos_rep_mean = mean_key(pos_b, rep_key, "entropy_norm")
        neg_rep_mean = mean_key(neg_b, rep_key, "entropy_norm")
        pos_bio_mean = mean_key(pos_b, bio_key, "plddt")
        neg_bio_mean = mean_key(neg_b, bio_key, "plddt")

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
            "rep_key": rep_key,
            "bio_key": bio_key,
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
                        "entropy_norm": r["entropy_norm"],
                        "repetition": r.get("repetition", np.nan),
                        "homopolymer": r.get("homopolymer", np.nan),
                        "rep_2": r.get("rep_2", np.nan),
                        "rep_3": r.get("rep_3", np.nan),
                        "plddt": r["plddt"],
                        "ptm": r["ptm"],
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
                    f"|ent={r['entropy_norm']:.4f}|plddt={r['plddt']:.1f}\n"
                )
                s = r["sequence"]
                for i in range(0, len(s), 60):
                    f.write(s[i : i + 60] + "\n")


# ===================== Provider 包装（无 registry） ===================== #

class PosNegProvider(DatasetProvider):
    """
    负责：
    - 构建 PosNegDataset 并缓存
    - 提供 items()/stats()/iter_pos()/iter_neg() 这些接口
    - 将构建结果写到 out_dir（manifest + balanced.fasta）
    """

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self._last_ds: Optional[PosNegDataset] = None
        self._last_out: Optional[Path] = None

    # ---------- private helpers ---------- #
    @staticmethod
    def _manifest_path(out_dir: Path) -> Path:
        return Path(out_dir) / "dataset_manifest.json"

    def _try_load_from_manifest(self, out_dir: Path) -> Optional[PosNegDataset]:
        """如果 manifest 存在，则从中构造一个轻量级 PosNegDataset。"""
        manifest = self._manifest_path(out_dir)
        if not manifest.exists():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            items = payload.get("items", [])
            stats = payload.get("stats", {})
            # 构造一个不走 __init__ 的实例
            dummy_filter_cfg = FilterConfig(**(stats.get("cfg", {}) or {}))
            dummy_opt_cfg = OptimizeConfig(**(stats.get("opt", {}) or {}))
            ds = PosNegDataset.__new__(PosNegDataset)  # bypass __init__
            ds.filter_cfg = dummy_filter_cfg
            ds.opt_cfg = dummy_opt_cfg
            ds.pos_pairs = []
            ds.neg_pairs = []
            ds.pos_m = {}
            ds.neg_m = {}
            ds.items = items
            ds.stats = stats
            return ds
        except Exception:
            # manifest 损坏或旧格式：忽略，强制重建
            return None

    @staticmethod
    def _to_plain(d):
        if isinstance(d, (DictConfig, ListConfig)):
            return OmegaConf.to_container(d, resolve=True)
        return dict(d) if isinstance(d, dict) else {}

    @staticmethod
    def _maybe_path(v) -> Optional[Path]:
        # 允许 None / "" / "null"（字符串）等空值
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

    # ---------- public API ---------- #
    def build(self, out_dir: Path):
        """
        Cache-first build. 如果 dataset manifest 已存在且可用，直接加载；
        否则从头构建新的 PosNegDataset，并落盘各种 artifact。
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._last_out = out_dir

        # 1) 尝试从 cache 读取
        ds = self._try_load_from_manifest(out_dir)
        if ds is None:
            fcfg = self._to_plain(self.cfg.get("filter", {}))
            ocfg = self._to_plain(self.cfg.get("opt", {}))

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
            )

            # 4) 备份原始 pos/neg FASTA
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

            # 5) 写出平衡后的 combined FASTA
            try:
                ds.to_fasta(out_dir / "balanced.fasta")
            except Exception:
                pass

        self._last_ds = ds
        return ds

    # ---- accessors for downstream scripts ---- #
    def items(self) -> List[dict]:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        return self._last_ds.items

    def stats(self) -> Dict:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        return self._last_ds.stats

    def iter_pos(self) -> Iterable[Tuple[str, str]]:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        for r in self._last_ds.items:
            if r.get("source") == "pos":
                yield (r["seq_id"], r["sequence"])

    def iter_neg(self) -> Iterable[Tuple[str, str]]:
        if self._last_ds is None:
            raise RuntimeError("Call build(out_dir) first; no dataset cached.")
        for r in self._last_ds.items:
            if r.get("source") == "neg":
                yield (r["seq_id"], r["sequence"])
