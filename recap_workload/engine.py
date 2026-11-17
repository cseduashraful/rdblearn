"""fastdfs's SQL engine extended to RECAP's specifications ("recap2sql"), at fastdfs's own level of optimization.

RECAP describes a feature as a specification: a path from the entity along foreign keys, an aggregation over the
rows the path reaches, a time window and filters on the last table's rows. This module turns a specification into
one or more featuretools features of fastdfs's own form -- aggregations over relationship paths, through direct
features -- runs them through fastdfs's own execution path (features2sql with its cutoff-time CTEs, merge_queries,
its concurrent SQL workers, its frame assembly), and combines them where a specification is a ratio. What is added,
only where fastdfs has no equivalent:

  1. A conditioned value. At the path's last table the value becomes CASE WHEN <filters and window> THEN <value> END
     -- the value being the column, 1 for a count, or an expression of the row's time and the cutoff (a decay weight,
     an age in days) -- and fastdfs's own aggregate (COUNT, SUM, AVG, MIN, MAX, STDDEV_SAMP, COUNT DISTINCT) runs
     over it; all of them skip NULLs, so this is exact. The condition sits in the query's projection, not its WHERE,
     so the specifications of a path keep one query shape and fastdfs's merge_queries merges them. The last table's
     cutoff CTE carries all its columns, identical for every specification of the path, so the CTEs merge too.
  2. A final combination for ratios: share, time-part shares and velocity are two counts or sums divided, a mean
     over two levels is a sum over a count (done in pandas on fastdfs's output, as RDBLearn assembles its frames).

Shapes: one hop down from the entity then hops up (1N N1*), and, through a shared parent, down-up-down
(1N N1* 1N N1*) as fastdfs's nested aggregation: per parent row an inner aggregate of its rows, then an outer one
over the entity's middle rows (count: SUM of COUNT, sum: SUM of SUM, min/max: MIN of MIN / MAX of MAX, mean: SUM of
SUM over SUM of COUNT) -- the join's rows, as RECAP counts them. Aggregations: count, sum, mean, min, max, std,
distinct (one level only), share, the time-part shares, decay, velocity, since, age. Everything else raises
Unsupported and is reported as not covered. Rows dated at the cutoff are included (include_cutoff_time=True), as
RECAP includes them.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import featuretools as ft
import numpy as np
import pandas as pd
from featuretools.entityset.relationship import RelationshipPath
from featuretools.primitives import AggregationPrimitive
from sqlglot import exp
from woodwork.column_schema import ColumnSchema

from fastdfs.dfs import gen_sqls
from fastdfs.dfs.base_engine import dfs_engine
from fastdfs.dfs.dfs2sql_engine import DFS2SQLEngine
from fastdfs.dfs.gen_sqls import FeatureBlockWithCutoffTime, str_to_quoted_identifier

SQL_FUNC = {"count": "count", "sum": "sum", "avg": "avg", "min": "min", "max": "max", "std": "stddev_samp"}
TIME_PARTS = ("weekend", "night", "aff_dow", "aff_dom", "aff_month")
PERIODS = ("p_slope", "p_ratio", "p_active", "p_peak")
PEAK_TIE = 1e-9                          # RECAP's p_peak tie (task_module.features.PEAK_TIE, switch peak_tie on)
ONE_LEVEL = ("count", "sum", "mean", "min", "max", "std", "distinct", "share", "decay", "velocity", "since", "age",
             "last", "slope", "gap_mean", "overdue") + TIME_PARTS + PERIODS
TWO_LEVEL = ("count", "sum", "mean", "min", "max", "share", "decay", "velocity", "since", "age", "last", "std",
             "per_parent", "slope") + TIME_PARTS + PERIODS


class Unsupported(Exception):
    pass


EXISTS = "RECAP_ENTITY_EXISTS"


def _h(*parts) -> str:
    return hashlib.blake2b(json.dumps(parts, sort_keys=True, default=str).encode(), digest_size=8).hexdigest()


class RecapAgg(AggregationPrimitive):
    """A carrier: fastdfs reads the aggregation from the primitive; the part's identity rides along."""
    name = "recap"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = True

    def __init__(self, agg: str, key: str, zero: bool, pick: Optional[tuple] = None, extract: Optional[str] = None,
                 cond: Optional[str] = None, value: Optional[str] = None):
        self.agg, self.key, self.pick = agg, key, pick       # pick: (time column, join column) for agg "latest"
        self.extract = extract                               # a field of the struct the aggregate returns
        self.cond, self.value = cond, value                  # over the aggregated rows' own table (a pivot's window)
        # fastdfs coalesces an empty aggregation to 0 when its primitive is named "count"
        self.name = "count" if zero else f"recap_{agg}"

    def generate_name(self, base_feature_names, relationship_path_name, parent_dataframe_name, where_str,
                      use_prev_str):
        return "RECAP_" + _h(self.key)


# ----------------------------------------------------------------------------------------------- specifications
@dataclass
class Part:
    """One fastdfs feature: `agg` over CASE WHEN cond THEN value END at the last table; for a two-level path,
    `outer` over the middle rows of the inner aggregate. `zero`: 0 over no rows (else NULL)."""
    agg: str
    cond: Optional[str] = None           # SQL over the last table's row: {T} its alias, {C} the cutoff column
    value: Optional[str] = None          # SQL template, {V} the column; None: the column; "1": a count
    zero: bool = False
    outer: Optional[str] = None
    outer_zero: bool = False
    extract: Optional[str] = None        # the last aggregate returns a struct: this field of it
    cut: Optional[int] = None            # a one-level part on the path's first `cut` nodes, over that node's key

    def key(self, spec_key: str) -> str:
        return _h(spec_key, self.agg, self.cond, self.value, self.zero, self.outer, self.outer_zero, self.extract,
                  self.cut)


def window_days(spec: dict, delta_days: int) -> Optional[int]:
    if spec.get("window_days"):
        return int(spec["window_days"])
    if spec.get("window"):          # label windows, or periods of period_days (task_module.features.window_len)
        return int(spec["window"]) * int(spec.get("period_days") or delta_days)
    return None


def _q(c: str) -> str:
    return '"{T}"."' + c + '"'


T_COL = '"{T}"."{C}"'                        # the cutoff time, as fastdfs carries it along the path


def _lit(val) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, str):
        return "'" + val.replace("'", "''") + "'"
    return repr(val)


def _and(*conds) -> Optional[str]:
    c = [x for x in conds if x]
    return " AND ".join(f"({x})" for x in c) if c else None


def window_cond(spec, time_col, delta_days, days=None) -> Optional[str]:
    w = days if days is not None else window_days(spec, delta_days)
    if w is None:
        return None
    if not time_col:
        raise Unsupported("window on a table without a time column")
    return f"{_q(time_col)} > {T_COL} - INTERVAL {int(w)} DAY"


def filter_cond(spec) -> Optional[str]:
    parts = []
    for col, op, val in spec.get("filter") or []:
        if op not in ("=", "!=", "<", "<=", ">", ">="):
            raise Unsupported(f"filter operator {op}")
        parts.append(f"{_q(col)} {'<>' if op == '!=' else op} {_lit(val)}")
    return _and(*parts)


def time_part_cond(agg: str, ts: str, delta_days: int) -> str:
    """RECAP's time parts (task_module.features.time_part_sql), the row's time ts against the cutoff."""
    d, t = int(delta_days), T_COL
    if agg == "weekend":
        return f"dayofweek({ts}) IN (0, 6)"
    if agg == "night":
        return f"hour({ts}) < 6"
    if agg == "aff_dow":
        return "TRUE" if d >= 7 else f"((dayofweek({ts}) - dayofweek({t}) + 7) % 7) BETWEEN 1 AND {d}"
    if agg == "aff_dom":
        if d >= 31:
            return "TRUE"
        return f"dayofmonth({ts}) IN ({', '.join(f'dayofmonth({t} + INTERVAL {k} DAY)' for k in range(1, d + 1))})"
    if agg == "aff_month":
        return (f"((month({ts}) - month({t} + INTERVAL 1 DAY) + 12) % 12) <= "
                f"date_diff('month', {t} + INTERVAL 1 DAY, {t} + INTERVAL {d} DAY)")
    raise Unsupported(agg)


def plan_spec(spec: dict, time_col: Optional[str], delta_days: int, two_level: bool) -> Tuple[List[Part], tuple]:
    """The parts of a specification and how they combine (see `combine`)."""
    agg, col = spec["agg"], spec.get("column")
    n_nodes = len(spec["nodes"])
    win, flt = window_cond(spec, time_col, delta_days), filter_cond(spec)
    ok = _and(win, flt)
    days = (lambda: f"(epoch({T_COL}) - epoch({_q(time_col)})) / 86400.0") if time_col else None
    outer = {"count": "sum", "sum": "sum", "min": "min", "max": "max", "maxs": "maxs"} if two_level else {}

    def part(a, cond, value=None, zero=False, extract=None):
        if not two_level:
            return Part(a, cond, value, zero, extract=extract)
        if a not in outer:
            raise Unsupported(f"{a} over two levels")
        return Part(a, cond, value, zero, outer=outer[a], outer_zero=zero, extract=extract)

    if agg == "last":                  # the largest value at the latest time among the rows with a value
        if not time_col:
            raise Unsupported("last on a table without a time column")
        return [part("maxs", _and(ok, "{V} IS NOT NULL"), f"struct_pack(t := {_q(time_col)}, v := {{V}})",
                     extract="v")], ("id",)
    if agg == "slope" and two_level:   # over two levels: from nested sums of n, x, y, xy, xx (x: time in days)
        if not time_col:
            raise Unsupported("slope without a time column")
        okv = _and(ok, "{V} IS NOT NULL")
        x = f"(epoch({_q(time_col)}) / 86400.0)"
        return [part("count", okv, None, True), part("sum", okv, x, True), part("sum", okv, None, True),
                part("sum", okv, "{V} * " + x, True), part("sum", okv, f"{x} * {x}", True)], ("slope2",)
    if agg == "slope":                 # least-squares slope of the value against the row's time in days
        if not time_col:
            raise Unsupported("slope without a time column")
        return [Part("slope", _and(ok, "{V} IS NOT NULL"),
                     f"struct_pack(y := {{V}}, x := epoch({_q(time_col)}) / 86400.0)")], ("id",)

    if agg == "count":
        return [part("count", ok, "1", True)], ("id",)
    if agg in ("sum", "min", "max"):
        return [part(agg, ok)], ("id",)
    if agg == "mean":
        if not two_level:
            return [Part("avg", ok)], ("id",)
        return [part("sum", ok), part("count", ok, None, True)], ("mean",)
    if agg == "std" and not two_level:
        return [Part("std", ok)], ("id",)
    if agg == "distinct":
        if two_level:
            raise Unsupported("distinct over two levels")
        return [Part("distinct", ok, None, True)], ("id",)
    if agg == "share":
        return [part("count", ok, "1", True), part("count", win, "1", True)], ("ratio", 1.0)
    if agg in TIME_PARTS:
        if not time_col:
            raise Unsupported("time part on a table without a time column")
        tp = time_part_cond(agg, _q(time_col), delta_days)
        return [part("count", _and(ok, f"({tp}) IS TRUE"), "1", True), part("count", ok, "1", True)], ("ratio", 1.0)
    if agg == "decay":
        if not spec.get("half_life") or not time_col:
            raise Unsupported("decay without a half-life or a time column")
        wt = f"POW(0.5, {days()} / {float(spec['half_life'])!r})"
        if col is None:
            return [part("sum", ok, wt, True)], ("id",)
        return [part("sum", ok, "{V} * " + wt)], ("id",)
    if agg == "velocity":
        long, short = window_days(spec, delta_days), spec.get("short_days")
        if not long or not short:
            raise Unsupported("velocity without a window")
        rec = _and(ok, window_cond(spec, time_col, delta_days, days=short))
        if col is None:
            return [part("count", rec, "1", True), part("count", ok, "1", True)], ("ratio", long / short)
        return [part("sum", rec, None, True), part("sum", ok)], ("ratio", long / short)
    if agg in ("since", "age"):
        if not time_col:
            raise Unsupported(f"{agg} on a table without a time column")
        return [part("min" if agg == "since" else "max", ok, days())], ("id",)
    if agg == "std":                   # two levels (one level returned above): from sums of v and v*v, and the count
        return [part("sum", ok), part("sum", ok, "{V} * {V}"), part("count", ok, None, True)], ("std",)
    if agg == "per_parent":            # rows (or the column's sum) over the distinct rows of the node before the last
        if not two_level:
            raise Unsupported("per_parent on one level")
        num = part("count", ok, "1", True) if col is None else part("sum", ok, None, True)
        return [num, Part("distinct", None, None, True, cut=n_nodes - 1)], ("ratio", 1.0)
    if agg in ("gap_mean", "overdue"):
        if two_level or not time_col:
            raise Unsupported(f"{agg} over two levels" if two_level else f"{agg} without a time column")
        return [Part("min", ok, days()), Part("max", ok, days()), Part("distinct", ok, _q(time_col), True)], (agg,)
    if agg in PERIODS:                 # one value per period of the window (a count, or the column's sum, 0 if none)
        w = spec.get("window")
        if not w or w < 2 or spec.get("window_days") or not time_col:
            raise Unsupported("periods need a window of 2 or more periods")
        plen = int(spec.get("period_days") or delta_days) * 86400
        ago = f"(epoch({T_COL}) - epoch({_q(time_col)}))"
        bucket = f"(CASE WHEN {ago} <= 0 THEN 0 ELSE CAST(CEIL({ago} / {plen}.0) AS BIGINT) - 1 END)"
        ps = [part("count", _and(ok, f"{bucket} = {j}"), "1", True) if col is None else
              part("sum", _and(ok, f"{bucket} = {j}"), None, True) for j in range(int(w))]
        return ps, (agg, int(w))
    raise Unsupported(f"agg {agg}")


LATEST_ROW = ("count", "sum", "mean", "min", "max", "distinct", "share", "decay", "velocity", "since", "age",
              "last") + TIME_PARTS


def single_row(p: Part) -> Part:
    """A part over the one row a latest pick reaches: its conditioned value is the aggregate over that row (a
    count is 1 or 0 by its zero, a distinct count 1 if the value is there)."""
    if p.agg == "distinct":
        return Part("latest", p.cond, "CASE WHEN {V} IS NOT NULL THEN 1 END", True)
    if p.agg in ("std", "slope"):
        raise Unsupported(f"{p.agg} over one row")
    return Part("latest", p.cond, p.value, p.zero, extract=p.extract)


def shape(spec: dict) -> int:
    """1 for 1N N1*, 2 for 1N N1* 1N N1*; else Unsupported."""
    if spec.get("e_lag") is not None:
        raise Unsupported("family e_lag")
    if spec.get("sib_keys"):
        raise Unsupported("family sibling")
    kinds = list(spec.get("kinds") or [])
    if spec.get("pivot") is not None:
        i = spec["pivot"]
        if i != 1 or kinds[:2] != ["1N", "1N"] or any(k != "N1" for k in kinds[2:]):
            raise Unsupported("pivot reached through a shared parent" if "N1" in kinds[:i] else "pivot shape")
        return 9                                       # a per-event summary reached by going down
    if not kinds:
        return 0                                       # the entity's own columns
    downs = [i for i, k in enumerate(kinds) if k == "1N"]
    if len(downs) > 3 or (len(downs) == 3 and spec.get("latest")):
        raise Unsupported("path shape " + "-".join(kinds))
    if spec.get("latest") and kinds[0] != "1N":
        raise Unsupported("latest after a hop up")
    if not downs:
        return -1                                      # only hops up: the one row reached
    return len(downs)


def _relationship(es: ft.EntitySet, eid: str):
    child, parent = eid.split("->")
    (c_t, c_c), (p_t, p_c) = child.split(".", 1), parent.split(".", 1)
    for r in es.relationships:
        if (r.child_dataframe.ww.name, r._child_column_name, r.parent_dataframe.ww.name, r._parent_column_name) == \
                (c_t, c_c, p_t, p_c):
            return r
    raise Unsupported(f"no relationship {eid}")


def _ups(feat, nodes, eids, lo: int, hi: int, es):
    """Direct features for the hops up from nodes[lo] to nodes[hi] (the feature is on nodes[hi])."""
    for i in range(hi, lo, -1):
        feat = ft.DirectFeature(feat, child_dataframe_name=nodes[i - 1], relationship=_relationship(es, eids[i - 1]))
    return feat


def to_features(spec: dict, es: ft.EntitySet, target: str, target_rel, delta_days: int):
    """([featuretools features, one per part], combination) for `spec`, the conditions attached to the leaves."""
    levels = shape(spec)
    if levels == 0:
        return anchor_features(spec, es, target, target_rel)
    if levels == -1:
        return up_features(spec, es, target, target_rel, delta_days)
    if levels == 9:
        return pivot_features(spec, es, target, target_rel, delta_days)
    if spec.get("latest") and levels == 1 and spec["agg"] == "std":
        return [], ("null",)                           # std over the one row the latest pick reaches: NULL
    # hops up before the first hop down: the aggregation is built at that node and reached from the entity through
    # direct features, as fastdfs's own features of a parent's children are
    a = list(spec["kinds"]).index("1N")
    pre_nodes, pre_eids = list(spec["nodes"])[:a + 1], list(spec["eids"])[:a]
    if a:
        spec = dict(spec, nodes=list(spec["nodes"])[a:], eids=list(spec["eids"])[a:], kinds=list(spec["kinds"])[a:])
    wrap = lambda f: _wrap(f, pre_nodes, pre_eids, es, target, target_rel)
    nodes, eids, kinds = list(spec["nodes"]), list(spec["eids"]), list(spec["kinds"])
    sink = nodes[-1]
    df = es[sink]
    col = spec.get("column")
    if col is None:
        col = df.ww.index
    if col not in df.columns:
        raise Unsupported(f"column {sink}.{col}")
    time_col = df.ww.time_index
    if not spec.get("latest") and spec["agg"] not in (TWO_LEVEL if levels >= 2 else ONE_LEVEL):
        raise Unsupported(f"agg {spec['agg']}" + (" over two levels" if levels >= 2 else ""))
    if spec.get("latest") and spec["agg"] not in ONE_LEVEL:
        raise Unsupported(f"agg {spec['agg']} after latest")
    latest = bool(spec.get("latest"))
    if latest:
        # of the entity's rows in nodes[1] (dated <= t) only the latest is kept, then the path continues from it:
        # an outer pick over those rows of the continuation's value (one row, or an inner aggregate)
        mid_df = es[nodes[1]]
        if not mid_df.ww.time_index:
            raise Unsupported("latest on a table without a time column")
        nxt = eids[1].split("->")                      # the next hop's column on nodes[1]
        jcol = nxt[0].split(".", 1)[1] if kinds[1] == "N1" else nxt[1].split(".", 1)[1]
        pick = (mid_df.ww.time_index, jcol)
        levels_cont = levels                           # 1: the continuation is one row; 2: aggregated below a node
        if levels_cont == 1 and spec["agg"] not in LATEST_ROW:
            raise Unsupported(f"agg {spec['agg']} after latest")
        parts, comb = plan_spec(spec, time_col, delta_days, False)
        if levels_cont == 1:
            parts = [single_row(p) for p in parts]
    else:
        parts, comb = plan_spec(spec, time_col, delta_days, levels >= 2)
    skey = json.dumps(spec, sort_keys=True, default=str)
    d2 = [i for i, k in enumerate(kinds) if k == "1N"][-1]          # the second hop down (two levels)
    out = []
    for p in parts:
        pk = p.key(skey)
        if p.cut:                                      # a one-level part on a shorter path, over its last node's key
            cn, ce = nodes[:p.cut], eids[:p.cut - 1]
            cdf = es[cn[-1]]
            leaf = ft.IdentityFeature(cdf.ww[cdf.ww.index])
            leaf._recap = dict(cond=p.cond, value=p.value, name="__rv_" + pk)
            agg = ft.AggregationFeature(_ups(leaf, cn, ce, 1, len(cn) - 1, es), parent_dataframe_name=cn[0],
                                        primitive=RecapAgg(p.agg, pk, p.zero),
                                        relationship_path=RelationshipPath([(False, _relationship(es, ce[0]))]))
            out.append(wrap(agg))
            continue
        leaf = ft.IdentityFeature(df.ww[col])
        leaf._recap = dict(cond=p.cond, value=p.value, name="__rv_" + pk)
        rel0 = RelationshipPath([(False, _relationship(es, eids[0]))])
        if latest and levels == 1:                     # the continuation is one row: pick its conditioned value
            feat = _ups(leaf, nodes, eids, 1, len(nodes) - 1, es)
            agg = ft.AggregationFeature(feat, parent_dataframe_name=nodes[0],
                                        primitive=RecapAgg("latest", pk, p.zero, pick, p.extract),
                                        relationship_path=rel0)
        elif latest:                                   # inner aggregate below nodes[d2], picked at the latest row
            feat = _ups(leaf, nodes, eids, d2 + 1, len(nodes) - 1, es)
            inner = ft.AggregationFeature(feat, parent_dataframe_name=nodes[d2],
                                          primitive=RecapAgg(p.agg, pk + ":inner", p.zero),
                                          relationship_path=RelationshipPath([(False, _relationship(es, eids[d2]))]))
            mid = _ups(inner, nodes, eids, 1, d2, es)
            agg = ft.AggregationFeature(mid, parent_dataframe_name=nodes[0],
                                        primitive=RecapAgg("latest", pk, p.zero, pick, p.extract),
                                        relationship_path=rel0)
        elif levels == 1:
            feat = _ups(leaf, nodes, eids, 1, len(nodes) - 1, es)
            agg = ft.AggregationFeature(feat, parent_dataframe_name=nodes[0],
                                        primitive=RecapAgg(p.agg, pk, p.zero, extract=p.extract),
                                        relationship_path=rel0)
        else:
            # fastdfs's nested aggregation: innermost over the last hop down's rows, then one level per hop down
            downs = [i for i, k in enumerate(kinds) if k == "1N"]
            feat = _ups(leaf, nodes, eids, downs[-1] + 1, len(nodes) - 1, es)
            for lvl in range(len(downs) - 1, 0, -1):
                d = downs[lvl]
                prim = RecapAgg(p.agg, f"{pk}:inner{lvl}", p.zero) if lvl == len(downs) - 1 else \
                    RecapAgg(p.outer, f"{pk}:mid{lvl}", p.outer_zero)
                feat = ft.AggregationFeature(feat, parent_dataframe_name=nodes[d], primitive=prim,
                                             relationship_path=RelationshipPath([(False, _relationship(es, eids[d]))]))
                feat = _ups(feat, nodes, eids, downs[lvl - 1] + 1, d, es)
            agg = ft.AggregationFeature(feat, parent_dataframe_name=nodes[0],
                                        primitive=RecapAgg(p.outer, pk, p.outer_zero, extract=p.extract),
                                        relationship_path=RelationshipPath([(False, _relationship(es, eids[0]))]))
        out.append(wrap(agg))
    return out, comb


def _wrap(feat, pre_nodes, pre_eids, es, target, target_rel):
    """A feature on pre_nodes[-1], reached from the entity pre_nodes[0] by hops up, then from the instances."""
    feat = _ups(feat, pre_nodes, pre_eids, 0, len(pre_nodes) - 1, es)
    return ft.DirectFeature(feat, child_dataframe_name=target, relationship=target_rel)


def up_features(spec: dict, es: ft.EntitySet, target: str, target_rel, delta_days: int):
    """A path of hops up only: the one row it reaches (dated <= t at every node), its conditioned value passed to
    the entity. A count or distinct count over that row is 1 or 0 (0 also when no row is reached)."""
    nodes, eids = list(spec["nodes"]), list(spec["eids"])
    df = es[nodes[-1]]
    col = spec.get("column")
    if col is None:
        col = df.ww.index
    if col not in df.columns:
        raise Unsupported(f"column {nodes[-1]}.{col}")
    if spec["agg"] not in LATEST_ROW:
        raise Unsupported(f"agg {spec['agg']} over one row")
    parts, comb = plan_spec(spec, df.ww.time_index, delta_days, False)
    out, fills = [], []
    skey = json.dumps(spec, sort_keys=True, default=str)
    for p in parts:
        r = single_row(p)
        value = "{V}" if r.extract else r.value         # last over one row: the value itself (cond has IS NOT NULL)
        pk = r.key(skey)
        leaf = ft.IdentityFeature(df.ww[col])
        leaf._recap = dict(cond=r.cond, value=value, name="__rv_" + pk)
        feat = _ups(leaf, nodes, eids, 0, len(nodes) - 1, es)
        out.append(ft.DirectFeature(feat, child_dataframe_name=target, relationship=target_rel, name="RECAP_" + pk))
        fills.append(r.zero)
    return out, ("fill",) + (tuple(fills),) + (comb,)


PIVOT_INNER = {"count": "count", "sum": "sum", "mean": "avg", "min": "min", "max": "max", "std": "std"}


def pivot_features(spec: dict, es: ft.EntitySet, target: str, target_rel, delta_days: int):
    """A per-event summary reached by going down (RECAP's pivot at nodes[1], entity > events > rows): fastdfs's nested
    aggregation -- per event an inner aggregate of its rows that pass the filter, then an aggregate over the entity's
    events dated in the window. Each event is reached once per entity, so the nesting counts what RECAP counts."""
    nodes, eids = list(spec["nodes"]), list(spec["eids"])
    inner, agg = spec.get("inner"), spec["agg"]
    if inner not in PIVOT_INNER:
        raise Unsupported(f"pivot inner {inner}")
    df, ev = es[nodes[-1]], es[nodes[1]]
    col = spec.get("column") if inner != "count" else df.ww.index
    if col not in df.columns:
        raise Unsupported(f"column {nodes[-1]}.{col}")
    ets = ev.ww.time_index
    win = window_cond(spec, ets, delta_days)
    if not ets and agg in ("last", "slope", "since"):
        raise Unsupported("time-based pivot aggregation on events without a time column")
    key = _h(json.dumps(spec, sort_keys=True, default=str))
    leaf = ft.IdentityFeature(df.ww[col])
    leaf._recap = dict(cond=filter_cond(spec), value="1" if inner == "count" else None, name="__rv_" + key)
    feat = _ups(leaf, nodes, eids, 2, len(nodes) - 1, es)
    per_event = ft.AggregationFeature(feat, parent_dataframe_name=nodes[1],
                                      primitive=RecapAgg(PIVOT_INNER[inner], key + ":inner", inner == "count"),
                                      relationship_path=RelationshipPath([(False, _relationship(es, eids[1]))]))
    v = "COALESCE({V}, 0)" if inner == "count" else "{V}"     # an event without rows: a count of 0, as in RECAP
    vok = _and(win, f"({v}) IS NOT NULL")
    if agg in ("sum", "mean", "min", "max", "std"):
        prim = RecapAgg({"mean": "avg"}.get(agg, agg), key, False, cond=win, value=v)
    elif agg == "last":
        prim = RecapAgg("maxs", key, False, extract="v", cond=vok, value=f"struct_pack(t := {_q(ets)}, v := {v})")
    elif agg == "slope":
        prim = RecapAgg("slope", key, False, cond=vok,
                        value=f"struct_pack(y := {v}, x := epoch({_q(ets)}) / 86400.0)")
    elif agg == "since":
        prim = RecapAgg("min", key, False, cond=win, value=f"(epoch({T_COL}) - epoch({_q(ets)})) / 86400.0")
    else:
        raise Unsupported(f"pivot agg {agg}")
    agg_f = ft.AggregationFeature(per_event, parent_dataframe_name=nodes[0], primitive=prim,
                                  relationship_path=RelationshipPath([(False, _relationship(es, eids[0]))]))
    return [ft.DirectFeature(agg_f, child_dataframe_name=target, relationship=target_rel)], ("id",)


def anchor_features(spec: dict, es: ft.EntitySet, target: str, target_rel):
    """The entity's own columns (RECAP's anchor family; task_module.features._compile_anchor): value, since, is --
    a direct feature from the instances to the entity's row, which fastdfs joins only if it is dated <= the cutoff."""
    df = es[spec["nodes"][0]]
    agg, col = spec["agg"], spec.get("column")
    if agg == "value":
        if col not in df.columns or "numeric" not in df.ww.semantic_tags.get(col, set()):
            raise Unsupported("anchor value of a non-numeric column")
        return [ft.DirectFeature(ft.IdentityFeature(df.ww[col]), child_dataframe_name=target,
                                 relationship=target_rel, name="RECAP_" + _h(spec))], ("id",)
    if agg == "since":
        if col not in df.columns:
            raise Unsupported(f"column {col}")
        value = f"CASE WHEN {{V}} <= {T_COL} THEN (epoch({T_COL}) - epoch({{V}})) / 86400.0 END"
    elif agg == "is":
        flt = spec.get("filter") or []
        if not flt:
            raise Unsupported("anchor is without a filter")
        nulls = " OR ".join(f"{_q(c)} IS NULL" for c, _, _ in flt)
        value = f"CASE WHEN {nulls} THEN NULL WHEN {filter_cond(spec)} THEN 1 ELSE 0 END"
        col = df.ww.index
    else:
        raise Unsupported(f"anchor {agg}")
    key = _h(json.dumps(spec, sort_keys=True, default=str))
    leaf = ft.IdentityFeature(df.ww[col])
    leaf._recap = dict(cond=None, value=value, name="__rv_" + key)
    # a name of its own: featuretools would name it after the column, the same for every anchor spec
    return [ft.DirectFeature(leaf, child_dataframe_name=target, relationship=target_rel,
                             name="RECAP_" + key)], ("id",)


def combine(fm: pd.DataFrame, names: List[str], comb: tuple) -> np.ndarray:
    cols = [pd.to_numeric(fm[n], errors="coerce").to_numpy(float) if n in fm.columns else
            np.full(len(fm), np.nan) for n in names]
    if comb[0] == "fill":                              # parts over one row: a count reaching no row is 0
        cols = [np.where(np.isnan(c), 0.0, c) if z else c for c, z in zip(cols, comb[1])]
        comb = comb[2]
        fm = pd.DataFrame({n: c for n, c in zip(names, cols)})
    if comb[0] == "null":
        return np.full(len(fm), np.nan)
    if comb[0] == "id":
        return cols[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        if comb[0] == "ratio":
            num, den = cols
            return np.where(den == 0, np.nan, num * comb[1] / den)
        if comb[0] == "mean":
            num, den = cols
            return np.where(den == 0, np.nan, num / den)
        if comb[0] == "slope2":                        # least squares from sums; NULL without two distinct times
            n, sx, sy, sxy, sxx = cols
            den = n * sxx - sx * sx
            ok = (n >= 2) & (den > 1e-12 * np.abs(n * sxx))
            return np.where(ok, (n * sxy - sx * sy) / np.where(ok, den, 1.0), np.nan)
        if comb[0] == "std":                           # sample standard deviation from sums (NULL below 2 values)
            s1, s2, n = cols
            var = (s2 - s1 * s1 / n) / (n - 1)
            var = np.where(np.abs(var) <= 1e-11 * np.abs(s2 / n), 0.0, var)   # rounding of equal values: 0
            return np.where(n >= 2, np.sqrt(np.clip(var, 0, None)), np.nan)
        if comb[0] in ("gap_mean", "overdue"):
            since, age, nd = cols
            gap = np.where(nd - 1 == 0, np.nan, (age - since) / (nd - 1))
            if comb[0] == "gap_mean":
                return gap
            return np.where(gap == 0, np.nan, since / gap)
        if comb[0] in PERIODS:                         # y[0] is the latest period (task_module.features)
            w, Y = comb[1], np.column_stack(cols)
            xs = np.array([w - 1 - j for j in range(w)], float)
            xm = xs.mean()
            mean = Y.mean(axis=1)
            if comb[0] == "p_slope":
                return Y @ ((xs - xm) / ((xs - xm) ** 2).sum())
            if comb[0] == "p_ratio":
                return np.where(mean == 0, np.nan, Y[:, 0] / mean)
            if comb[0] == "p_active":
                return (Y > 0).sum(axis=1).astype(float)
            mx = Y.max(axis=1)
            tied = Y >= (mx - PEAK_TIE * np.abs(mx))[:, None]
            return np.where(mx <= 0, np.nan, np.argmax(tied, axis=1).astype(float))
    raise ValueError(comb)


# --------------------------------------------------------------------------------------------- SQL generation
class RecapFeatureBlock(FeatureBlockWithCutoffTime):
    """fastdfs's cutoff-time feature block, with the condition inside the aggregate (see the module docstring)."""

    def _recap(self):
        return getattr(self._feature, "_recap", None) if self._child is None else None

    def _build(self):
        super()._build()
        # A last table below hops up passes its (conditioned) value to the block above under a name of its own: the
        # specifications of a path are merged into one query, so a shared column name ("round", or the key itself
        # for a count) would let the aggregate above read another specification's value.
        r = self._recap()
        if r is not None and self._group_by_primitive is None and not self._is_first:
            self._generated_column_name = r["name"]

    def _picks(self) -> bool:
        g = self._group_by_primitive
        return isinstance(g, RecapAgg) and (g.agg == "latest" or g.cond is not None or g.value is not None)

    def _get_intermediate_columns_to_select(self, parent_temp_time_table, time_col):
        cols = super()._get_intermediate_columns_to_select(parent_temp_time_table, time_col)
        if self._recap() is None and not self._picks():
            return cols
        # the last table: all its columns (the condition reads them), identical for every spec of the path
        star = exp.Column(this=exp.Star(), table=str_to_quoted_identifier(self._target_table_name))
        keep = [c for c in cols if isinstance(c, exp.Column) and c.table == parent_temp_time_table]
        return keep + [star]

    def _sql(self, template: str) -> exp.Expression:
        # DuckDB SQL inserted verbatim (fastdfs prints its queries in sqlglot's generic dialect, which would rename
        # DuckDB functions such as epoch); parenthesized, so it composes as one operand
        return exp.Var(this="(" + template.replace("{T}", self._get_target_table_name())
                                         .replace("{C}", self._time_col) + ")")

    def _conditioned(self, source):
        r = self._recap()
        if r is None:
            return source
        if r["value"] is None:
            value = source
        elif r["value"] == "1":
            value = exp.Literal.number(1)
        else:
            value = self._sql(r["value"].replace("{V}", source.sql(dialect="duckdb")))
        if r["cond"] is None:
            return value
        cond = r["cond"].replace("{V}", source.sql(dialect="duckdb"))
        return exp.Case(ifs=[exp.If(this=self._sql(cond), true=value)])

    def handle_agg(self, source_column):
        source_column = self._conditioned(source_column)
        g = self._group_by_primitive
        if isinstance(g, RecapAgg) and (g.cond is not None or g.value is not None):
            src = source_column.sql(dialect="duckdb")
            value = self._sql(g.value.replace("{V}", src)) if g.value is not None else source_column
            source_column = value if g.cond is None else \
                exp.Case(ifs=[exp.If(this=self._sql(g.cond.replace("{V}", src)), true=value)])
        if isinstance(self._group_by_primitive, RecapAgg):
            agg = self._group_by_primitive.agg
            g, src = self._group_by_primitive, source_column.sql()
            if agg == "distinct":
                return exp.Count(this=exp.Distinct(expressions=[source_column]))
            if agg == "latest":                        # the value at the latest row; ties: largest join key, NULLs last
                ts, j = g.pick
                t = self._get_target_table_name()
                out = (f'arg_max_null({src}, struct_pack(t := "{t}"."{ts}", '
                       f'n := "{t}"."{j}" IS NOT NULL, j := "{t}"."{j}"))')
            elif agg == "maxs":                        # structs compare field by field
                out = f"max({src})"
            elif agg == "slope":
                out = f"regr_slope(({src}).y, ({src}).x)"
            else:
                return exp.Anonymous(this=SQL_FUNC[agg], expressions=[source_column])
            return exp.Var(this=f"(({out}).{g.extract})" if g.extract else out)
        return super().handle_agg(source_column)

    def _get_columns_to_select(self, table_name=None):
        cols = super()._get_columns_to_select()
        r = self._recap()
        if r is None or self._group_by_primitive is not None:     # aggregated here: handle_agg conditions it
            return cols
        # the last table below hops up: condition the value it passes up (the aggregation is in a block above).
        # fastdfs's columns here: [the key to join the parent (if any), the value, the cutoff time]
        i = 1 if self._key_to_join_parent is not None else 0
        out = list(cols)
        c = out[i]
        base = c.this if isinstance(c, exp.Alias) else c
        out[i] = exp.Alias(this=self._conditioned(base), alias=str_to_quoted_identifier(self._generated_column_name))
        return out


@contextlib.contextmanager
def _recap_blocks():
    """features2sql builds FeatureBlockWithCutoffTime by name; use the extended block for this engine's calls."""
    saved = gen_sqls.FeatureBlockWithCutoffTime
    gen_sqls.FeatureBlockWithCutoffTime = RecapFeatureBlock
    try:
        yield
    finally:
        gen_sqls.FeatureBlockWithCutoffTime = saved


@dfs_engine
class Recap2SQLEngine(DFS2SQLEngine):
    """fastdfs's dfs2sql engine on RECAP's specifications (prepare_specs instead of featuretools' DFS search)."""

    name = "recap2sql"

    def prepare_specs(self, rdb, target_dataframe, key_mappings, cutoff_time_column, specs: List[dict],
                      delta_days: int):
        """([features], {spec index: ([feature names], combination)}, {reason: count}) for the covered specs."""
        es = self._build_entity_set_from_rdb(rdb)
        target = "__target__"
        es = es.add_dataframe(dataframe_name=target, dataframe=target_dataframe, index="__target_index__",
                              time_index=cutoff_time_column)
        self._add_target_relationships(es, target, key_mappings)
        target_rel = [r for r in es.relationships if r.child_dataframe.ww.name == target][0]
        feats, plan, skipped, seen = [], {}, {}, set()
        for i, spec in enumerate(specs):
            if spec.get("_sampled_out"):               # not part of this run (the harness's subsets)
                continue
            try:
                fs, comb = to_features(spec, es, target, target_rel, delta_days)
            except Unsupported as e:
                skipped[str(e)] = skipped.get(str(e), 0) + 1
                continue
            names = [f.get_name() for f in fs]
            for f, n in zip(fs, names):          # a part shared by several specs (e.g. a share's denominator) once
                if n not in seen:
                    seen.add(n)
                    feats.append(f)
            plan[i] = (names, comb)
        # RECAP: an entity whose own row is dated after t does not exist yet, and every feature is NULL. fastdfs's
        # join from the instances to the entity drops such rows; a direct feature of the entity's key shows it.
        ent = list(key_mappings.values())[0].split(".")[0]
        edf = es[ent]
        if plan and edf.ww.time_index:
            feats.append(ft.DirectFeature(ft.IdentityFeature(edf.ww[edf.ww.index]), child_dataframe_name=target,
                                          relationship=target_rel, name=EXISTS))
        return feats, plan, skipped

    def compute_feature_matrix(self, rdb, target_dataframe, key_mappings, cutoff_time_column, features, config):
        with _recap_blocks():
            return super().compute_feature_matrix(rdb, target_dataframe, key_mappings, cutoff_time_column,
                                                  features, config)

    @staticmethod
    def assemble(fm: pd.DataFrame, plan: Dict[int, tuple]) -> pd.DataFrame:
        """One column per covered specification (named by its index), from fastdfs's feature matrix; NULL where the
        entity does not exist yet at the instance's time."""
        out = pd.DataFrame({str(i): combine(fm, names, comb) for i, (names, comb) in plan.items()})
        if EXISTS in fm.columns:
            out.loc[fm[EXISTS].isna().to_numpy(), :] = np.nan
        return out
