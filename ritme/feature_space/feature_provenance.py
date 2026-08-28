import re
import warnings
from typing import Dict, Literal, NamedTuple, Optional

from ritme.feature_space.aggregate_features import build_taxonomy_mapping
from ritme.feature_space.utils import _PAST_SUFFIX_RE

_TRANSFORM_PREFIXES = ("clr", "alr", "pa", "rank")
_ILR_RE = re.compile(r"^ilr_\d+$")
_LUMPED_COLUMNS = ("F_low_abun", "F_low_var")
_SHANNON_COLUMN = "shannon_entropy"

# What a design-matrix column is. "opaque" marks a trial whose coefficients
# are not labeled by design-matrix column at all (trac), so its columns
# cannot be compared.
ProvenanceKind = Literal[
    "taxon", "lumped", "balance", "diversity", "metadata", "opaque"
]
# How a reference feature relates to another trial's feature space. A
# separate vocabulary from ProvenanceKind: "lumped" here means the feature
# was absorbed into the other trial's bucket, not that the column IS one.
MatchKind = Literal[
    "identical", "contained", "split", "lumped", "absent", "not_attributable"
]
_MICROBIAL_KINDS = ("taxon", "lumped")


class FeatureProvenance(NamedTuple):
    column: str
    kind: ProvenanceKind
    snapshot: str
    otus: frozenset[str]


def split_snapshot_suffix(column: str) -> tuple:
    """Return ``(base, snapshot_label)``; unsuffixed columns are ``t0``."""
    match = _PAST_SUFFIX_RE.search(column)
    if match is None:
        return column, "t0"
    return column[: match.start()], column[match.start() + 2 :]


def strip_transform_prefix(name: str, data_transform: Optional[str]) -> str:
    """Undo the column prefix added by ``transform_microbial_features``."""
    if data_transform in _TRANSFORM_PREFIXES and name.startswith(f"{data_transform}_"):
        return name[len(data_transform) + 1 :]
    return name


def _aggregation_universe(tmodel) -> Dict[str, frozenset]:
    """Map post-aggregation feature name -> source OTU set, derived from the
    trial taxonomy. Without aggregation each taxonomy id maps to itself;
    ids absent from the taxonomy (e.g. metadata columns) are not in the
    universe. Empty when the trial carries no taxonomy."""
    method = tmodel.data_config.get("data_aggregation")
    if method is None:
        if tmodel.tax is None:
            return {}
        return {otu: frozenset({otu}) for otu in tmodel.tax.index}
    tax_entity = method.split("_", 1)[1]
    mapping = build_taxonomy_mapping(tmodel.tax, tax_entity)
    universe: Dict[str, set] = {}
    for otu, label in mapping.items():
        universe.setdefault(label, set()).add(otu)
    return {label: frozenset(otus) for label, otus in universe.items()}


def _microbial_otus(
    base: str, universe: Dict[str, frozenset], selected: list
) -> frozenset:
    if base in _LUMPED_COLUMNS:
        survivors = set(selected) - set(_LUMPED_COLUMNS)
        lumped = set()
        for label, otus in universe.items():
            if label not in survivors:
                lumped |= otus
        return frozenset(lumped)
    if base in universe:
        return universe[base]
    return frozenset({base})


def build_provenance_map(tmodel) -> Dict[str, FeatureProvenance]:
    """Map every design-matrix column of a fitted ``TunedModel`` to its
    provenance (kind, snapshot, source OTUs).

    A ``trac`` model maps entirely to ``opaque``: its coefficients are
    labeled by clade, not by design-matrix column, so
    :func:`match_provenance` refuses to compare against it.
    """
    if tmodel.model_type == "trac":
        return {
            column: FeatureProvenance(column, "opaque", "t0", frozenset())
            for column in tmodel.final_feature_cols
        }

    data_transform = tmodel.data_config.get("data_transform")
    universe = _aggregation_universe(tmodel)

    provenance: Dict[str, FeatureProvenance] = {}
    unresolved_lumped = []
    for column in tmodel.final_feature_cols:
        base_suffixed, snapshot = split_snapshot_suffix(column)
        selected = tmodel.snapshot_selected_map.get(snapshot, [])
        if base_suffixed == _SHANNON_COLUMN:
            provenance[column] = FeatureProvenance(
                column, "diversity", snapshot, frozenset()
            )
            continue
        if _ILR_RE.match(base_suffixed):
            provenance[column] = FeatureProvenance(
                column, "balance", snapshot, frozenset()
            )
            continue
        base = strip_transform_prefix(base_suffixed, data_transform)
        if base in selected or base in _LUMPED_COLUMNS:
            kind = "lumped" if base in _LUMPED_COLUMNS else "taxon"
            otus = _microbial_otus(base, universe, selected)
            if kind == "lumped" and not otus:
                unresolved_lumped.append(column)
            provenance[column] = FeatureProvenance(column, kind, snapshot, otus)
            continue
        provenance[column] = FeatureProvenance(
            column, "metadata", snapshot, frozenset()
        )
    if unresolved_lumped:
        warnings.warn(
            f"Column(s) {sorted(unresolved_lumped)} hold the features that "
            "selection discarded, but no taxonomy was available to resolve "
            "which ones those are; they will be reported as not attributable "
            "in cross-trial comparisons. Pass a taxonomy to compare them."
        )
    return provenance


_FEATURE_SPACE_KEYS = (
    "data_aggregation",
    "data_selection",
    "data_selection_i",
    "data_selection_q",
    "data_selection_t",
    "data_transform",
)


class MatchResult(NamedTuple):
    kind: str
    columns: tuple


def same_feature_space(config_a: dict, config_b: dict) -> bool:
    """True when two trial configs produce identical microbial feature spaces."""
    return all(config_a.get(k) == config_b.get(k) for k in _FEATURE_SPACE_KEYS)


def _trial_uses_ilr(trial_map: Dict[str, FeatureProvenance]) -> bool:
    return any(p.kind == "balance" for p in trial_map.values())


def _trial_is_opaque(trial_map: Dict[str, FeatureProvenance]) -> bool:
    return any(p.kind == "opaque" for p in trial_map.values())


def match_provenance(
    target: FeatureProvenance,
    trial_map: Dict[str, FeatureProvenance],
    identical_space: bool,
) -> MatchResult:
    """Locate a reference feature inside another trial's feature space.

    Returns one of ``MatchKind``:

    - ``identical``: same source OTUs, whether found by column name or by
      set equality across differing transform prefixes (``clr_F1`` in one
      trial and ``pa_F1`` in another are the same taxon).
    - ``contained``: the trial holds this feature inside a strictly coarser
      column.
    - ``split``: the feature decomposes into several finer columns.
    - ``lumped``: the feature was absorbed into the trial's
      low-abundance/variance bucket.
    - ``absent``: comparable feature space, but no counterpart.
    - ``not_attributable``: comparison is undefined -- an ILR or ``trac``
      trial, or a target whose source OTUs are unknown. Callers must not
      read this as disagreement.
    """
    if _trial_is_opaque(trial_map):
        return MatchResult("not_attributable", ())
    if target.kind in _MICROBIAL_KINDS and not target.otus:
        # Membership unknown (a lumped column built without a taxonomy);
        # the empty set would match everything below vacuously.
        return MatchResult("not_attributable", ())
    match = trial_map.get(target.column)
    if (
        match is not None
        # always true for factory-built maps; guards hand-built ones
        and match.snapshot == target.snapshot
        and (target.kind != "balance" or identical_space)
        # F_low_abun/F_low_var hold whatever each trial's selection
        # discarded, so equal names can mean different features
        and (target.kind not in _MICROBIAL_KINDS or match.otus == target.otus)
    ):
        return MatchResult("identical", (target.column,))
    if target.kind == "balance":
        return MatchResult("not_attributable", ())
    if target.kind in ("diversity", "metadata"):
        return MatchResult("absent", ())
    # target is microbial (taxon / lumped): compare OTU sets per snapshot
    if _trial_uses_ilr(trial_map):
        return MatchResult("not_attributable", ())
    same_snapshot = [p for p in trial_map.values() if p.snapshot == target.snapshot]
    # equality before containment: the same taxon under a different
    # transform prefix is identical, not "contained in a coarser taxon"
    for prov in same_snapshot:
        if prov.kind == "taxon" and prov.otus == target.otus:
            return MatchResult("identical", (prov.column,))
    for prov in same_snapshot:
        if prov.kind == "taxon" and prov.otus >= target.otus:
            return MatchResult("contained", (prov.column,))
    parts = tuple(
        p.column
        for p in same_snapshot
        if p.kind == "taxon" and p.otus and p.otus < target.otus
    )
    if parts:
        return MatchResult("split", parts)
    for prov in same_snapshot:
        if prov.kind == "lumped" and prov.otus >= target.otus:
            return MatchResult("lumped", (prov.column,))
    return MatchResult("absent", ())


def matched_otu_coverage(
    target: FeatureProvenance,
    trial_map: Dict[str, FeatureProvenance],
    matched_columns: tuple,
) -> float:
    """Fraction of the target's source OTUs the matched column(s) account
    for, or NaN when the target has no known OTUs.

    A ``split`` match need not cover the target: parts that did not survive
    the other trial's selection sit in its lumped bucket and do not appear
    in ``matched_columns``.
    """
    if not target.otus:
        return float("nan")
    covered = frozenset().union(
        *(
            [trial_map[c].otus for c in matched_columns if c in trial_map]
            or [frozenset()]
        )
    )
    return len(covered & target.otus) / len(target.otus)
