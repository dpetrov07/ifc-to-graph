"""
relationship_extractor.py
--------------------------
Extracts all IFC relationships and builds a graph of edges between components.
Relationships are represented as typed directed edges suitable for GNN training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Relationship type constants (mirrors IFC schema names) ────────────────────
REL_CONTAINED_IN    = "IfcRelContainedInSpatialStructure"
REL_AGGREGATES      = "IfcRelAggregates"
REL_VOIDS           = "IfcRelVoidsElement"
REL_FILLS           = "IfcRelFillsElement"
REL_CONNECTS        = "IfcRelConnectsElements"
REL_CONNECTS_PATH   = "IfcRelConnectsPathElements"
REL_SPACE_BOUNDARY  = "IfcRelSpaceBoundary"
REL_ADJACENT        = "proximity_adjacent"   # synthetic — computed from geometry


Edge = Dict  # {"source_id", "target_id", "relationship_type", "attributes"}


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_all_relationships(model, elements: List[Any]) -> List[Edge]:
    """
    Run every extractor and return a deduplicated list of edges.
    """
    element_ids: Set[str] = {e.GlobalId for e in elements}
    edges: List[Edge] = []

    extractors = [
        _extract_contained_in,
        _extract_aggregates,
        _extract_voids,
        _extract_fills,
        _extract_connects,
        _extract_space_boundary,
    ]

    for fn in extractors:
        try:
            new_edges = fn(model, element_ids)
            edges.extend(new_edges)
        except Exception as exc:
            logger.warning("Relationship extractor %s failed: %s",
                           fn.__name__, exc)

    # deduplicate
    seen: Set[Tuple] = set()
    unique: List[Edge] = []
    for e in edges:
        key = (e["source_id"], e["target_id"], e["relationship_type"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    logger.info("Extracted %d unique relationship edges", len(unique))
    return unique


# ── Individual extractors ──────────────────────────────────────────────────────

def _extract_contained_in(model, element_ids: Set[str]) -> List[Edge]:
    edges = []
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        container_id = rel.RelatingStructure.GlobalId
        for el in rel.RelatedElements:
            if el.GlobalId in element_ids:
                edges.append(_edge(
                    el.GlobalId, container_id,
                    REL_CONTAINED_IN,
                    {"container_type": rel.RelatingStructure.is_a()},
                ))
    return edges


def _extract_aggregates(model, element_ids: Set[str]) -> List[Edge]:
    edges = []
    for rel in model.by_type("IfcRelAggregates"):
        whole = rel.RelatingObject
        for part in rel.RelatedObjects:
            # include if either end is a known element
            if part.GlobalId in element_ids or whole.GlobalId in element_ids:
                edges.append(_edge(
                    whole.GlobalId, part.GlobalId,
                    REL_AGGREGATES,
                    {"whole_type": whole.is_a(), "part_type": part.is_a()},
                ))
    return edges


def _extract_voids(model, element_ids: Set[str]) -> List[Edge]:
    edges = []
    for rel in model.by_type("IfcRelVoidsElement"):
        wall = rel.RelatingBuildingElement
        opening = rel.RelatedOpeningElement
        if wall.GlobalId in element_ids:
            edges.append(_edge(
                wall.GlobalId, opening.GlobalId,
                REL_VOIDS,
                {"opening_type": opening.is_a()},
            ))
    return edges


def _extract_fills(model, element_ids: Set[str]) -> List[Edge]:
    edges = []
    for rel in model.by_type("IfcRelFillsElement"):
        opening = rel.RelatingOpeningElement
        filler  = rel.RelatedBuildingElement
        if filler.GlobalId in element_ids:
            edges.append(_edge(
                filler.GlobalId, opening.GlobalId,
                REL_FILLS,
                {"filler_type": filler.is_a()},
            ))
    return edges


def _extract_connects(model, element_ids: Set[str]) -> List[Edge]:
    edges = []
    for rel_type in ("IfcRelConnectsElements", "IfcRelConnectsPathElements"):
        try:
            for rel in model.by_type(rel_type):
                a = rel.RelatingElement
                b = rel.RelatedElement
                if a.GlobalId in element_ids or b.GlobalId in element_ids:
                    attrs: Dict = {"rel_schema": rel_type}
                    # path-specific data
                    if rel_type == "IfcRelConnectsPathElements":
                        attrs["relating_priority"] = rel.RelatingPriority
                        attrs["related_priority"]  = rel.RelatedPriority
                    edges.append(_edge(
                        a.GlobalId, b.GlobalId,
                        REL_CONNECTS,
                        attrs,
                    ))
        except Exception:
            pass
    return edges


def _extract_space_boundary(model, element_ids: Set[str]) -> List[Edge]:
    edges = []
    try:
        for rel in model.by_type("IfcRelSpaceBoundary"):
            space = rel.RelatingSpace
            el    = rel.RelatedBuildingElement
            if el and el.GlobalId in element_ids:
                edges.append(_edge(
                    el.GlobalId, space.GlobalId,
                    REL_SPACE_BOUNDARY,
                    {
                        "physical_or_virtual": str(rel.PhysicalOrVirtualBoundary),
                        "internal_or_external": str(rel.InternalOrExternalBoundary),
                    },
                ))
    except Exception:
        pass
    return edges


# ── Proximity / adjacency (geometric) ─────────────────────────────────────────

def compute_proximity_edges(
    components: List[Dict],
    threshold: float = 1.0,          # metres
) -> List[Edge]:
    """
    For every pair of components whose bounding boxes are within `threshold`
    metres of each other (AABB gap), add a synthetic proximity edge.

    This is O(n²) — acceptable for building-scale datasets (hundreds of
    elements). For large models, use a spatial index.
    """
    edges: List[Edge] = []
    # filter to elements that have bounding box data
    with_bb = [
        c for c in components
        if c.get("geometry") and c["geometry"].get("bounding_box")
    ]

    for i, a in enumerate(with_bb):
        bb_a = a["geometry"]["bounding_box"]
        min_a = np.array(bb_a["min"])
        max_a = np.array(bb_a["max"])

        for b in with_bb[i + 1:]:
            bb_b = b["geometry"]["bounding_box"]
            min_b = np.array(bb_b["min"])
            max_b = np.array(bb_b["max"])

            gap = _aabb_gap(min_a, max_a, min_b, max_b)

            if gap <= threshold:
                rel_pos = _relative_position(min_a, max_a, min_b, max_b)
                edges.append(_edge(
                    a["global_id"], b["global_id"],
                    REL_ADJACENT,
                    {"gap_m": round(float(gap), 4), "relative_position": rel_pos},
                ))

    logger.info("Computed %d proximity edges (threshold=%.2fm)",
                len(edges), threshold)
    return edges


def _aabb_gap(min_a: np.ndarray, max_a: np.ndarray,
              min_b: np.ndarray, max_b: np.ndarray) -> float:
    """Axis-aligned bounding box gap (0 if overlapping)."""
    d = np.maximum(0.0, np.maximum(min_a - max_b, min_b - max_a))
    return float(np.linalg.norm(d))


def _relative_position(
    min_a: np.ndarray, max_a: np.ndarray,
    min_b: np.ndarray, max_b: np.ndarray,
) -> str:
    """Classify the approximate relative position of b with respect to a."""
    center_a = (min_a + max_a) / 2
    center_b = (min_b + max_b) / 2
    delta = center_b - center_a

    if abs(delta[2]) > max(abs(delta[0]), abs(delta[1])):
        return "above" if delta[2] > 0 else "below"
    if abs(delta[0]) > abs(delta[1]):
        return "east" if delta[0] > 0 else "west"
    return "north" if delta[1] > 0 else "south"


# ── Adjacency matrix / edge list helpers for ML ───────────────────────────────

def build_adjacency_structures(
    components: List[Dict],
    edges: List[Edge],
) -> Dict:
    """
    Build integer-indexed edge list and COO-format adjacency data for GNNs.
    """
    id_to_idx = {c["global_id"]: i for i, c in enumerate(components)}
    n = len(components)

    edge_index: List[List[int]] = [[], []]   # [src_indices, dst_indices]
    edge_types: List[str] = []
    edge_attrs: List[Dict] = []

    for e in edges:
        src = id_to_idx.get(e["source_id"])
        dst = id_to_idx.get(e["target_id"])
        if src is None or dst is None:
            continue
        edge_index[0].append(src)
        edge_index[1].append(dst)
        edge_types.append(e["relationship_type"])
        edge_attrs.append(e.get("attributes", {}))

    return {
        "num_nodes": n,
        "num_edges": len(edge_types),
        "edge_index": edge_index,       # PyG-compatible format
        "edge_types": edge_types,
        "edge_attributes": edge_attrs,
        "id_to_index": id_to_idx,
        "index_to_id": {v: k for k, v in id_to_idx.items()},
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _edge(
    source_id: str,
    target_id: str,
    rel_type: str,
    attributes: Optional[Dict] = None,
) -> Edge:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relationship_type": rel_type,
        "attributes": attributes or {},
    }
