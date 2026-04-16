"""
component_extractor.py
-----------------------
Orchestrates per-element extraction: geometry, placement, materials,
property sets, storey, and contextual neighbours.
Produces the canonical component dict used throughout the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ifc_parser import (
    get_geometry,
    get_global_placement,
    get_location,
    get_materials,
    get_predefined_type,
    get_property_sets,
    get_rotation_matrix,
    get_storey,
    matrix_to_dict,
)

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_component(element: Any) -> Optional[Dict]:
    """
    Extract a full component dict for a single IFC element.
    Returns None if the element has no usable data.
    """
    try:
        return _extract(element)
    except Exception as exc:
        logger.warning(
            "Failed to extract component %s (%s): %s",
            getattr(element, "GlobalId", "?"),
            element.is_a(),
            exc,
            exc_info=True,
        )
        return None


def extract_all_components(elements: List[Any]) -> List[Dict]:
    """
    Extract all components in *elements*.  Failures are logged and skipped.
    """
    results: List[Dict] = []
    total = len(elements)
    for i, el in enumerate(elements, 1):
        comp = extract_component(el)
        if comp:
            results.append(comp)
        if i % 100 == 0 or i == total:
            logger.info("Extracted %d / %d elements", i, total)
    logger.info("Total components extracted: %d", len(results))
    return results


# ── Internal ──────────────────────────────────────────────────────────────────

def _extract(element: Any) -> Dict:
    # ---- A. Identifiers ----
    global_id    = element.GlobalId
    entity_type  = element.is_a()
    name         = element.Name or ""
    description  = getattr(element, "Description", None)
    pred_type    = get_predefined_type(element)

    # ---- B. Geometry ----
    geometry = get_geometry(element)

    # ---- C. Spatial Placement ----
    matrix = get_global_placement(element)
    loc    = get_location(matrix)
    rot    = get_rotation_matrix(matrix)

    placement = {
        "global_matrix": matrix_to_dict(matrix),
        "location":      {"x": loc[0], "y": loc[1], "z": loc[2]},
        "rotation_matrix": rot,
        "storey": get_storey(element),
    }

    # ---- D. Materials ----
    materials = get_materials(element)

    # ---- E. Property Sets ----
    psets = get_property_sets(element)

    # ---- Derived from Psets ----
    load_bearing   = _find_pset_value(psets, "LoadBearing")
    fire_rating    = _find_pset_value(psets, "FireRating")
    is_external    = _find_pset_value(psets, "IsExternal")
    thermal_transm = _find_pset_value(psets, "ThermalTransmittance")

    # ---- F. Feature vector placeholder ----
    # These will be populated later by a feature-engineering step.
    embedding_placeholder: List[Optional[float]] = []

    component = {
        # ── Identifiers ──────────────────────────────────────────────────────
        "global_id":        global_id,
        "entity_type":      entity_type,
        "name":             name,
        "description":      description,
        "predefined_type":  pred_type,

        # ── Geometry ─────────────────────────────────────────────────────────
        "geometry": geometry,

        # ── Placement ────────────────────────────────────────────────────────
        "placement": placement,

        # ── Semantics ────────────────────────────────────────────────────────
        "materials":     materials,
        "property_sets": psets,
        "attributes": {
            "load_bearing":          load_bearing,
            "fire_rating":           fire_rating,
            "is_external":           is_external,
            "thermal_transmittance": thermal_transm,
        },

        # ── Graph placeholders (filled in by relationship_extractor) ─────────
        "neighbours":         [],   # [global_id, ...]
        "adjacency_edges":    [],   # subset of edges touching this node

        # ── ML hooks ─────────────────────────────────────────────────────────
        "node_index":         None,  # integer index in the graph
        "embedding":          embedding_placeholder,
    }

    return component


def _find_pset_value(psets: Dict, key: str):
    """Search all property sets for a given property name."""
    for props in psets.values():
        if key in props:
            return props[key]
    return None


# ── Post-processing: attach adjacency data to components ─────────────────────

def attach_graph_data(
    components: List[Dict],
    edges: List[Dict],
    adjacency_structures: Dict,
) -> None:
    """
    Mutate each component in-place to add:
    - node_index
    - neighbours (list of adjacent global_ids)
    - adjacency_edges (edges touching this node)
    """
    id_to_idx = adjacency_structures["id_to_index"]
    idx_to_id = adjacency_structures["index_to_id"]
    edge_index = adjacency_structures["edge_index"]

    # Build per-node neighbour lookup
    neighbour_map: Dict[str, set] = {c["global_id"]: set() for c in components}
    edge_map: Dict[str, List[Dict]] = {c["global_id"]: [] for c in components}

    for e in edges:
        src, dst = e["source_id"], e["target_id"]
        if src in neighbour_map:
            neighbour_map[src].add(dst)
            edge_map[src].append(e)
        if dst in neighbour_map:
            neighbour_map[dst].add(src)
            edge_map[dst].append(e)

    for comp in components:
        gid = comp["global_id"]
        comp["node_index"]      = id_to_idx.get(gid)
        comp["neighbours"]      = list(neighbour_map.get(gid, []))
        comp["adjacency_edges"] = edge_map.get(gid, [])
