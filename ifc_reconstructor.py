"""
ifc_reconstructor.py
---------------------
Proof-of-reversibility module.
Reads the extracted JSON dataset and writes a new, valid IFC 2×3 file
containing the core building elements.

NOTE: This is a *faithful approximation*, not a lossless round-trip.
IFC stores rich NURBS / BRep geometry; the reconstructed file uses simple
IfcExtrudedAreaSolid boxes derived from the bounding-box dimensions, which
is sufficient to verify that spatial placement, hierarchy, and semantics
are preserved.

For full geometry round-trip you would persist the raw IfcOpenShell
geometry to STEP format and re-attach it — that extension point is
documented below.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import ifcopenshell
    import ifcopenshell.api
    import ifcopenshell.api.root
    import ifcopenshell.api.unit
    import ifcopenshell.api.context
    import ifcopenshell.api.project
    import ifcopenshell.api.spatial
    import ifcopenshell.api.element
    import ifcopenshell.api.geometry
    import ifcopenshell.api.material
    import ifcopenshell.api.type
except ImportError:
    ifcopenshell = None  # type: ignore


# ── Entity type → IFC class mapping ──────────────────────────────────────────

_ENTITY_MAP = {
    "IfcWall":                  "IfcWall",
    "IfcWallStandardCase":      "IfcWall",
    "IfcDoor":                  "IfcDoor",
    "IfcWindow":                "IfcWindow",
    "IfcSlab":                  "IfcSlab",
    "IfcRoof":                  "IfcRoof",
    "IfcBeam":                  "IfcBeam",
    "IfcColumn":                "IfcColumn",
    "IfcStair":                 "IfcStair",
    "IfcStairFlight":           "IfcStairFlight",
    "IfcRailing":               "IfcRailing",
    "IfcFurnishingElement":     "IfcFurnishingElement",
    "IfcSpace":                 "IfcSpace",
    "IfcBuildingElementProxy":  "IfcBuildingElementProxy",
    "IfcCurtainWall":           "IfcCurtainWall",
    "IfcPlate":                 "IfcPlate",
    "IfcMember":                "IfcMember",
    "IfcFooting":               "IfcFooting",
}


# ── Public API ────────────────────────────────────────────────────────────────

def reconstruct_ifc(
    dataset: Dict,
    output_path: str | Path,
) -> Optional[Path]:
    """
    Build a new IFC file from the dataset dict and save it to *output_path*.
    Returns the output path on success, None on failure.
    """
    if ifcopenshell is None:
        logger.error("IfcOpenShell not available – cannot reconstruct IFC")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model = _build_model(dataset)
        model.write(str(output_path))
        logger.info("Reconstructed IFC written → %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("IFC reconstruction failed: %s", exc, exc_info=True)
        return None


# ── Internal builder ──────────────────────────────────────────────────────────

def _build_model(dataset: Dict) -> "ifcopenshell.file":
    model = ifcopenshell.file(schema="IFC2X3")

    # ── Project boilerplate ──────────────────────────────────────────────────
    project = ifcopenshell.api.run("root.create_entity", model,
                                    ifc_class="IfcProject", name="Reconstructed")
    ifcopenshell.api.run("unit.assign_unit", model)

    context = ifcopenshell.api.run("context.add_context", model,
                                    context_type="Model")
    body = ifcopenshell.api.run("context.add_context", model,
                                 context_type="Model",
                                 context_identifier="Body",
                                 target_view="MODEL_VIEW",
                                 parent=context)

    # ── Spatial hierarchy ────────────────────────────────────────────────────
    site     = ifcopenshell.api.run("root.create_entity", model,
                                     ifc_class="IfcSite", name="Default Site")
    building = ifcopenshell.api.run("root.create_entity", model,
                                     ifc_class="IfcBuilding", name="Default Building")

    ifcopenshell.api.run("aggregate.assign_object", model,
                          product=building, relating_object=site)
    ifcopenshell.api.run("aggregate.assign_object", model,
                          product=site, relating_object=project)

    # Collect storeys from dataset hierarchy
    storey_map: Dict[str, Any] = {}   # global_id → IfcBuildingStorey
    hierarchy  = dataset.get("spatial_hierarchy", {})

    storey_defs = _collect_storeys(hierarchy)
    if not storey_defs:
        # fallback: single storey
        storey_defs = [{"global_id": "DEFAULT_STOREY", "name": "Level 0",
                        "elevation": 0.0}]

    for sdef in storey_defs:
        storey = ifcopenshell.api.run(
            "root.create_entity", model,
            ifc_class="IfcBuildingStorey",
            name=sdef.get("name") or "Storey",
        )
        if sdef.get("elevation") is not None:
            storey.Elevation = float(sdef["elevation"])
        ifcopenshell.api.run("aggregate.assign_object", model,
                              product=storey, relating_object=building)
        storey_map[sdef["global_id"]] = storey

    default_storey = list(storey_map.values())[0]

    # ── Elements ─────────────────────────────────────────────────────────────
    components = dataset.get("components", [])
    logger.info("Reconstructing %d components …", len(components))

    for comp in components:
        try:
            _create_element(model, comp, body, storey_map, default_storey)
        except Exception as exc:
            logger.debug("Skip element %s: %s", comp.get("global_id"), exc)

    return model


def _create_element(
    model: "ifcopenshell.file",
    comp: Dict,
    body_context: Any,
    storey_map: Dict[str, Any],
    default_storey: Any,
) -> None:
    entity_type = comp.get("entity_type", "IfcBuildingElementProxy")
    ifc_class   = _ENTITY_MAP.get(entity_type, "IfcBuildingElementProxy")

    # Create element
    el = ifcopenshell.api.run(
        "root.create_entity", model,
        ifc_class=ifc_class,
        name=comp.get("name") or comp.get("global_id"),
    )

    # ── Placement ────────────────────────────────────────────────────────────
    loc = (comp.get("placement") or {}).get("location") or {}
    x = float(loc.get("x") or 0.0)
    y = float(loc.get("y") or 0.0)
    z = float(loc.get("z") or 0.0)

    ifcopenshell.api.run(
        "geometry.edit_object_placement", model,
        product=el,
        matrix=_translation_matrix(x, y, z),
    )

    # ── Box geometry from bounding box ────────────────────────────────────────
    geo  = comp.get("geometry") or {}
    dims = geo.get("dimensions") or {}
    length = max(float(dims.get("length") or 0.1), 0.01)
    width  = max(float(dims.get("width")  or 0.1), 0.01)
    height = max(float(dims.get("height") or 0.1), 0.01)

    representation = ifcopenshell.api.run(
        "geometry.add_box_representation", model,
        context=body_context,
        x=length,
        y=width,
        z=height,
    )
    ifcopenshell.api.run(
        "geometry.assign_representation", model,
        product=el,
        representation=representation,
    )

    # ── Assign to storey ─────────────────────────────────────────────────────
    storey_info = (comp.get("placement") or {}).get("storey") or {}
    storey_gid  = storey_info.get("global_id")
    storey      = storey_map.get(storey_gid, default_storey)

    ifcopenshell.api.run(
        "spatial.assign_container", model,
        products=[el],
        relating_structure=storey,
    )


def _translation_matrix(x: float, y: float, z: float):
    """Return a 4×4 translation matrix as a nested list (IfcOpenShell format)."""
    import numpy as np
    m = np.eye(4)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def _collect_storeys(hierarchy: Dict) -> List[Dict]:
    storeys = []
    for site in hierarchy.get("sites", []):
        for bldg in site.get("buildings", []):
            for storey in bldg.get("storeys", []):
                storeys.append(storey)
    return storeys
