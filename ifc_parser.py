"""
ifc_parser.py
-------------
Core IFC parsing layer. Wraps IfcOpenShell to provide clean, typed access
to IFC entities, geometry, placements, and property sets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as ifc_util
    import ifcopenshell.util.placement as ifc_placement
    import ifcopenshell.util.shape as ifc_shape
except ImportError:
    raise ImportError(
        "IfcOpenShell is required. Install with: pip install ifcopenshell"
    )

logger = logging.getLogger(__name__)

# ── Entity types we care about ──────────────────────────────────────────────
ELEMENT_TYPES = [
    "IfcWall", "IfcWallStandardCase",
    "IfcDoor", "IfcWindow",
    "IfcSlab", "IfcRoof",
    "IfcBeam", "IfcColumn",
    "IfcStair", "IfcStairFlight",
    "IfcRailing",
    "IfcFurnishingElement",
    "IfcFlowSegment", "IfcFlowFitting",
    "IfcSpace",
    "IfcBuildingElementProxy",
    "IfcCurtainWall",
    "IfcPlate",
    "IfcMember",
    "IfcFooting",
    "IfcPile",
]

SPATIAL_TYPES = ["IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace"]


# ── Geometry settings ────────────────────────────────────────────────────────
def _make_geometry_settings() -> ifcopenshell.geom.settings:
    s = ifcopenshell.geom.settings()
    s.set(s.USE_WORLD_COORDS, False)   # keep local; we resolve manually
    s.set(s.WELD_VERTICES, True)
    s.set(s.GENERATE_UVS, False)
    return s


GEO_SETTINGS = _make_geometry_settings()


# ── Public helpers ────────────────────────────────────────────────────────────

def load_ifc(path: str | Path) -> ifcopenshell.file:
    """Load an IFC file and return the IfcOpenShell file object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IFC file not found: {path}")
    logger.info("Loading IFC file: %s", path)
    model = ifcopenshell.open(str(path))
    logger.info(
        "Loaded %s  |  schema: %s",
        path.name,
        model.schema,
    )
    return model


def get_all_elements(model: ifcopenshell.file) -> List[Any]:
    """Return every building element we want to process."""
    elements: List[Any] = []
    for etype in ELEMENT_TYPES:
        try:
            entities = model.by_type(etype)
            elements.extend(entities)
        except Exception:
            pass  # entity type not present in this schema version
    # deduplicate by id
    seen = set()
    unique = []
    for e in elements:
        if e.id() not in seen:
            seen.add(e.id())
            unique.append(e)
    return unique


# ── Placement resolution ──────────────────────────────────────────────────────

def get_global_placement(element: Any) -> np.ndarray:
    """
    Resolve the full placement hierarchy and return a 4×4 homogeneous
    transformation matrix in global (world) coordinates.
    """
    try:
        matrix = ifc_placement.get_local_placement(element.ObjectPlacement)
        return np.array(matrix, dtype=np.float64).reshape(4, 4)
    except Exception:
        return np.eye(4, dtype=np.float64)


def matrix_to_dict(m: np.ndarray) -> Dict:
    """Serialise a 4×4 numpy matrix to a plain dict."""
    return {"rows": m.tolist()}


def get_location(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Extract the translation (x, y, z) from a 4×4 matrix."""
    return float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])


def get_rotation_matrix(matrix: np.ndarray) -> List[List[float]]:
    """Extract the 3×3 rotation sub-matrix."""
    return matrix[:3, :3].tolist()


# ── Geometry extraction ───────────────────────────────────────────────────────

def get_geometry(element: Any) -> Optional[Dict]:
    """
    Attempt to tessellate the element and return vertices, faces,
    bounding box and dimensions.  Returns None if geometry is unavailable.
    """
    try:
        shape = ifcopenshell.geom.create_shape(GEO_SETTINGS, element)
    except Exception as exc:
        logger.debug("Geometry failed for %s: %s", element.GlobalId, exc)
        return None

    try:
        verts_flat = list(shape.geometry.verts)   # flat list x,y,z,...
        faces_flat = list(shape.geometry.faces)   # flat list i,j,k,...

        verts = np.array(verts_flat, dtype=np.float64).reshape(-1, 3)
        faces = np.array(faces_flat, dtype=np.int32).reshape(-1, 3)

        # ---- bounding box ----
        if len(verts):
            bb_min = verts.min(axis=0).tolist()
            bb_max = verts.max(axis=0).tolist()
            dims = (np.array(bb_max) - np.array(bb_min)).tolist()
        else:
            bb_min = bb_max = dims = [0.0, 0.0, 0.0]

        # ---- shape rep type ----
        rep_type = _get_rep_type(element)

        return {
            "vertices": verts.tolist(),
            "faces": faces.tolist(),
            "vertex_count": len(verts),
            "face_count": len(faces),
            "bounding_box": {"min": bb_min, "max": bb_max},
            "dimensions": {
                "length": dims[0],
                "width":  dims[1],
                "height": dims[2],
            },
            "representation_type": rep_type,
        }

    except Exception as exc:
        logger.debug("Geometry post-processing failed for %s: %s",
                     element.GlobalId, exc)
        return None


def _get_rep_type(element: Any) -> str:
    """Inspect the representation to identify its type (Extrusion, BRep…)."""
    try:
        for rep in element.Representation.Representations:
            for item in rep.Items:
                return item.is_a()
    except Exception:
        pass
    return "Unknown"


# ── Material extraction ───────────────────────────────────────────────────────

def get_materials(element: Any) -> List[str]:
    """Return a list of material names associated with the element."""
    materials: List[str] = []
    try:
        for rel in element.HasAssociations:
            if rel.is_a("IfcRelAssociatesMaterial"):
                mat = rel.RelatingMaterial
                if mat.is_a("IfcMaterial"):
                    materials.append(mat.Name)
                elif mat.is_a("IfcMaterialList"):
                    materials.extend(
                        m.Name for m in mat.Materials if m.Name
                    )
                elif mat.is_a("IfcMaterialLayerSetUsage"):
                    for layer in mat.ForLayerSet.MaterialLayers:
                        if layer.Material:
                            materials.append(layer.Material.Name)
                elif mat.is_a("IfcMaterialLayerSet"):
                    for layer in mat.MaterialLayers:
                        if layer.Material:
                            materials.append(layer.Material.Name)
    except Exception:
        pass
    return [m for m in materials if m]


# ── Property sets ─────────────────────────────────────────────────────────────

def get_property_sets(element: Any) -> Dict[str, Dict[str, Any]]:
    """
    Return all Psets as nested dicts:
    { "Pset_WallCommon": { "LoadBearing": True, ... }, ... }
    """
    psets: Dict[str, Dict[str, Any]] = {}
    try:
        for rel in element.IsDefinedBy:
            if rel.is_a("IfcRelDefinesByProperties"):
                pdef = rel.RelatingPropertyDefinition
                if pdef.is_a("IfcPropertySet"):
                    props: Dict[str, Any] = {}
                    for prop in pdef.HasProperties:
                        if prop.is_a("IfcPropertySingleValue"):
                            val = prop.NominalValue
                            props[prop.Name] = (
                                val.wrappedValue if val else None
                            )
                        elif prop.is_a("IfcPropertyEnumeratedValue"):
                            props[prop.Name] = [
                                v.wrappedValue
                                for v in prop.EnumerationValues or []
                            ]
                    psets[pdef.Name] = props
    except Exception:
        pass
    return psets


# ── Spatial storey ────────────────────────────────────────────────────────────

def get_storey(element: Any) -> Optional[Dict]:
    """Find the building storey that contains this element."""
    try:
        for rel in element.ContainedInStructure:
            container = rel.RelatingStructure
            if container.is_a("IfcBuildingStorey"):
                return {
                    "global_id": container.GlobalId,
                    "name": container.Name,
                    "elevation": (
                        float(container.Elevation)
                        if container.Elevation is not None
                        else None
                    ),
                }
            if container.is_a("IfcBuilding"):
                return {
                    "global_id": container.GlobalId,
                    "name": container.Name,
                    "elevation": None,
                }
    except Exception:
        pass
    return None


# ── Predefined type ───────────────────────────────────────────────────────────

def get_predefined_type(element: Any) -> Optional[str]:
    try:
        pt = element.PredefinedType
        return str(pt) if pt else None
    except Exception:
        return None


# ── Spatial hierarchy ─────────────────────────────────────────────────────────

def extract_spatial_hierarchy(model: ifcopenshell.file) -> Dict:
    """Build the Site → Building → Storey → Space tree."""
    hierarchy: Dict = {"sites": []}

    for site in model.by_type("IfcSite"):
        site_node = _spatial_node(site)
        site_node["buildings"] = []

        for rel in site.IsDecomposedBy:
            for building in rel.RelatedObjects:
                if not building.is_a("IfcBuilding"):
                    continue
                bldg_node = _spatial_node(building)
                bldg_node["storeys"] = []

                for rel2 in building.IsDecomposedBy:
                    for storey in rel2.RelatedObjects:
                        if not storey.is_a("IfcBuildingStorey"):
                            continue
                        storey_node = _spatial_node(storey)
                        storey_node["elevation"] = (
                            float(storey.Elevation)
                            if storey.Elevation is not None
                            else None
                        )
                        storey_node["spaces"] = []

                        for rel3 in storey.IsDecomposedBy:
                            for space in rel3.RelatedObjects:
                                if space.is_a("IfcSpace"):
                                    storey_node["spaces"].append(
                                        _spatial_node(space)
                                    )

                        # elements contained
                        storey_node["element_ids"] = _contained_ids(storey)
                        bldg_node["storeys"].append(storey_node)

                site_node["buildings"].append(bldg_node)

        hierarchy["sites"].append(site_node)

    return hierarchy


def _spatial_node(entity: Any) -> Dict:
    return {
        "global_id": entity.GlobalId,
        "entity_type": entity.is_a(),
        "name": entity.Name,
        "long_name": getattr(entity, "LongName", None),
    }


def _contained_ids(container: Any) -> List[str]:
    ids: List[str] = []
    try:
        for rel in container.ContainsElements:
            for el in rel.RelatedElements:
                ids.append(el.GlobalId)
    except Exception:
        pass
    return ids
