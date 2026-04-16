"""
serializer.py
-------------
Handles all output serialisation:
 - Full dataset JSON
 - Per-component JSON files (for individual training examples)
 - OBJ export (visualization hook)
 - Lightweight adjacency CSV (for quick graph inspection)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Custom JSON encoder to handle numpy types ─────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            return None if math.isnan(f) or math.isinf(f) else f
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dump(obj: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, cls=_NumpyEncoder, indent=indent, ensure_ascii=False)


# ── Full dataset ──────────────────────────────────────────────────────────────

def write_full_dataset(
    dataset: Dict,
    output_dir: str | Path,
    filename: str = "dataset.json",
) -> Path:
    """
    Write the complete extracted dataset to a single JSON file.
    """
    out = Path(output_dir) / filename
    _dump(dataset, out)
    size_mb = out.stat().st_size / 1_048_576
    logger.info("Full dataset written → %s  (%.2f MB)", out, size_mb)
    return out


# ── Split into per-component files ────────────────────────────────────────────

def write_component_files(
    components: List[Dict],
    output_dir: str | Path,
    subdir: str = "components",
) -> List[Path]:
    """
    Write one JSON file per component.  Files are organised by entity type.
    """
    base = Path(output_dir) / subdir
    paths: List[Path] = []

    for comp in components:
        etype = comp.get("entity_type", "Unknown").replace("Ifc", "")
        gid   = comp.get("global_id", "unknown")
        dest  = base / etype / f"{gid}.json"
        _dump(comp, dest, indent=None)   # compact for speed
        paths.append(dest)

    logger.info("Wrote %d component files → %s", len(paths), base)
    return paths


# ── OBJ export (visualisation hook) ──────────────────────────────────────────

def export_obj(
    components: List[Dict],
    output_dir: str | Path,
    filename: str = "model.obj",
) -> Optional[Path]:
    """
    Export all component meshes to a single Wavefront OBJ file.
    Each component is written as a named group.  Skips components without
    geometry.

    The OBJ uses the local coordinate system from IfcOpenShell (component
    space).  Apply the placement matrices in a viewer for world coordinates.
    """
    out = Path(output_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    vertex_offset = 0  # OBJ indices are 1-based and global

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# IFC Building Dataset – OBJ export\n")
        fh.write(f"# Components: {len(components)}\n\n")

        for comp in components:
            geo = comp.get("geometry")
            if not geo or not geo.get("vertices") or not geo.get("faces"):
                continue

            gid   = comp["global_id"]
            name  = (comp.get("name") or gid).replace(" ", "_")
            etype = comp.get("entity_type", "Unknown")

            fh.write(f"\ng {etype}_{name}\n")

            verts = geo["vertices"]
            faces = geo["faces"]

            for v in verts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            for f in faces:
                i, j, k = (
                    f[0] + vertex_offset + 1,
                    f[1] + vertex_offset + 1,
                    f[2] + vertex_offset + 1,
                )
                fh.write(f"f {i} {j} {k}\n")

            vertex_offset += len(verts)

    size_kb = out.stat().st_size / 1024
    logger.info("OBJ export written → %s  (%.1f KB)", out, size_kb)
    return out


# ── Adjacency CSV ─────────────────────────────────────────────────────────────

def write_edge_list_csv(
    edges: List[Dict],
    output_dir: str | Path,
    filename: str = "edges.csv",
) -> Path:
    """Write a simple CSV edge list for quick inspection in spreadsheet tools."""
    out = Path(output_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["source_id", "target_id", "relationship_type"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(edges)

    logger.info("Edge list written → %s  (%d rows)", out, len(edges))
    return out


# ── Node feature matrix CSV ───────────────────────────────────────────────────

def write_node_features_csv(
    components: List[Dict],
    output_dir: str | Path,
    filename: str = "node_features.csv",
) -> Path:
    """
    Write a CSV of numeric node features suitable for immediate use in sklearn /
    PyTorch Geometric without re-parsing the full JSON.
    """
    out = Path(output_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for comp in components:
        geo     = comp.get("geometry") or {}
        bb      = geo.get("bounding_box") or {}
        dims    = geo.get("dimensions") or {}
        loc     = (comp.get("placement") or {}).get("location") or {}
        attrs   = comp.get("attributes") or {}

        bb_min  = bb.get("min") or [None, None, None]
        bb_max  = bb.get("max") or [None, None, None]

        rows.append({
            "global_id":      comp["global_id"],
            "node_index":     comp.get("node_index"),
            "entity_type":    comp["entity_type"],
            "x":              loc.get("x"),
            "y":              loc.get("y"),
            "z":              loc.get("z"),
            "bb_min_x":       bb_min[0],
            "bb_min_y":       bb_min[1],
            "bb_min_z":       bb_min[2],
            "bb_max_x":       bb_max[0],
            "bb_max_y":       bb_max[1],
            "bb_max_z":       bb_max[2],
            "dim_length":     dims.get("length"),
            "dim_width":      dims.get("width"),
            "dim_height":     dims.get("height"),
            "vertex_count":   geo.get("vertex_count"),
            "neighbour_count": len(comp.get("neighbours") or []),
            "load_bearing":   attrs.get("load_bearing"),
            "is_external":    attrs.get("is_external"),
        })

    if rows:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Node features CSV written → %s  (%d rows)", out, len(rows))
    return out


# ── Summary report ─────────────────────────────────────────────────────────────

def write_summary(
    dataset: Dict,
    output_dir: str | Path,
    filename: str = "summary.json",
) -> Path:
    """Write a compact human-readable summary of the dataset."""
    components   = dataset.get("components", [])
    relationships = dataset.get("relationships", [])

    type_counts: Dict[str, int] = {}
    for c in components:
        t = c.get("entity_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts: Dict[str, int] = {}
    for r in relationships:
        t = r.get("relationship_type", "Unknown")
        rel_counts[t] = rel_counts.get(t, 0) + 1

    geom_count = sum(1 for c in components if c.get("geometry"))

    summary = {
        "total_components":          len(components),
        "components_with_geometry":  geom_count,
        "total_relationships":       len(relationships),
        "entity_type_counts":        type_counts,
        "relationship_type_counts":  rel_counts,
        "metadata":                  dataset.get("metadata", {}),
    }

    out = Path(output_dir) / filename
    _dump(summary, out)
    logger.info("Summary written → %s", out)
    return out
