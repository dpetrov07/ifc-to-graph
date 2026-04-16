"""
pipeline.py
-----------
Top-level orchestrator.  Wires together all pipeline stages:

  1. Parse IFC
  2. Extract spatial hierarchy
  3. Extract all component data
  4. Extract relationships
  5. Compute proximity edges
  6. Build graph adjacency structures
  7. Attach graph data to components
  8. Serialize outputs

Usage:
    python pipeline.py path/to/model.ifc --output ./output --obj --split --reconstruct
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_pipeline(
    ifc_path: str | Path,
    output_dir: str | Path,
    proximity_threshold: float = 1.0,
    export_obj: bool = True,
    split_components: bool = True,
    reconstruct: bool = False,
) -> Dict:
    """
    Run the full extraction pipeline on a single IFC file.

    Parameters
    ----------
    ifc_path            : Path to the input IFC file.
    output_dir          : Root directory for all output artefacts.
    proximity_threshold : Distance (metres) for synthetic adjacency edges.
    export_obj          : Write a merged OBJ file for quick visualisation.
    split_components    : Write one JSON per component.
    reconstruct         : Attempt to write a reconstructed IFC file.

    Returns
    -------
    dataset dict (mirrors what is written to dataset.json)
    """
    t0 = time.time()
    ifc_path   = Path(ifc_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 1 — Parse IFC")
    from ifc_parser import (
        load_ifc,
        get_all_elements,
        extract_spatial_hierarchy,
    )

    model    = load_ifc(ifc_path)
    elements = get_all_elements(model)
    logger.info("Found %d elements to process", len(elements))

    # ── 2. Spatial hierarchy ──────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 2 — Spatial hierarchy")
    hierarchy = extract_spatial_hierarchy(model)

    # ── 3. Extract components ─────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 3 — Component extraction")
    from component_extractor import extract_all_components
    components = extract_all_components(elements)

    # ── 4. Relationships ──────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 4 — Relationship extraction")
    from relationship_extractor import (
        extract_all_relationships,
        compute_proximity_edges,
        build_adjacency_structures,
    )

    schema_edges    = extract_all_relationships(model, elements)
    proximity_edges = compute_proximity_edges(components, proximity_threshold)
    all_edges       = schema_edges + proximity_edges

    # ── 5. Graph structures ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 5 — Build graph adjacency structures")
    adjacency = build_adjacency_structures(components, all_edges)

    # ── 6. Attach graph data to components ────────────────────────────────────
    logger.info("STAGE 6 — Attach graph data")
    from component_extractor import attach_graph_data
    attach_graph_data(components, all_edges, adjacency)

    # ── 7. Assemble dataset ───────────────────────────────────────────────────
    meta = _build_metadata(ifc_path, model, components, all_edges)

    dataset: Dict = {
        "metadata":          meta,
        "spatial_hierarchy": hierarchy,
        "components":        components,
        "relationships":     all_edges,
        "graph": {
            "num_nodes":        adjacency["num_nodes"],
            "num_edges":        adjacency["num_edges"],
            "edge_index":       adjacency["edge_index"],
            "edge_types":       adjacency["edge_types"],
            "edge_attributes":  adjacency["edge_attributes"],
            "id_to_index":      adjacency["id_to_index"],
            "index_to_id":      adjacency["index_to_id"],
        },
    }

    # ── 8. Serialize ──────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STAGE 7 — Serialize outputs")
    from serializer import (
        write_full_dataset,
        write_component_files,
        write_edge_list_csv,
        write_node_features_csv,
        write_summary,
        export_obj as _export_obj,
    )

    write_full_dataset(dataset, output_dir)
    write_edge_list_csv(all_edges, output_dir)
    write_node_features_csv(components, output_dir)
    write_summary(dataset, output_dir)

    if split_components:
        write_component_files(components, output_dir)

    if export_obj:
        _export_obj(components, output_dir)

    # ── 9. (Optional) Reconstruct IFC ─────────────────────────────────────────
    if reconstruct:
        logger.info("═" * 60)
        logger.info("STAGE 8 — Reconstruct IFC (proof of reversibility)")
        from ifc_reconstructor import reconstruct_ifc
        rpath = output_dir / "reconstructed.ifc"
        reconstruct_ifc(dataset, rpath)

    elapsed = time.time() - t0
    logger.info("═" * 60)
    logger.info(
        "Pipeline complete in %.1fs  |  components=%d  edges=%d",
        elapsed, len(components), len(all_edges),
    )
    return dataset


# ── Metadata helper ───────────────────────────────────────────────────────────

def _build_metadata(ifc_path, model, components, edges) -> Dict:
    from collections import Counter
    type_counts = dict(Counter(c["entity_type"] for c in components))
    rel_counts  = dict(Counter(e["relationship_type"] for e in edges))

    return {
        "source_file":        ifc_path.name,
        "ifc_schema":         model.schema,
        "extracted_at":       datetime.datetime.utcnow().isoformat() + "Z",
        "pipeline_version":   "1.0.0",
        "total_components":   len(components),
        "total_edges":        len(edges),
        "entity_type_counts": type_counts,
        "relationship_type_counts": rel_counts,
        "units":              "metres",
        "coordinate_system":  "IFC world coordinates (right-handed, Z-up)",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="IFC → ML Dataset pipeline"
    )
    parser.add_argument("ifc_file", help="Path to the input .ifc file")
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=1.0,
        help="Proximity threshold in metres for adjacency edges (default: 1.0)",
    )
    parser.add_argument(
        "--no-obj", dest="obj", action="store_false",
        help="Skip OBJ export",
    )
    parser.add_argument(
        "--no-split", dest="split", action="store_false",
        help="Skip per-component JSON files",
    )
    parser.add_argument(
        "--reconstruct", "-r", action="store_true",
        help="Attempt IFC reconstruction from extracted data",
    )
    parser.set_defaults(obj=True, split=True)

    args = parser.parse_args()

    run_pipeline(
        ifc_path            = args.ifc_file,
        output_dir          = args.output,
        proximity_threshold = args.threshold,
        export_obj          = args.obj,
        split_components    = args.split,
        reconstruct         = args.reconstruct,
    )


if __name__ == "__main__":
    _cli()
