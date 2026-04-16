# IFC → ML Dataset Pipeline

A production-quality Python pipeline that converts Industry Foundation Classes (IFC) building models into structured, graph-rich JSON datasets suitable for training generative AI models.

---

## Architecture

```
ifc_parser.py           ← Low-level IfcOpenShell wrappers
component_extractor.py  ← Per-element enrichment (geometry, psets, materials…)
relationship_extractor.py ← Graph edge extraction + proximity analysis
serializer.py           ← JSON / OBJ / CSV writers
ifc_reconstructor.py    ← Proof-of-reversibility IFC writer
pipeline.py             ← Orchestrator + CLI
demo.py                 ← Self-contained demo (generates + processes a building)
```

---

## Quick start

```bash
pip install ifcopenshell numpy

# Run on your own IFC file
python pipeline.py my_building.ifc --output ./output --reconstruct

```
Use pip3 & python3 on Mac

---

## Output artefacts

| File | Description |
|---|---|
| `dataset.json` | Complete dataset (components + edges + graph + hierarchy) |
| `summary.json` | Human-readable stats |
| `edges.csv` | Edge list (source_id, target_id, type) |
| `node_features.csv` | Numeric feature matrix for ML |
| `model.obj` | Merged mesh for visualisation |
| `components/<Type>/<id>.json` | One JSON per component |
| `reconstructed.ifc` | Rebuilt IFC from extracted data |

---

## Dataset Schema

### Top-level

```json
{
  "metadata":          { ... },
  "spatial_hierarchy": { "sites": [ ... ] },
  "components":        [ <Component>, ... ],
  "relationships":     [ <Edge>, ... ],
  "graph": {
    "num_nodes":       123,
    "num_edges":       456,
    "edge_index":      [[src...], [dst...]],
    "edge_types":      ["IfcRelContainedInSpatialStructure", ...],
    "edge_attributes": [ { ... }, ... ],
    "id_to_index":     { "GlobalId": 0, ... },
    "index_to_id":     { "0": "GlobalId", ... }
  }
}
```

### Component object

```json
{
  "global_id":       "2Zw9u...",
  "entity_type":     "IfcWall",
  "name":            "North Wall",
  "description":     null,
  "predefined_type": "SOLIDWALL",

  "geometry": {
    "vertices":           [[x, y, z], ...],
    "faces":              [[i, j, k], ...],
    "vertex_count":       48,
    "face_count":         32,
    "bounding_box":       { "min": [x,y,z], "max": [x,y,z] },
    "dimensions":         { "length": 10.0, "width": 0.2, "height": 3.0 },
    "representation_type":"IfcExtrudedAreaSolid"
  },

  "placement": {
    "global_matrix":  { "rows": [[...], ...] },
    "location":       { "x": 0.0, "y": 8.0, "z": 0.0 },
    "rotation_matrix":[[1,0,0],[0,1,0],[0,0,1]],
    "storey": {
      "global_id": "3Xk7...",
      "name":      "Ground Floor",
      "elevation": 0.0
    }
  },

  "materials":     ["Concrete", "Plaster"],
  "property_sets": {
    "Pset_WallCommon": {
      "LoadBearing": true,
      "IsExternal":  true,
      "FireRating":  "60"
    }
  },
  "attributes": {
    "load_bearing":          true,
    "fire_rating":           "60",
    "is_external":           true,
    "thermal_transmittance": null
  },

  "neighbours":      ["4Yz8...", "5Ab2..."],
  "adjacency_edges": [ { "source_id": "...", "target_id": "...",
                         "relationship_type": "proximity_adjacent",
                         "attributes": { "gap_m": 0.0 } } ],
  "node_index":      7,
  "embedding":       []
}
```

### Edge object

```json
{
  "source_id":        "2Zw9u...",
  "target_id":        "3Xk7...",
  "relationship_type":"IfcRelContainedInSpatialStructure",
  "attributes":       { "container_type": "IfcBuildingStorey" }
}
```

---

## Relationship types

| Type | Origin | Meaning |
|---|---|---|
| `IfcRelContainedInSpatialStructure` | Schema | Element belongs to storey/space |
| `IfcRelAggregates` | Schema | Part-of hierarchy |
| `IfcRelVoidsElement` | Schema | Opening cuts through wall |
| `IfcRelFillsElement` | Schema | Door/window fills opening |
| `IfcRelConnectsElements` | Schema | Structural connection |
| `IfcRelSpaceBoundary` | Schema | Element bounds a space |
| `proximity_adjacent` | Computed | AABB gap ≤ threshold metres |
