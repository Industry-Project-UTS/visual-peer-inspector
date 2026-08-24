# System architecture

## Purpose

Visual Peer Inspector ranks unusual objects by comparing each object with the
other objects visible in the same scene. It uses multiple RGB views so that a
decision can be supported by more than one camera position.

The implementation has two related journeys:

1. a controlled research pipeline used to select and evaluate the method; and
2. a web application pipeline used to process new user images.

The research pipeline keeps labels and source metadata outside the model path.
The application pipeline replaces dataset references with user confirmation
when automatic object discovery or identity matching is uncertain.

## Research and evaluation pipeline

```mermaid
flowchart LR
    A["Trusted ToysAD preparation"] --> B["Opaque multi-view RGB package"]
    B --> C["LocateAnything object boxes"]
    B --> D["Supplied-mask reference route"]
    C --> E["SAM 2.1 automatic-box masks"]
    D --> F["Reference boxes"]
    F --> G["SAM 2.1 reference-box masks"]
    D --> H["Supplied masks"]
    E --> I["Common crop preparation"]
    G --> I
    H --> I
    I --> J["Frozen visual encoder"]
    J --> K["B0, B1, and P1 peer scores"]
    B --> L["Geometry features"]
    K --> M["Per-view object ranking"]
    L --> M
    M --> N["Cross-view identity and fusion"]
    N --> O["Ranked unusual objects and review state"]
    O --> P["Evaluator-only metrics and failure analysis"]
    Q["Protected references and labels"] --> P
```

The supplied-mask route is the scientific reference. The reference-box and
automatic-box routes measure how performance changes when prompted
segmentation and automatic object discovery are introduced.

## Web application pipeline

```mermaid
flowchart LR
    A["User uploads two to five related views"] --> B["LocateAnything proposes boxes"]
    B --> C["Box count and quality checks"]
    C --> D["User confirms or corrects boxes when required"]
    D --> E["SAM 2.1 produces whole-object masks"]
    E --> F["Object crops and frozen visual features"]
    F --> G["B0, B1, and P1 peer comparison"]
    G --> H["Cross-view identity and score fusion"]
    H --> I["Ranked unusual object and relative score"]
    I --> J["Highlighted views, coarse heatmap, and inspection report"]
```

The application must return a review state when boxes, object identity, view
support, or peer evidence are insufficient. It must not silently convert an
uncertain result into a confident anomaly claim.

## Component responsibilities

| Component | Responsibility | Not responsible for |
| --- | --- | --- |
| Trusted preparation | Selects scenes, enforces roles, resolves private source records, and builds opaque resources | Model inference |
| LocateAnything | Proposes object-level bounding boxes from RGB images | Segmentation, anomaly scoring, or identity tracking |
| SAM 2.1 | Produces a whole-object mask from a supplied box | Finding the defective region or scoring anomalies |
| Crop preparation | Applies the same masking, padding, and resizing rules to every route | Preserving full-scene geometry |
| Frozen visual encoder | Produces global and patch-level object features | Deciding which object is anomalous by itself |
| B0 | Compares one global feature with the leave-one-out peer mean | Local defect localisation |
| B1 | Aggregates pairwise global-feature distances with a median | Guaranteed improvement over B0 |
| P1 | Compares local patch features and produces a coarse mismatch map | Pixel-accurate defect segmentation |
| Geometry branch | Retains centroid, size, aspect-ratio, and relative-layout evidence | Visual appearance comparison |
| Identity and fusion | Links supported objects across views and combines per-view evidence | Inventing correspondence when evidence is weak |
| Evaluator | Joins frozen predictions with protected references and labels | Supplying information to model execution |

## Route names

Use these names consistently in notebooks, reports, and code:

| Route | Meaning |
| --- | --- |
| `supplied_mask` | Dataset-provided whole-object mask used as the scientific reference |
| `reference_box` | SAM 2.1 prompted with a box derived from the supplied mask |
| `automatic_box` | SAM 2.1 prompted with a frozen LocateAnything box |

The term *oracle* may be used when explaining the supplied reference, but
machine-readable route names must remain the three names above.

## Implementation map

| Area | Main repository location | Notebook stage |
| --- | --- | --- |
| Source verification and scene selection | `src/visual_peer_inspector/data/`, `chunks.py`, `verification.py` | 01 and 02 |
| Opaque model package | `src/visual_peer_inspector/data/model_safe_package.py` | 03 |
| Object proposals | `src/visual_peer_inspector/models/locateanything.py` | 04 |
| SAM prompts and inference | `src/visual_peer_inspector/data/sam_prompt_package.py`, `models/sam2.py`, `workflows/sam_ds0.py` | 05 |
| Protected proposal and mask evaluation | `src/visual_peer_inspector/evaluation/`, `workflows/mask_evaluation_ds0.py` | 06 |
| Representation and peer comparison | `src/visual_peer_inspector/baseline.py`, with the multi-view modules added in Notebook 07 | 07 |
| Reproducible entry points | `notebooks/` and `scripts/` | All stages |
| Verification | `tests/` and stage-specific requirement locks | All stages |

Notebook cells should coordinate these modules rather than contain a second
implementation. Report folders contain reviewed evidence, not reusable model
logic.

## Data zones and leakage controls

### Trusted preparation

Trusted preparation may read source paths, object order, labels, categories,
base-asset groups, protocol roles, and private crosswalks. It creates derived
resources with opaque identifiers and declared schemas.

### Model execution

Model jobs receive only the inputs required by their stage, such as opaque RGB
paths, boxes, masks, crops, or feature records. The following fields are
forbidden in model-facing resources:

- anomaly labels or anomaly counts;
- source object order or source indices;
- source names, paths, or private identifiers;
- category and base-asset information;
- development, calibration, internal-lock, or official-test roles; and
- evaluator references or protocol decisions.

### Evaluator-only analysis

Evaluator code verifies prediction hashes and configuration records before it
opens protected references. It reports paired aggregate metrics, subgroup
results, runtime, failures, and confidence intervals without returning labels
to the model path.

## Experiment gates

| Gate | Scope | Purpose | Current status |
| --- | --- | --- | --- |
| DS0 | 20 scenes, 20 views, 400 images | Validate packaging, proposals, segmentation, repeatability, and evaluator separation | Complete; candidate evidence supports progression |
| DS1 | 100 scenes, all 20 views | Screen proposal settings, mask routes, encoders, scoring, and geometry | Next |
| DS2 | Separate 300 scenes, all 20 views | Confirm the selected component settings without retuning | Planned |
| Development scale | Remaining development scenes | Build the selected feature cache and evaluate view policies | Planned |
| Calibration | 600 scenes | Freeze thresholds and review behaviour | Planned |
| Internal lock | 400 scenes | One evaluation after predictions and configuration are saved | Planned |

The optional 4,107-scene analysis is a post-freeze sensitivity check. It must
not be used to retune the primary 2,000-scene experiment.

## Current implementation status

Notebooks 01 to 06 are complete and merged. The codebase currently implements
data verification, leakage-safe selection, opaque packaging, LocateAnything
proposal generation, SAM 2.1 mask generation, proposal evaluation, mask
evaluation, safe evidence generation, and ClearML finalisation.

Notebook 07 is the next implementation stage. It will validate B0, B1, and P1
on the frozen mask routes, screen the bounded encoder set using supplied masks,
add the geometry branch, and create report-ready ranking and heatmap evidence.

## Claim limits

- ToysAD scenes contain at least one anomalous object, so the project does not
  establish false-alarm behaviour on all-normal scenes.
- Whole-object masks are not defect masks.
- P1 heatmaps indicate feature mismatch, not a causal or pixel-accurate defect.
- Twenty dataset views are repeated observations of one scene, not twenty
  independent samples.
- The automatic route is not the final application default until DS1, DS2, and
  downstream ranking checks are complete.
- The final encoder, identity method, fusion rule, and review threshold are not
  yet frozen.

## Evidence and reproducibility

GitHub stores code, tests, notebook sources, dependency locks, compact tables,
figures, hashes, and sanitized run records. Kaggle stores the private source
and large versioned model inputs or outputs. ClearML stores approved aggregate
metrics, plots, configuration, runtime, and provenance. Credentials and private
data remain outside all tracked files.
