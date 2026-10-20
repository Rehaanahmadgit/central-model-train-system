# 06 — Verification Pipeline

## 1. Purpose
Before any training: prove the dataset is usable, extract facts needed later (so downstream stages never recompute them), detect the task, and decide the best pretrained checkpoint. Outcome: `passed`, `passed_with_warnings`, `failed`.

## 2. Pipeline structure
```
Input: dataset_hash, model_family, (optional) task_type
  1. Fingerprint + cache lookup  → if hit (same hash + verifier_version + model_family) return cached report
  2. Unpack + manifest build
  3. Format detection
  4. Task detection
  5. Structural checks
  6. Content checks
  7. Statistical checks
  8. Model compatibility checks
  9. Checkpoint recommendation
 10. Persist report.json + manifest.parquet (atomic writes) + write cache pointer file
```
Each check is a class implementing `Check.run(ctx) -> CheckResult` with `name`, `severity`, `status`, `message`, `details`, `fix_hint`. Checks declare dependencies and are run by a runner that streams progress events. A failing `error`-severity check can short-circuit dependent checks (mark them `skipped`).

## 3. Format detection
Detect and normalize to an internal `DatasetManifest`:
- Image classification: folder-per-class, CSV (path,label).
- Object detection: YOLO txt (+ data.yaml), COCO json, Pascal VOC xml.
- Segmentation: masks folder, COCO polygons.
- Tabular: CSV/Parquet/Excel with target column.
If ambiguous, report candidates and ask user to choose (UI prompt) — do not guess silently.

## 4. Checks catalog

### Structural (P0)
| Check | Severity | Detail |
|---|---|---|
| archive_integrity | error | zip/tar readable, no path traversal (`../`), size limits, zip-bomb guard (compression ratio) |
| format_supported | error | format recognized and compatible with selected model |
| required_files_present | error | labels/yaml/annotation file exist |
| file_extensions | warn | unsupported files ignored/listed |
| split_presence | warn | train/val/test exist; else auto-split suggestion (stratified) |

### Content (P0)
| Check | Severity | Detail |
|---|---|---|
| corrupt_files | error/warn | images that fail decode; zero-byte files; list first N |
| label_file_pairing | error | every image has label (if required) and vice versa |
| label_validity | error | class ids in range, no NaN, bbox in [0,1] (YOLO) or inside image (COCO/VOC), no zero-area boxes |
| empty_labels | warn | images with no annotations (could be background; show %) |
| duplicate_files | warn | exact duplicates by hash; near-duplicates (perceptual hash) optional |
| leakage_between_splits | error/warn | same hash/pHash in train and val/test |
| class_names_consistent | error | names/ids mapping consistent across files |

### Statistical (P0/P1)
| Check | Severity | Detail |
|---|---|---|
| num_classes | info | count and names; error if < 2 for classification |
| class_distribution | warn | imbalance ratio threshold (e.g. min class < 2% or ratio > 20:1) with suggestion (weighted loss, resampling, augmentation) |
| min_samples_per_class | warn/error | e.g. < 10 → error, < 50 → warn |
| dataset_size | info/warn | tiny dataset → recommend freezing backbone / strong augmentation |
| image_resolution_stats | info/warn | min/max/median, odd aspect ratios, very small images |
| object_size_stats | info/warn | detection: many tiny objects → recommend larger input size |
| color_mode | info | grayscale vs RGB vs RGBA consistency |
| tabular_quality | warn | missing values %, constant columns, high cardinality, target leakage heuristics, duplicated rows |

### Model compatibility (P0)
| Check | Detail |
|---|---|
| task_model_match | selected model supports detected task |
| input_requirements | e.g. min image size, channels, number of classes limits |
| resource_estimate | estimate VRAM/time; warn if exceeds available hardware |

## 5. Task detection rules (heuristic, explainable)
- Annotation has boxes → object_detection; polygons/masks → segmentation; folder-per-class or single label column on images → image_classification; tabular file with target → tabular_classification/regression (decide by target dtype/cardinality).
- Store `task_detection_reason` string in report.

## 6. Derived facts to cache (consumed by training, never recomputed)
```json
{
  "task_type": "...", "format": "...",
  "num_classes": 5, "class_names": [], "class_to_id": {}, "class_counts": {},
  "num_samples": {"train":0,"val":0,"test":0},
  "image_stats": {}, "object_stats": {}, "tabular_schema": {},
  "split_manifest_path": "datasets/{hash}/manifest.parquet",   // normalized file list + labels + split
  "recommended_hparams": {"imgsz": 640, "batch_size": 16, "freeze_backbone": true},
  "recommended_checkpoints": [],
  "dataset_hash": "...", "verifier_version": "1.0.0"
}
```
Cache file: `cache/verif/{sha256(dataset_hash|model_family|verifier_version)}.json` pointing to the stored `verifications/{id}/report.json`. No database or Redis. The **normalized manifest** is the key artifact: training reads the manifest, not raw folders, so every trainer plugin sees one consistent format.

## 7. Performance rules
- Stream files; never load a whole dataset in memory.
- Parallelize image decoding with a process pool; cap with sampling for huge datasets (full check for structure, sampled check for pixels, clearly reported as sampled).
- Emit progress events at least every 2 seconds.
- Hard timeouts per check.

## 8. UI contract
Each check streams: `{name, status, message, progress}`. UI shows checklist with spinner → tick/warn/cross. Failed checks show `fix_hint` and sample offending files. "Start Training" button enabled only when `can_train == true`. Warnings require explicit acknowledgement checkbox.

## 9. Staleness
If dataset version changes, or `verifier_version` is bumped with breaking changes, the verification is stale; `POST /runs` must reject with 409 `VERIFICATION_STALE`.
