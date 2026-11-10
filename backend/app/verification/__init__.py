"""Verification pipeline [doc 06]. Verifier version participates in the
cache key [doc 06 §6: sha256(dataset_hash|model_family|verifier_version)]."""

# Bumped to 1.1.0 (2026-10-23): manifest ``boxes`` now carries the per-box
# class id as [class_id, cx, cy, w, h] (COCO normalized to YOLO) — S14b.
VERIFIER_VERSION = "1.3.0"
