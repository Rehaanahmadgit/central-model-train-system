"""Synthetic datasets for tests [doc 11 §2].

The implementation lives in ``app.samples.datasets`` so the DIST-11
"Try sample dataset" button can reuse exactly the same generator; this
module keeps the path the testing doc names and re-exports it.
"""

from app.samples.datasets import (  # noqa: F401 - re-exported as the doc-11 surface
    SPEC_MODEL_FAMILY,
    SPECS,
    Entries,
    build_classification,
    build_coco,
    build_empty,
    build_image_csv,
    build_sample_zip,
    build_tabular,
    build_yolo,
    grayscale_png_bytes,
    jpeg_bytes,
    materialize,
    png_bytes,
    spec_model_family,
    spec_names,
    to_zip,
)
