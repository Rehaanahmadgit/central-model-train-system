# Checkpoint catalog

Shipped: `backend/app/checkpoints/default_catalog.yaml`; override: `$TRAINHUB_HOME/checkpoints/catalog.override.yaml`.
"Offered" = real URL + verified 64-hex sha256. Nothing is guessed; unknown = `TODO(user)`.

| Name | Source | sha256 | Size | Code licence | Weights licence | min_vram_gb | Offered |
|---|---|---|---|---|---|---|---|
| rtdetr-r18vd@coco-1.0 | github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth | `3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0` (file downloaded + hashed 2026-11-09) | 81001853 | Apache-2.0 | NOT FOUND (VERIFY), commercial use unknown | 0 (not measured) | yes |
| rtdetr-l@1.0 | TODO(user) | TODO(user) | — | Apache-2.0 | TODO(user) | 0 | no |
| resnet50@imagenet-1k | TODO(user) | TODO(user) | — | TODO(user) | TODO(user) | 0 | no |

Licence acceptance: `<home>/licences/accepted.json` stores `catalog_id`, `terms_sha256`, `accepted_at` only. Changed terms ask again.
Offline: `trainhub checkpoints import <file> --catalog-id <name>` (sha256 verified, not licence-gated).
