# 07 — Checkpoint (Pretrained Weights) Management

## 1. Goals
- Choose the **right** pretrained checkpoint automatically from dataset facts.
- Download **once**; reuse from local folder afterwards.
- Never train on a corrupt/partial file. Never block two jobs on duplicate downloads.

## 2. Registry
`checkpoints/catalog.yaml` (what exists, where to get it, expected sha256; maintained by hand in git) and `checkpoints/index.json` (what is cached locally: path, sha256, size, status, last_used_at). See doc 04. No database.
Seed catalog entries in a versioned YAML (`checkpoints/catalog.yaml`) loaded at startup and on `POST /checkpoints/reload`. Example:
```yaml
- name: rtdetr-l
  version: "1.0"
  model_family: rtdetr
  task_types: [object_detection]
  pretrained_on: coco
  source_type: url
  source_uri: https://example.com/rtdetr-l.pt
  sha256: "<hash>"
  size_bytes: 128000000
  license: "<license>"
- name: resnet50
  version: "imagenet-1k"
  model_family: resnet
  task_types: [image_classification]
  pretrained_on: imagenet
  ...
```
(Replace URLs/hashes with real, verified values. Never trust unverified hashes.)

## 3. Selection algorithm (rule-based v1)
Inputs: task_type, model_family, dataset size bucket (tiny <1k, small <10k, medium <100k, large), image size stats, num_classes, domain hints.
1. Filter catalog by `task_types` and `model_family` and `is_active`.
2. Score each candidate:
   - +5 exact task match, +3 family match.
   - +2 if pretrained domain closer (e.g., COCO for detection, ImageNet for classification).
   - Size fit: tiny/small data → prefer smaller or frozen-backbone friendly models; large → bigger variants.
   - −3 if license not allowed for org; −5 if requires more VRAM than available.
3. Return ranked list with **reasons** (strings) and recommended fine-tune strategy:
   - tiny data: freeze backbone, low LR, heavy augmentation
   - small/medium: partial unfreeze, discriminative LR
   - large: full fine-tune
4. Option `from_scratch` always available (with warning that accuracy may be lower).
5. User can override; override choice is logged in audit.
(v2: learn from past runs — store metric outcomes per (dataset traits, checkpoint) and rank by historical performance.)

## 4. Resolution at train time (`CheckpointResolver.resolve(catalog_id)`)
```
1. Look up `index.json` for the catalog id with status=ready.
2. If found and file exists: verify size (fast); verify sha256 if first use after restart or if mtime changed. Update last_used_at. Return path.
3. If not found/corrupt:
   a. Acquire the file lock `checkpoints/locks/{catalog_id}.lock` (`flock`; released automatically if the process dies).
   b. Re-check (another worker may have finished downloading).
   c. Check free disk >= size * 1.5.
   d. Download to `{dir}/.tmp/{name}.{uuid}.part` with resume (HTTP Range) + retries (exponential backoff, max 5).
   e. Verify sha256 against catalog. Mismatch → delete, error.
   f. Atomic rename to `{dir}/{sha256[:2]}/{sha256}/{filename}`.
   g. Update `index.json` atomically (status=ready).
   h. Release lock. Emit `checkpoint_download` events to the run's live log (progress %).
4. Return path.
```
Folder layout: `$TRAINHUB_HOME/checkpoints/store/{sha256[:2]}/{sha256}/model.ext` (content-addressed → dedup).

## 5. Rules
- Download only inside the resolver; trainers never download by themselves (disable framework auto-download or point it to the cache dir; set offline flags where possible).
- Safe loading: prefer safetensors / state_dict; **never** unpickle untrusted files. If `.pt` pickles are required (e.g., Ultralytics), only load from verified catalog entries with matching sha256.
- Eviction: LRU when disk usage > threshold, never evict a checkpoint used by a running/queued job.
- Prefetch (`POST /checkpoints/{id}/prefetch`) lets the first run start without waiting.
- Offline mode: if no network and not local → fail with actionable message.
- Run record stores `checkpoint_catalog_id` + sha256 used.

## 6. User-facing behaviour
- Verification result shows: "Recommended: RT-DETR-L (COCO). Reason: detection task, medium dataset. Status: cached locally" or "will download (128 MB) on first use".
- On Start Training the live log shows either `Using cached checkpoint (sha verified)` or download progress bar.

## 7. Edge cases
- Two jobs need same checkpoint simultaneously → one downloads, the other waits on the lock then reuses.
- Checkpoint class head mismatch with dataset classes → trainer re-initializes the head; verify via plugin hook `adapt_head()`.
- Input size mismatch → plugin resizes/interpolates positional embeddings where needed or warns.
- Catalog entry deactivated while runs exist → keep history; block new use.
