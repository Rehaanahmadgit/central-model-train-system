# 09 — UI / UX Specification

## 1. Design principles
- Wizard-first for the main flow; dashboard-first for monitoring.
- Show **why** (reasons for verification results and checkpoint choice), not just status.
- Never lose state: refresh/reconnect restores everything.
- Dense but readable: this is an engineering audit tool, not a marketing dashboard.
- Dark and light themes; keyboard friendly; responsive down to tablet.

## 2. Navigation
Sidebar: Projects · Datasets · Runs · Compare · Model Registry · Checkpoints · Audit Log · System. Top bar: search, notifications, theme toggle. **No login, no user menu, no profile** (if an access token is configured, show a single token prompt once).

## 3. Screens

### 3.1 Projects list / project home
Cards or table: name, last run, status counts. Project home tabs: Overview · Datasets · Runs · Models · Settings.

### 3.2 New Training Wizard (core experience)
Stepper: **1 Dataset → 2 Model → 3 Verify → 4 Configure → 5 Train**
1. **Dataset:** drag-and-drop upload (chunked, progress, pause/resume), or select existing dataset version. Shows size, file count, detected format after upload.
2. **Model:** searchable cards from `/models` (family, tasks, description). Selecting shows short info.
3. **Verify:** button **Start Verification**. Live checklist (spinner → ✓ / ⚠ / ✗) with expandable details, sample offending files, fix hints. Summary panel: classes, counts, splits, image stats, charts (class distribution bar). Recommended checkpoint card with reason + "cached locally / will download X MB". Result banner: "Dataset verified successfully" → **Next** button appears. If warnings: checkbox "I understand the warnings". If failed: **Next** disabled, "Re-run verification" after fix.
4. **Configure:** auto-generated form from `hparam_schema`, pre-filled with recommended values (badge "recommended"), Basic/Advanced toggle, checkpoint override dropdown + "train from scratch", run name, notifications.
5. **Train:** summary + **Start Training** button → navigates to Live Run page.

### 3.3 Live Run page (dashboard / audit tracker)
Header: run name, state badge, elapsed, ETA, progress bar (epoch x/y), buttons Pause/Resume/Stop.
Layout (grid, user can rearrange panels):
- **Metrics charts:** loss (train/val), primary metric, LR, others; zoom, smoothing slider, log scale, hover tooltips, compare-with-run overlay.
- **Live logs:** virtual-scrolled terminal view, filter by level, search, auto-scroll toggle, download.
- **System panel:** GPU util / memory / temp, CPU, RAM sparklines.
- **Run info panel (audit):** full resolved hyperparameters, dataset name+hash, verification link, checkpoint name+sha+source (cached/downloaded), seed, environment versions, hardware, timestamps.
- **Timeline / events:** checkpoint saved, LR change, early stop, pause/resume, warnings.
- **Checkpoints/Artifacts tab:** list with download.
- Connection indicator (live / reconnecting / polling).
After completion: tabs **Evaluation** (confusion matrix, per-class table, PR/ROC, error gallery, prediction samples) and **Export / Register**.

### 3.4 Runs list
Table with filters (state, model, project, date), sortable columns incl. best metric. Multi-select → Compare.

### 3.5 Compare
Side-by-side: config diff (highlight differences), metric overlays, final metric table, dataset/checkpoint differences.

### 3.6 Datasets
List with versions, hashes, verification status badges; dataset explorer (grid of images with label overlays, class filter, stats).

### 3.7 Checkpoints
Catalog table: name, task, pretrained-on, size, license, local availability, last used; actions: prefetch, evict, reload catalog, disk usage.

### 3.8 Model Registry
Registered models, versions, stage badges, export formats, linked run; promote/demote with confirmation.

### 3.9 Audit Log
Filterable immutable table (what, when, entity, details drawer); export CSV. Contains no personal data.

### 3.10 System
System health (running/queued jobs, GPU utilisation, disk and workspace size, recovery warnings), retention/cleanup actions, read-only config view.

## 4. Components checklist
Stepper, FileUploader (chunked), CheckList (streaming), StatCard, MetricChart (ECharts/Recharts, downsampled), LogViewer (virtualized), JsonSchemaForm, DiffViewer, StateBadge, Toasts, ConfirmDialog, EmptyState, Skeleton loaders, ErrorBoundary.

## 5. Frontend technical rules
- TypeScript strict; API client generated from OpenAPI.
- TanStack Query for REST caching; a `useRunStream(runId)` hook for WebSocket with auto-reconnect, `last_seq` resume, and fallback to polling.
- Keep chart data in a ring buffer; downsample on the server for history; throttle rendering (~4 fps).
- Accessibility: ARIA labels, contrast, no color-only status (use icons + text).
- i18n-ready strings.

## 6. Empty/error states
Every list has an empty state with a call-to-action. Errors show code + friendly message + "copy details" for support. Long operations show progress, never a frozen button.
