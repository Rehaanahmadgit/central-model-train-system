# Changelog

## Unreleased
- Slice G1: new "Try it" tab on detection runs: drop up to 8 of your own images (not saved on the server) or use 6 sample images, see the boxes, filter by confidence/class and download the annotated PNG. Runs on CPU with a note when the GPU is busy training.
- Slice H: the model list always comes from the app version you run (an old saved catalog no longer hides RT-DETR r18vd; the old file is kept as a backup). Models without a verified download can no longer be selected, and Next says why it is off. VRAM check only rejects when a real measurement exceeds your GPU. Starting a run no longer loads PyTorch inside the web server.
- Slice F: Configure now says which device will be used (GPU name/VRAM, or a clear "no GPU, running on CPU" warning); the System endpoint reports the real GPU. Re-checked: numbers stay numbers on Run again / Edit & run, Predictions default to confidence 0.5, finished runs show Best and GPU n/a.
- Slice E: access token is now on by default (generated on first boot, shown once in the logs, stored in `/data/access-token`; `trainhub --reset-token` to rotate). Uploads over the limit get a clear 413, and checkpoint downloads only use https from public hosts.
- Fix: an old `catalog.override.yaml` no longer wipes the licence info of the default model entries.
- Slice D: the best checkpoint is picked for you (with the reason shown) and downloaded when the run starts, with progress; download problems (offline, bad hash, full disk, interrupted) give a clear message and an Import from file option. A frozen backbone with no pretrained weights is now blocked instead of silently training nothing.
- Slice C: model variants with a licence card; you must accept the weights licence before training (stored locally, no user info).
- Slice B: verification now fails zero-annotation detection data, scans every image for corruption, warns on tiny/out-of-range boxes and shows the fix hint on the row.
- A2: shared training engine (resume now remembers the best epoch it had just finished).
- Earlier: honest val/test evaluation, Predictions tab, weights download, Run again.
