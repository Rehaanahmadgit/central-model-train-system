# Changelog

## Unreleased
- Slice D: the best checkpoint is picked for you (with the reason shown) and downloaded when the run starts, with progress; download problems (offline, bad hash, full disk, interrupted) give a clear message and an Import from file option. A frozen backbone with no pretrained weights is now blocked instead of silently training nothing.
- Slice C: model variants with a licence card; you must accept the weights licence before training (stored locally, no user info).
- Slice B: verification now fails zero-annotation detection data, scans every image for corruption, warns on tiny/out-of-range boxes and shows the fix hint on the row.
- A2: shared training engine (resume now remembers the best epoch it had just finished).
- Earlier: honest val/test evaluation, Predictions tab, weights download, Run again.
