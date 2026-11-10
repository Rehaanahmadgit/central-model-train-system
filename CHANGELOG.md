# Changelog

## Unreleased
- Slice C: model variants with a licence card; you must accept the weights licence before training (stored locally, no user info).
- Slice B: verification now fails zero-annotation detection data, scans every image for corruption, warns on tiny/out-of-range boxes and shows the fix hint on the row.
- A2: shared training engine (resume now remembers the best epoch it had just finished).
- Earlier: honest val/test evaluation, Predictions tab, weights download, Run again.
