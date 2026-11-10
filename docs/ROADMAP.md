# Roadmap (build slices)

Original phase plan: `doc/12_ROADMAP.md`. Current work is slice-by-slice (one commit, `make verify`, push, stop).

| Slice | Content | Status | Commit |
|---|---|---|---|
| A1 | RT-DETR golden test made mAP-sensitive | done | `2dca53b` |
| A2 | Shared TrainEngine | done | `7c4dbef` |
| B | Verification honesty | done | `ef169eb` |
| C | Model zoo + licence card + gate | done | `718761b` |
| D | rank_checkpoints + auto-select + auto-download + failure handling + frozen-backbone guard | done `7cf06d5` |
| E | Token default ON, upload cap, https-only source_uri (+ override merge fix `561eed7`) | done | `b219700` |
| F | Config number types, predictions default, header fixes, device auto + CPU banner | done | Slice F commit (see `git log`) |
