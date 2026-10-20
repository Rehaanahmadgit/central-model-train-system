# 02 — Goals, Non-Goals and Success Metrics

## 1. Goals
1. **Eliminate wasted training:** catch data problems before training starts.
2. **Automate the starting point:** choose and cache the right pretrained checkpoint automatically.
3. **Full visibility:** real-time, audit-grade view of every training run.
4. **Reproducibility:** any run can be explained and recreated.
5. **Production quality:** reliable, secure, observable — with **no database and no stored user/client information** (privacy by design).
6. **Extensible:** adding a new model family = writing one plugin.

## 2. Non-goals (v1)
- Not an annotation tool.
- Not an inference/serving platform.
- Not a full AutoML that beats expert tuning in every case.
- Not supporting every framework from day one.
- No user accounts, SSO, billing, or any database (files only).

## 3. Success metrics
| Metric | Target |
|---|---|
| Time from upload to training start (MVP dataset ~5k images) | < 5 min |
| % of bad datasets caught in verification (seeded test set) | > 95% |
| Checkpoint re-download rate on repeated runs | 0% |
| Dashboard metric latency (emit → visible) | < 1 s p95 |
| Platform crash due to a single failed training job | 0 |
| Runs with complete technical record (`run.json` + `events.jsonl`) | 100% |
| Personal data found in workspace/logs (email/IP/user-agent grep) | 0 |
| Run state recovered correctly after `kill -9` of API or trainer | 100% |
| New engineer first successful run | < 10 min |
| Accuracy gain vs training from scratch using auto-selected pretrained weights | measurable improvement on benchmark datasets |

## 4. North-star
"Number of successful, fully reproducible training runs per engineer per week."

## 5. Definition of done (any feature)
- Code + tests + docs updated.
- Works through `docker compose up`.
- Errors are handled with clear messages.
- Audit log entries exist for state changes.
- No secrets in code or logs.
