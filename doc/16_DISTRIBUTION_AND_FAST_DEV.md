# 16 — Distribution to other people + Fast development loop

Priority: this doc is an ADDENDUM to docs 00 and 14. It does not remove any rule there (no database, no user data, checkpoints never baked into the image).

## 1. The goal in one paragraph
The author builds TrainHub on a laptop (CPU only). When it works, ANY person (a teammate, another company, a friend) who owns a strong GPU server must be able to start it with a Docker image and train their own models, without cloning the repo, without installing Python/Node, and without asking the author for help. Each person runs THEIR OWN instance on THEIR OWN server with THEIR OWN data volume. There is no central server, no accounts, no shared database, and the author never sees anyone's data.

## 2. Distribution requirements (DIST-xx)
- DIST-01 Two public/private images: `<registry>/trainhub:<ver>-cpu` and `:<ver>-gpu`, plus `:latest-cpu` / `:latest-gpu`. Registry name is `TODO(user)`.
- DIST-02 Start with ONE command (GPU server, Docker + NVIDIA Container Toolkit installed):
  `docker run -d --name trainhub --gpus all -p 8080:8080 -v trainhub_data:/data <registry>/trainhub:<ver>-gpu`
  and the same without `--gpus all` for the cpu image. VERIFY flags against current Docker docs before publishing.
- DIST-03 Also ship `docker-compose.yml` + `.env.example` (copy-paste ready) in a `deploy/` folder and in the README.
- DIST-04 First start must work with zero configuration: sensible defaults, `TRAINHUB_HOME=/data`, device `auto`, port 8080. All options are env vars documented in one table in the README.
- DIST-05 `docker exec trainhub trainhub doctor` must tell a stranger in plain words what is wrong (no GPU driver, no toolkit, disk full, no internet for checkpoints) and how to fix it.
- DIST-06 README section "Run on your server in 3 steps" (install prerequisites → run command → open browser) with an offline/air-gapped note (`trainhub checkpoints import`, doc 14 §5).
- DIST-07 Security for strangers: default to requiring `TRAINHUB_ACCESS_TOKEN` when the app is bound to a non-localhost address inside Docker; if it is not set, print a loud startup warning and show a banner in the UI. Token identifies nobody (doc 10 §2). `ASSUMPTION` until the author approves.
- DIST-08 No telemetry, no phone-home, no update checks. The only outbound traffic is checkpoint downloads from catalog URLs.
- DIST-09 Image is small and reproducible: pinned deps, multi-stage, non-root, healthcheck, labels with version + git commit, SBOM/licence list in the repo.
- DIST-10 Upgrade path documented: `docker pull` + recreate container; volume keeps datasets, checkpoints, runs; schema migration on startup (doc 04).
- DIST-11 A "sample dataset" button and a tiny model preset so a new person can see the full flow working in minutes before using real data.
- DIST-12 Version compatibility table in README: image tag ↔ CUDA version ↔ minimum NVIDIA driver (VERIFY against official docs).

## 3. Fast development loop (rules for the builder)
Problem to avoid: slow turns with dozens of shell commands.
- FAST-01 Exactly ONE verification entrypoint: `make verify` (or `./scripts/verify.sh`) = lint + type-check + fast tests. Run THAT, not each tool separately. Show its output.
- FAST-02 Test tiers: `make test-fast` (< 60 s, default, no network, no GPU, tiny fixtures), `make test-full` (slow/integration, run only at the end of a milestone), `pytest -m gpu` (server only).
- FAST-03 Per slice: at most 1 verify run + 1 targeted test run of new tests, unless something fails. Do not re-run the whole suite repeatedly.
- FAST-04 Docker is built/tested only at milestones (end of slice 8 = first image, and at release), not every slice. Mark `NOT RUN` instead of building images mid-slice.
- FAST-05 Prefer fewer, larger, complete files per turn over many tiny edits. No exploratory commands unless needed to avoid a wrong guess.
- FAST-06 Keep turn replies short: do not repeat the Spec Index or the docs; reference Req IDs only.

## 4. Build-order milestone addition (append to doc 12/13 build order)
- Milestone D1 (after the core flow works on CPU): build the `-cpu` image, run the one-line command, run `trainhub doctor`, finish the sample dataset flow inside the container.
- Milestone D2 (release): `-gpu` image, README "3 steps", compose + `.env.example`, version/driver table, publish to registry, then ONE `pytest -m gpu` smoke test on a real GPU server by the author.
