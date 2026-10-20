# Docker for beginners — running TrainHub on your server

This guide assumes you have used `git` before but are new to Docker. Every
command is copy-paste ready, with the expected output and the two or three
things that usually go wrong.

## What Docker does here (30 seconds)

A **Docker image** is a frozen snapshot of TrainHub: the Python app, the web UI,
and all their dependencies. A **container** is a running copy of that image.
A **volume** is a folder that lives on your server, outside the container, so
your datasets and trained models survive when the container is removed.

TrainHub keeps every byte of your data in one volume mounted at `/data`.
The image itself never contains your data.

## Step 0 — install Docker

```bash
docker --version
```

Expected: `Docker version 27.x.y` (or newer).

Common errors + fixes:

- `command not found: docker` → install Docker Engine + the Compose plugin:
  <https://docs.docker.com/engine/install/>. Follow the "server" instructions.
- `docker: permission denied` → add yourself to the docker group once, then
  log out and back in: `sudo usermod -aG docker $USER`.

**Only if you want GPU training**, also install the NVIDIA Container Toolkit and
a current NVIDIA driver. Check with:

```bash
nvidia-smi
```

If that prints a driver version and a GPU name, you are good. If it says
`command not found`, install the driver first.

## Step 1 — pull (or build) the image

Pull the prebuilt CPU image:

```bash
docker pull trainhub:0.1.0-cpu
```

Expected: a progress bar ending in `Status: Downloaded newer image for trainhub:0.1.0-cpu`.

> **NOTE:** replace `trainhub:0.1.0-cpu` with the real registry path your team
> gives you, e.g. `ghcr.io/yourorg/trainhub:0.1.0-cpu`.

If you are building from this repo instead:

```bash
docker build -f deploy/Dockerfile --target cpu -t trainhub:0.1.0-cpu .
```

## Step 2 — run it

```bash
docker run -d --name trainhub -p 8080:8080 -v trainhub_data:/data trainhub:0.1.0-cpu
```

What each flag does:

- `-d` — run in the background.
- `--name trainhub` — give the container a fixed name.
- `-p 8080:8080` — publish the app on your server's port 8080.
- `-v trainhub_data:/data` — create (once) and attach a **named volume** so data persists.
- `trainhub:0.1.0-cpu` — the image.

On a GPU server, add `--gpus all` and use the `-gpu` tag:

```bash
docker run -d --name trainhub --gpus all -p 8080:8080 -v trainhub_data:/data trainhub:0.1.0-gpu
```

Expected: `docker run` prints a long container id (e.g. `7f3a…c2d1`) and returns.

Common errors + fixes:

- `port is already allocated` → something else uses 8080. Use another host port:
  `-p 8081:8080`, then open `http://<ip>:8081`.
- `could not select device driver "nvidia"` → the NVIDIA Container Toolkit is
  missing or Docker was restarted without it. Reinstall the toolkit, then
  `sudo systemctl restart docker`.
- Container starts then exits → look at the logs: `docker logs trainhub`.

## Step 3 — check it is healthy

```bash
docker exec trainhub trainhub doctor
```

Expected: a plain-English report starting with `TrainHub doctor`. `[OK]` lines
mean fine; `[FAIL]` lines say what to fix. The most common `[FAIL]`:

- `cannot write to /data` with a note about **uid 10001** → you used a *bind
  mount* (`-v /host/dir:/data`) but the host directory is owned by another user.
  Fix: `mkdir -p /host/dir && sudo chown 10001:10001 /host/dir`, **or** switch to
  a named volume (`-v trainhub_data:/data`) which Docker sets up correctly.

Then open `http://<server-ip>:8080`. Click **New training → Try sample dataset**
and you should see a real (tiny) training run complete in under a minute.

## Daily operations

See the logs:

```bash
docker logs -f trainhub
```

Stop / start / remove:

```bash
docker stop trainhub        # stop
docker start trainhub       # start again (data intact)
docker rm trainhub          # removes the CONTAINER (not your data)
```

> **WARNING:** `docker rm trainhub` deletes the container, not the volume — your
> data in `trainhub_data` survives. But `docker volume rm trainhub_data`
> **permanently deletes all datasets, checkpoints and runs**. Never run it
> unless you mean it. Back up the volume before big changes.

Upgrade to a newer version:

```bash
docker pull trainhub:0.2.0-cpu                       # 1. fetch the new image
docker stop trainhub && docker rm trainhub           # 2. remove the old container (data stays)
docker run -d --name trainhub -p 8080:8080 -v trainhub_data:/data trainhub:0.2.0-cpu  # 3. recreate
```

## Air-gapped servers (no internet)

Download checkpoints on a machine with internet, copy the files over, then:

```bash
docker exec trainhub trainhub checkpoints import ./weights.bin --catalog-id rtdetr-l@1.0
```

The file's sha256 is checked against the catalog before it is accepted.

## If you are truly stuck

`docker logs trainhub` is the first place to look. The app also writes its own
log line at startup and keeps a person-free audit trail under `/data/audit/`.
