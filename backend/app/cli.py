"""``trainhub`` command line [doc 14 §5].

Ships inside the image so an operator can prefetch, import weights on an
air-gapped server, or rescan the local store without touching the API:

    docker exec trainhub trainhub checkpoints import ./rtdetr-l.pt --catalog-id rtdetr-l@1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.checkpoints.catalog import CatalogError, ensure_catalog, load_catalog
from app.checkpoints.resolver import CheckpointError, CheckpointIndex, CheckpointResolver
from app.core.config import Settings


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_home(args: argparse.Namespace) -> Path:
    return Path(args.home) if args.home else Settings().home


def _prefetch(args: argparse.Namespace) -> int:
    resolver = CheckpointResolver(_resolve_home(args))
    try:
        resolved = resolver.prefetch(args.catalog_id)
    except CheckpointError as exc:
        _print({"error": {"code": exc.code, "message": exc.message, "details": exc.details}})
        return 1
    _print(
        {
            "catalog_id": resolved.catalog_id,
            "path": str(resolved.path),
            "sha256": resolved.sha256,
            "size_bytes": resolved.size_bytes,
            "source": resolved.source,
        }
    )
    return 0


def _import(args: argparse.Namespace) -> int:
    resolver = CheckpointResolver(_resolve_home(args))
    try:
        resolved = resolver.import_file(args.catalog_id, Path(args.file))
    except CheckpointError as exc:
        _print({"error": {"code": exc.code, "message": exc.message, "details": exc.details}})
        return 1
    _print(
        {
            "catalog_id": resolved.catalog_id,
            "path": str(resolved.path),
            "sha256": resolved.sha256,
            "size_bytes": resolved.size_bytes,
        }
    )
    return 0


def _list(args: argparse.Namespace) -> int:
    home = _resolve_home(args)
    try:
        ensure_catalog(home)
        entries = load_catalog(home)
    except CatalogError as exc:
        _print({"error": {"code": "CATALOG_INVALID", "message": str(exc)}})
        return 1
    index = CheckpointIndex(home).read()
    _print(
        [
            {
                "catalog_id": entry.catalog_id,
                "model_family": entry.model_family,
                "verified_sha256": entry.verified,
                "local": (index.get(entry.catalog_id) or {}).get("status") == "ready",
            }
            for entry in entries
        ]
    )
    return 0


def _rescan(args: argparse.Namespace) -> int:
    """Rebuild index.json from the content-addressed store [doc 14 §5]."""
    home = _resolve_home(args)
    store = home / "checkpoints" / "store"
    index = CheckpointIndex(home)
    entries = {entry.catalog_id: entry for entry in load_catalog(home)}
    found = 0
    for entry in entries.values():
        if not entry.verified:
            continue
        digest = str(entry.sha256)
        candidate = store / digest[:2] / digest / entry.resolved_filename
        if candidate.is_file():
            index.put(
                entry.catalog_id,
                {
                    "path": str(candidate),
                    "sha256": entry.sha256,
                    "size": candidate.stat().st_size,
                    "status": "ready",
                    "source": "rescan",
                    "last_used_at": None,
                },
            )
            found += 1
    _print({"rescanned": True, "restored": found})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainhub", description="TrainHub operator CLI")
    parser.add_argument("--home", help="workspace root (default: TRAINHUB_HOME)")
    checkpoints = parser.add_subparsers(dest="command", required=True).add_parser(
        "checkpoints", help="catalog, prefetch, import, rescan"
    )
    nested = checkpoints.add_subparsers(dest="subcommand", required=True)

    prefetch = nested.add_parser("prefetch", help="download a catalog entry now")
    prefetch.add_argument("catalog_id")
    prefetch.set_defaults(handler=_prefetch)

    importer = nested.add_parser("import", help="import an already-downloaded file")
    importer.add_argument("file")
    importer.add_argument("--catalog-id", required=True, dest="catalog_id")
    importer.set_defaults(handler=_import)

    listing = nested.add_parser("list", help="show catalog entries and local status")
    listing.set_defaults(handler=_list)

    rescan = nested.add_parser("rescan", help="rebuild index.json from the store")
    rescan.set_defaults(handler=_rescan)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
