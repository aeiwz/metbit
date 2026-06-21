#!/usr/bin/env python3
"""Generate versioned API documentation snapshots from GitHub releases and Git tags."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DOCS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DOCS_ROOT.parent
OUTPUT_ROOT = DOCS_ROOT / "content" / "generated"
GITHUB_RELEASES_URL = "https://api.github.com/repos/aeiwz/metbit/releases?per_page=100&page={page}"


@dataclass(frozen=True)
class SourceFile:
    path: str
    source: str


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    )


def fetch_releases() -> list[dict[str, Any]]:
    releases: list[dict[str, Any]] = []
    page = 1
    while True:
        request = urllib.request.Request(
            GITHUB_RELEASES_URL.format(page=page),
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "metbit-docs-sync",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.load(response)
        if not batch:
            break
        releases.extend(batch)
        page += 1
    return releases


def load_release_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    releases: list[dict[str, Any]] = []
    for path in paths:
        releases.extend(json.loads(path.read_text(encoding="utf-8")))
    return releases


def source_files(tag: str) -> list[SourceFile]:
    paths = git("ls-tree", "-r", "--name-only", tag, "--", "metbit").splitlines()
    selected = []
    for path in paths:
        parts = Path(path).parts
        if not path.endswith(".py"):
            continue
        if "__pycache__" in parts or "test" in parts or "tests" in parts or "dev" in parts:
            continue
        try:
            source = git("show", f"{tag}:{path}")
        except subprocess.CalledProcessError:
            continue
        selected.append(SourceFile(path=path, source=source))
    return selected


def clean_docstring(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", value.strip())
    return text[:12_000]


def format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        rendered = ast.unparse(node.args)
    except Exception:
        rendered = ", ".join(arg.arg for arg in node.args.args)
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    return f"{prefix}{node.name}({rendered})"


def public_methods(node: ast.ClassDef) -> list[dict[str, str]]:
    methods = []
    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if child.name.startswith("_") and child.name != "__init__":
            continue
        methods.append(
            {
                "name": child.name,
                "signature": format_signature(child),
                "doc": clean_docstring(ast.get_docstring(child)),
            }
        )
    return methods


def category_for(path: str) -> str:
    if "/analysis/" in path or "/models/" in path or Path(path).stem in {
        "metbit",
        "opls",
        "pls",
        "vip",
        "cross_validation",
        "lazy_opls_da",
    }:
        return "Analysis and models"
    if "/nmr/" in path or "/preprocessing/" in path or Path(path).stem in {
        "baseline",
        "calibrate",
        "denoise_spec",
        "nmr_preprocess",
        "peak_processe",
        "pretreatment",
        "scaler",
        "spec_norm",
    }:
        return "NMR and preprocessing"
    if "/apps/" in path or "/viz/" in path or Path(path).stem in {
        "annotate_peak",
        "boxplot",
        "plotting",
        "pca_ellipse",
        "take_intensity",
        "ui_picky_peak",
        "ui_stocsy",
    }:
        return "Visualization and apps"
    if "/stats/" in path or Path(path).stem == "utility":
        return "Statistics and utilities"
    return "Other"


def module_name(path: str) -> str:
    relative = Path(path).with_suffix("")
    if relative.name == "__init__":
        relative = relative.parent
    return ".".join(relative.parts)


def parse_module(file: SourceFile) -> dict[str, Any] | None:
    try:
        tree = ast.parse(file.source)
    except SyntaxError as error:
        return {
            "name": module_name(file.path),
            "path": file.path,
            "category": category_for(file.path),
            "doc": "",
            "classes": [],
            "functions": [],
            "parseError": f"Line {error.lineno}: {error.msg}",
        }

    classes = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            classes.append(
                {
                    "name": node.name,
                    "doc": clean_docstring(ast.get_docstring(node)),
                    "methods": public_methods(node),
                }
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions.append(
                {
                    "name": node.name,
                    "signature": format_signature(node),
                    "doc": clean_docstring(ast.get_docstring(node)),
                }
            )

    if not classes and not functions and not ast.get_docstring(tree):
        return None
    return {
        "name": module_name(file.path),
        "path": file.path,
        "category": category_for(file.path),
        "doc": clean_docstring(ast.get_docstring(tree)),
        "classes": classes,
        "functions": functions,
        "parseError": None,
    }


def root_exports(files: list[SourceFile]) -> list[str]:
    init = next((item for item in files if item.path == "metbit/__init__.py"), None)
    if init is None:
        return []
    try:
        tree = ast.parse(init.source)
    except SyntaxError:
        return []

    exports: set[str] = set()
    explicit_all: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            exports.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Import):
            exports.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                    explicit_all = list(value)
    return sorted(explicit_all if explicit_all is not None else exports)


def normalized_version(tag: str) -> str:
    value = tag.lstrip("vV")
    return value if re.fullmatch(r"\d+(?:\.\d+)+(?:[-+][0-9A-Za-z.-]+)?", value) else tag


def create_snapshot(tag: str) -> dict[str, Any]:
    files = source_files(tag)
    modules = [module for item in files if (module := parse_module(item)) is not None]
    modules.sort(key=lambda item: item["name"].lower())
    return {
        "rootExports": root_exports(files),
        "modules": modules,
    }


def compact_release(release: dict[str, Any], snapshot_id: str) -> dict[str, Any]:
    tag = release["tag_name"]
    return {
        "tag": tag,
        "version": normalized_version(tag),
        "name": release.get("name") or tag,
        "publishedAt": release.get("published_at"),
        "url": release.get("html_url"),
        "body": (release.get("body") or "").strip()[:8_000],
        "snapshot": snapshot_id,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-json",
        type=Path,
        action="append",
        default=[],
        help="Read GitHub release API responses from local JSON files instead of the network.",
    )
    args = parser.parse_args()

    releases = load_release_files(args.release_json) if args.release_json else fetch_releases()
    releases = [
        release
        for release in releases
        if not release.get("draft") and release.get("tag_name")
    ]

    local_tags = set(git("tag").splitlines())
    missing = [release["tag_name"] for release in releases if release["tag_name"] not in local_tags]
    if missing:
        raise SystemExit(f"Missing local Git tags: {', '.join(missing)}")

    snapshots: dict[str, dict[str, Any]] = {}
    manifest = []
    for index, release in enumerate(releases, start=1):
        tag = release["tag_name"]
        snapshot = create_snapshot(tag)
        canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        snapshot_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        snapshots.setdefault(snapshot_id, snapshot)
        manifest.append(compact_release(release, snapshot_id))
        print(f"[{index:03}/{len(releases)}] {tag} -> {snapshot_id}")

    snapshot_root = OUTPUT_ROOT / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for old_file in snapshot_root.glob("*.json"):
        old_file.unlink()
    for snapshot_id, snapshot in snapshots.items():
        write_json(snapshot_root / f"{snapshot_id}.json", snapshot)

    write_json(
        OUTPUT_ROOT / "releases.json",
        {
            "generatedAt": releases[0].get("updated_at") if releases else None,
            "latest": releases[0]["tag_name"] if releases else None,
            "releaseCount": len(manifest),
            "snapshotCount": len(snapshots),
            "releases": manifest,
        },
    )
    print(f"Generated {len(manifest)} releases using {len(snapshots)} unique API snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
