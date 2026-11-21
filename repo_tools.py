"""
repo_tools.py — the core codebase tools.
Plain functions that clone a repo and let us explore it: list files,
read a file, and search the code. The agent will use these (via MCP later).
"""

import os
import re
import subprocess
from pathlib import Path

# Where we clone repos to analyze (gitignored).
REPOS_DIR = Path("repos")

# Directories that are noise for code analysis.
IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__",
               "dist", "build", ".next", ".idea", ".vscode"}
# Extensions we treat as binary / not worth reading as text.
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip",
               ".gz", ".tar", ".exe", ".bin", ".so", ".dll", ".woff",
               ".woff2", ".ttf", ".mp4", ".mp3", ".lock"}


def clone_repo(url: str) -> Path:
    """Shallow-clone a public GitHub repo into repos/<name>. Returns the local path."""
    name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = REPOS_DIR / name
    if dest.exists():
        return dest  # already cloned, reuse it
    REPOS_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True, capture_output=True, text=True,
    )
    return dest


def _safe_path(repo_path: Path, rel_path: str) -> Path:
    """Resolve rel_path inside repo_path, refusing to escape it (blocks ../.. attacks)."""
    repo_path = repo_path.resolve()
    target = (repo_path / rel_path).resolve()
    if repo_path != target and repo_path not in target.parents:
        raise ValueError(f"Path '{rel_path}' escapes the repo root.")
    return target


def list_files(repo_path: Path, subpath: str = "") -> list[str]:
    """List all files under the repo (or a subpath), skipping noise dirs."""
    repo_path = Path(repo_path).resolve()
    base = _safe_path(repo_path, subpath)
    results = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]  # prune noise in place
        for f in files:
            full = Path(root) / f
            results.append(str(full.relative_to(repo_path)))
    return sorted(results)


def read_file(repo_path: Path, file_path: str, max_bytes: int = 50_000) -> str:
    """Read a text file, capped at max_bytes. Handles missing and binary files."""
    repo_path = Path(repo_path)
    target = _safe_path(repo_path, file_path)
    if not target.is_file():
        return f"ERROR: '{file_path}' is not a file or does not exist."
    if target.suffix.lower() in BINARY_EXTS:
        return f"[binary file '{file_path}' — not shown]"
    size = target.stat().st_size
    text = target.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    if size > max_bytes:
        text += f"\n\n[...truncated, file is {size} bytes...]"
    return text


def search_code(repo_path: Path, pattern: str, max_results: int = 50) -> list[str]:
    """Search text files for a regex pattern. Returns 'path:line: text' matches."""
    repo_path = Path(repo_path)
    regex = re.compile(pattern)
    hits = []
    for rel in list_files(repo_path):
        if Path(rel).suffix.lower() in BINARY_EXTS:
            continue
        try:
            lines = (repo_path / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(hits) >= max_results:
                    return hits
    return hits