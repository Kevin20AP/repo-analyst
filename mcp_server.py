"""
mcp_server.py — a custom MCP server exposing codebase tools.
Any MCP client (our agent, Claude Desktop, etc.) connects to this server
and calls these tools to explore a repository.
The target repo is set via the REPO_PATH environment variable.
"""

import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

import repo_tools

mcp = FastMCP("repo-analyst-tools")


def _repo() -> Path:
    """Return the repo this server is pointed at (from REPO_PATH)."""
    path = os.environ.get("REPO_PATH")
    if not path:
        raise RuntimeError("REPO_PATH env var is not set.")
    return Path(path)


@mcp.tool()
def list_files(subpath: str = "") -> list[str]:
    """List all files in the repository, optionally under a subpath. Call this first to see the structure."""
    return repo_tools.list_files(_repo(), subpath)


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read the full contents of a single file in the repository, given its path like 'src/api.py'."""
    return repo_tools.read_file(_repo(), file_path)


@mcp.tool()
def search_code(pattern: str) -> list[str]:
    """Search the repository for a regex pattern. Returns 'path:line: text' matches. Use to find where something is defined or used."""
    return repo_tools.search_code(_repo(), pattern)


if __name__ == "__main__":
    mcp.run()
