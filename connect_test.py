import asyncio
import os
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    client = MultiServerMCPClient({
        "repo": {
            "command": sys.executable,        # the venv's python interpreter
            "args": ["mcp_server.py"],         # run your MCP server
            "transport": "stdio",              # talk over standard input/output
            "env": {**os.environ, "REPO_PATH": "repos/rag-chat-docs"},
        }
    })
    tools = await client.get_tools()
    print(f"Connected! Got {len(tools)} tools from the MCP server:\n")
    for t in tools:
        print(f"  - {t.name}: {t.description}")


asyncio.run(main())