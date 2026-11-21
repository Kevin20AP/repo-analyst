from repo_tools import clone_repo, list_files, read_file, search_code

path = clone_repo("https://github.com/Kevin20AP/rag-chat-docs")
print("Cloned to:", path)

files = list_files(path)
print(f"\n{len(files)} files found. First 10:")
for f in files[:10]:
    print(" ", f)

print("\n--- First 300 chars of db.py ---")
print(read_file(path, "db.py")[:300])

print("\n--- Searching for 'def ' ---")
for hit in search_code(path, "def ")[:5]:
    print(" ", hit)