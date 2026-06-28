#!/usr/bin/env python3
"""
Personal Information Memory Viewer/Debugger

A tool to inspect, analyze, and debug the PersonalInfoAgent's memory store.
Supports the new persistence-based memory format.
"""

import json
import sys
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

from cryptography.fernet import InvalidToken

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.utils import derive_encryption_key, decrypt_data
import _config


# ANSI color codes
class C:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        cls.HEADER = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.END = ''


# Retention bucket styling (v3). 'stable' kept as alias to 'core' for any
# pre-migration v2 file the user inspects before running the agent.
RETENTION_COLORS = {
    'core': C.RED,
    'stable': C.RED,
    'situational': C.CYAN,
    'ephemeral': C.DIM,
}

RETENTION_SYMBOLS = {
    'core': '●',
    'stable': '●',
    'situational': '○',
    'ephemeral': '·',
}


def get_retention(mem: dict) -> str:
    """Read the bucket field, tolerating v2's 'persistence' name."""
    return mem.get('retention') or mem.get('persistence', 'situational')


def get_secret_key(username: str) -> bytes:
    """Derive encryption key using ENCKEY env var and username."""
    key = derive_encryption_key(username)
    if not key:
        raise ValueError("Environment variable 'ENCKEY' not set")
    return key


def find_user_files() -> list[tuple[str, Path]]:
    """Find all personal info files in the data directory."""
    data_dir = Path(_config.data_directory)
    if not data_dir.exists():
        return []

    files = []
    for f in data_dir.iterdir():
        if f.name == ".pinf":
            files.append(("default", f))
        elif f.name.startswith(".pinf_"):
            # Can't reverse the hash, but we can show it
            hash_part = f.name[6:]
            files.append((f"user:{hash_part[:8]}...", f))
    return files


def load_memories(user_id: str) -> Optional[dict]:
    """Load and decrypt memory store for a user."""
    if user_id == "default":
        info_file = Path(_config.data_directory) / ".pinf"
    else:
        hashed_user = hashlib.sha1(user_id.encode()).hexdigest()
        info_file = Path(_config.data_directory) / f".pinf_{hashed_user}"

    if not info_file.exists():
        print(f"{C.RED}File not found: {info_file}{C.END}")
        return None

    with open(info_file, "rb") as f:
        encrypted_data = f.read()

    if not encrypted_data.strip():
        print(f"{C.YELLOW}Empty file{C.END}")
        return None

    try:
        secret_key = get_secret_key(user_id)
        store = decrypt_data(encrypted_data, secret_key)

        # Handle old format (list) vs new format (dict)
        if isinstance(store, list):
            print(f"{C.YELLOW}Old format detected (pre-migration){C.END}")
            return {"memories": store, "superseded": [], "format": "old"}
        return {**store, "format": "new"}

    except InvalidToken:
        print(f"{C.RED}Decryption failed: Invalid key{C.END}")
        return None
    except json.JSONDecodeError:
        print(f"{C.RED}Invalid JSON in decrypted data{C.END}")
        return None


def format_age(timestamp_str: str) -> str:
    """Format timestamp as human-readable age."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M")
        delta = datetime.now() - dt
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                return f"{delta.seconds // 60}m ago"
            return f"{hours}h ago"
        elif days < 7:
            return f"{days}d ago"
        elif days < 30:
            return f"{days // 7}w ago"
        else:
            return f"{days // 30}mo ago"
    except ValueError:
        return "?"


def print_memory(mem: dict, show_id: bool = False, old_format: bool = False):
    """Print a single memory entry with formatting."""
    if old_format:
        # v1: timestamp, category, data, relevance
        print(f"  {C.DIM}{mem.get('timestamp', '?')}{C.END}")
        print(f"  {C.BOLD}{mem.get('data', '?')}{C.END}")
        print(f"  Category: {mem.get('category', '?')} | Relevance: {mem.get('relevance', '?')}")
        return

    retention = get_retention(mem)
    color = RETENTION_COLORS.get(retention, '')
    symbol = RETENTION_SYMBOLS.get(retention, '?')

    print(f"  {color}{symbol}{C.END} {C.BOLD}{mem.get('content', '?')}{C.END}")

    details = [
        f"{color}{retention}{C.END}",
        mem.get('category', '?'),
        f"{C.DIM}{format_age(mem.get('created', ''))}{C.END}",
    ]
    if show_id:
        details.append(f"{C.DIM}id:{mem.get('id', '?')[:8]}{C.END}")

    print(f"    {' | '.join(details)}")


def print_stats(store: dict):
    """Print statistics about the memory store."""
    memories = store.get('memories', [])
    superseded = store.get('superseded', [])
    is_old = store.get('format') == 'old'

    print(f"\n{C.HEADER}{'=' * 60}{C.END}")
    print(f"{C.BOLD}MEMORY STORE STATISTICS{C.END}")
    print(f"{C.HEADER}{'=' * 60}{C.END}\n")

    print(f"  Active memories:    {C.GREEN}{len(memories)}{C.END}")
    print(f"  Superseded:         {C.DIM}{len(superseded)}{C.END}")

    if is_old:
        print(f"\n  {C.YELLOW}⚠ Old format - run the agent to migrate{C.END}")
        return

    # Group by retention
    by_retention = defaultdict(list)
    for mem in memories:
        by_retention[get_retention(mem)].append(mem)

    print(f"\n  {C.BOLD}By Retention:{C.END}")
    for level in ['core', 'situational', 'ephemeral']:
        count = len(by_retention.get(level, []))
        color = RETENTION_COLORS.get(level, '')
        symbol = RETENTION_SYMBOLS.get(level, '?')
        bar = '█' * min(count, 30)
        print(f"    {color}{symbol} {level:12}{C.END} {count:3}  {C.DIM}{bar}{C.END}")

    by_category = defaultdict(list)
    for mem in memories:
        by_category[mem.get('category', 'unknown')].append(mem)

    print(f"\n  {C.BOLD}By Category:{C.END}")
    for cat, mems in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:15} {len(mems):3}")


def list_memories(store: dict, filters: dict = None):
    """List memories with optional filtering."""
    memories = store.get('memories', [])
    is_old = store.get('format') == 'old'

    if filters:
        if filters.get('category'):
            memories = [m for m in memories if m.get('category') == filters['category']]
        if filters.get('retention'):
            memories = [m for m in memories if get_retention(m) == filters['retention']]
        if filters.get('search'):
            term = filters['search'].lower()
            memories = [m for m in memories
                        if term in m.get('content', '').lower()
                        or term in m.get('data', '').lower()]

    if not memories:
        print(f"{C.YELLOW}No memories match the filters{C.END}")
        return

    if not is_old:
        retention_order = {'core': 0, 'situational': 1, 'ephemeral': 2}
        memories = sorted(memories, key=lambda m: (
            retention_order.get(get_retention(m), 9),
            m.get('created', '')
        ))

    print(f"\n{C.HEADER}{'=' * 60}{C.END}")
    print(f"{C.BOLD}MEMORIES ({len(memories)}){C.END}")
    print(f"{C.HEADER}{'=' * 60}{C.END}\n")

    for mem in memories:
        print_memory(mem, show_id=True, old_format=is_old)
        print()


def list_superseded(store: dict):
    """List superseded (historical) memories."""
    superseded = store.get('superseded', [])

    if not superseded:
        print(f"{C.YELLOW}No superseded memories{C.END}")
        return

    print(f"\n{C.HEADER}{'=' * 60}{C.END}")
    print(f"{C.BOLD}SUPERSEDED MEMORIES ({len(superseded)}){C.END}")
    print(f"{C.HEADER}{'=' * 60}{C.END}\n")

    for mem in superseded:
        status = mem.get('status', 'superseded')
        print(f"  {C.DIM}[{status}]{C.END} {mem.get('content', mem.get('data', '?'))}")
        print(f"    {C.DIM}{mem.get('category', '?')} | {get_retention(mem)}{C.END}")
        print()


def export_json(store: dict, filepath: str):
    """Export memories to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(store, f, indent=2)
    print(f"{C.GREEN}Exported to {filepath}{C.END}")


def interactive_mode(store: dict):
    """Interactive menu for exploring memories."""
    while True:
        print(f"\n{C.BOLD}Commands:{C.END}")
        print("  [s]tats     - Show statistics")
        print("  [l]ist      - List all memories")
        print("  [f]ilter    - Filter memories")
        print("  [h]istory   - Show superseded memories")
        print("  [e]xport    - Export to JSON")
        print("  [q]uit      - Exit")

        cmd = input(f"\n{C.CYAN}>{C.END} ").strip().lower()

        if cmd in ('s', 'stats'):
            print_stats(store)

        elif cmd in ('l', 'list'):
            list_memories(store)

        elif cmd in ('f', 'filter'):
            print("\nFilter options (press Enter to skip):")
            filters = {}

            cat = input("  Category: ").strip().lower()
            if cat:
                filters['category'] = cat

            ret = input("  Retention (core/situational/ephemeral): ").strip().lower()
            if ret:
                filters['retention'] = ret

            search = input("  Search text: ").strip()
            if search:
                filters['search'] = search

            list_memories(store, filters)

        elif cmd in ('h', 'history'):
            list_superseded(store)

        elif cmd in ('e', 'export'):
            filepath = input("  Export path [memories.json]: ").strip() or "memories.json"
            export_json(store, filepath)

        elif cmd in ('q', 'quit', 'exit'):
            break

        else:
            print(f"{C.YELLOW}Unknown command{C.END}")


def main():
    parser = argparse.ArgumentParser(
        description='Personal Information Memory Viewer/Debugger',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python readpinf.py                    # Interactive mode for default user
  python readpinf.py -u user@email.com  # Specific user
  python readpinf.py --stats            # Show stats only
  python readpinf.py --list             # List all memories
  python readpinf.py -c work            # Filter by category
  python readpinf.py -r core            # Filter by retention
  python readpinf.py --search python    # Search in content
  python readpinf.py --users            # List all user files
  python readpinf.py --no-color         # Disable colors
        '''
    )

    parser.add_argument('-u', '--user', default='default', help='User ID (default: "default")')
    parser.add_argument('--users', action='store_true', help='List all user files')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--list', action='store_true', help='List all memories')
    parser.add_argument('--history', action='store_true', help='Show superseded memories')
    parser.add_argument('-c', '--category', help='Filter by category')
    parser.add_argument('-r', '--retention', help='Filter by retention level (core/situational/ephemeral)')
    parser.add_argument('--search', help='Search in content')
    parser.add_argument('--export', help='Export to JSON file')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        C.disable()

    # List users mode
    if args.users:
        files = find_user_files()
        if not files:
            print("No personal info files found")
            return
        print(f"\n{C.BOLD}User files found:{C.END}")
        for user_label, path in files:
            size = path.stat().st_size
            print(f"  {user_label:30} {size:>8} bytes  {path.name}")
        return

    # Load memories
    store = load_memories(args.user)
    if not store:
        return

    # Build filters
    filters = {}
    if args.category:
        filters['category'] = args.category
    if args.retention:
        filters['retention'] = args.retention
    if args.search:
        filters['search'] = args.search

    # Command modes
    if args.export:
        export_json(store, args.export)
    elif args.stats:
        print_stats(store)
    elif args.history:
        list_superseded(store)
    elif args.list or filters:
        list_memories(store, filters if filters else None)
    else:
        # Interactive mode
        print(f"\n{C.BOLD}Memory Viewer - User: {args.user}{C.END}")
        print_stats(store)
        interactive_mode(store)


if __name__ == "__main__":
    main()
