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
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.END = ''


# Persistence level styling
PERSISTENCE_COLORS = {
    'core': C.RED,
    'stable': C.YELLOW,
    'situational': C.CYAN,
    'ephemeral': C.DIM,
}

PERSISTENCE_SYMBOLS = {
    'core': '●',
    'stable': '◆',
    'situational': '○',
    'ephemeral': '·',
}


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
        # Old format: timestamp, category, data, relevance
        print(f"  {C.DIM}{mem.get('timestamp', '?')}{C.END}")
        print(f"  {C.BOLD}{mem.get('data', '?')}{C.END}")
        print(f"  Category: {mem.get('category', '?')} | Relevance: {mem.get('relevance', '?')}")
        return

    persistence = mem.get('persistence', 'situational')
    color = PERSISTENCE_COLORS.get(persistence, '')
    symbol = PERSISTENCE_SYMBOLS.get(persistence, '?')

    # Header line: symbol, content
    print(f"  {color}{symbol}{C.END} {C.BOLD}{mem.get('content', '?')}{C.END}")

    # Details line
    details = []
    details.append(f"{color}{persistence}{C.END}")
    details.append(mem.get('category', '?'))
    details.append(mem.get('memory_type', 'fact'))

    if mem.get('reinforced', 0) > 0:
        details.append(f"{C.GREEN}+{mem['reinforced']} reinforced{C.END}")

    age = format_age(mem.get('created', ''))
    details.append(f"{C.DIM}{age}{C.END}")

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

    # Group by persistence
    by_persistence = defaultdict(list)
    for mem in memories:
        by_persistence[mem.get('persistence', 'unknown')].append(mem)

    print(f"\n  {C.BOLD}By Persistence:{C.END}")
    for level in ['core', 'stable', 'situational', 'ephemeral']:
        count = len(by_persistence.get(level, []))
        color = PERSISTENCE_COLORS.get(level, '')
        symbol = PERSISTENCE_SYMBOLS.get(level, '?')
        bar = '█' * min(count, 30)
        print(f"    {color}{symbol} {level:12}{C.END} {count:3}  {C.DIM}{bar}{C.END}")

    # Group by category
    by_category = defaultdict(list)
    for mem in memories:
        by_category[mem.get('category', 'unknown')].append(mem)

    print(f"\n  {C.BOLD}By Category:{C.END}")
    for cat, mems in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"    {cat:15} {len(mems):3}")

    # Group by type
    by_type = defaultdict(list)
    for mem in memories:
        by_type[mem.get('memory_type', 'unknown')].append(mem)

    print(f"\n  {C.BOLD}By Type:{C.END}")
    for mtype, mems in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"    {mtype:15} {len(mems):3}")

    # Reinforcement stats
    reinforced = [m for m in memories if m.get('reinforced', 0) > 0]
    if reinforced:
        total_reinforcements = sum(m.get('reinforced', 0) for m in reinforced)
        print(f"\n  {C.BOLD}Reinforcement:{C.END}")
        print(f"    Memories reinforced: {len(reinforced)}")
        print(f"    Total reinforcements: {total_reinforcements}")
        top = sorted(reinforced, key=lambda x: -x.get('reinforced', 0))[:3]
        print(f"    Top reinforced:")
        for m in top:
            print(f"      +{m['reinforced']}: {m.get('content', '?')[:50]}")


def list_memories(store: dict, filters: dict = None):
    """List memories with optional filtering."""
    memories = store.get('memories', [])
    is_old = store.get('format') == 'old'

    if filters:
        if filters.get('category'):
            memories = [m for m in memories if m.get('category') == filters['category']]
        if filters.get('persistence'):
            memories = [m for m in memories if m.get('persistence') == filters['persistence']]
        if filters.get('type'):
            memories = [m for m in memories if m.get('memory_type') == filters['type']]
        if filters.get('search'):
            term = filters['search'].lower()
            memories = [m for m in memories if term in m.get('content', '').lower() or term in m.get('data', '').lower()]

    if not memories:
        print(f"{C.YELLOW}No memories match the filters{C.END}")
        return

    # Sort by persistence priority, then by reinforced, then by date
    if not is_old:
        persistence_order = {'core': 0, 'stable': 1, 'situational': 2, 'ephemeral': 3}
        memories = sorted(memories, key=lambda m: (
            persistence_order.get(m.get('persistence', 'situational'), 9),
            -m.get('reinforced', 0),
            m.get('created', '')
        ), reverse=False)

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
        print(f"    {C.DIM}{mem.get('category', '?')} | {mem.get('persistence', '?')}{C.END}")
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

            pers = input("  Persistence (core/stable/situational/ephemeral): ").strip().lower()
            if pers:
                filters['persistence'] = pers

            mtype = input("  Type (fact/preference/event/goal/relationship): ").strip().lower()
            if mtype:
                filters['type'] = mtype

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
  python readpinf.py -p core            # Filter by persistence
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
    parser.add_argument('-p', '--persistence', help='Filter by persistence level')
    parser.add_argument('-t', '--type', help='Filter by memory type')
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
    if args.persistence:
        filters['persistence'] = args.persistence
    if args.type:
        filters['type'] = args.type
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
