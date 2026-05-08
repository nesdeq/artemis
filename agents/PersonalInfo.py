# PersonalInfo.py
"""
Personal Information Agent - Memory as User Model

Extracts, evolves, and maintains a structured model of user information.
Uses persistence levels and memory types instead of numeric relevance scores.
"""

import json
import os
import uuid
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from collections import defaultdict

from cryptography.fernet import InvalidToken

from .Agent import Agent
from tools.utils import extract_json, derive_encryption_key, encrypt_data, decrypt_data

logger = logging.getLogger(__name__)

import _config

# Constants
ENCRYPTION_ENABLED = True

CATEGORIES = [
    "identity",
    "preferences",
    "interests",
    "relationships",
    "work",
    "health",
    "goals",
    "tasks",
    "milestones",
]

PERSISTENCE_LEVELS = ["core", "stable", "situational", "ephemeral"]
MEMORY_TYPES = ["fact", "preference", "event", "goal", "relationship"]

# Retention periods in days
RETENTION_DAYS = {
    "core": None,        # Forever
    "stable": None,      # Until superseded
    "situational": 30,   # 30 days
    "ephemeral": 2,      # 48 hours
}

MAX_MEMORIES = getattr(_config, 'max_personal_entries', 150)


class PersonalInfoAgent(Agent):
    """Agent that maintains an evolving model of the user."""

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        logger.info(f"Initializing PersonalInfoAgent for user: {user}")

        # Storage
        self.memories: list[dict[str, Any]] = []
        self.superseded: list[dict[str, Any]] = []  # Historical record

        # File setup
        hashed_user = hashlib.sha1(str(self.user).encode()).hexdigest() if user else None
        self.info_file = Path(_config.data_directory) / (f".pinf_{hashed_user}" if user else ".pinf")
        os.makedirs(self.info_file.parent, exist_ok=True)

        # Encryption
        self.secret_key = self._get_secret_key() if ENCRYPTION_ENABLED else None

        # State
        self._pending_save = False
        self.profile_cache = ""

        # Load and initialize
        self._load()
        self._apply_retention_policy()
        self._generate_profile()

    # -------------------------------------------------------------------------
    # Encryption (uses shared utilities)
    # -------------------------------------------------------------------------

    def _get_secret_key(self) -> Optional[bytes]:
        username = str(self.user) if self.user else "default"
        key = derive_encryption_key(username)
        if not key:
            logger.warning("ENCKEY not set, encryption disabled")
        return key

    def _encrypt(self, data: dict) -> bytes:
        if not ENCRYPTION_ENABLED or not self.secret_key:
            return json.dumps(data).encode()
        try:
            return encrypt_data(data, self.secret_key)
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return json.dumps(data).encode()

    def _decrypt(self, data: bytes) -> dict:
        if not ENCRYPTION_ENABLED or not self.secret_key:
            return json.loads(data.decode())
        return decrypt_data(data, self.secret_key)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save(self) -> None:
        try:
            store = {
                "memories": self.memories,
                "superseded": self.superseded[-_config.superseded_history_size:],
            }
            with open(self.info_file, "wb") as f:
                f.write(self._encrypt(store))
            self._pending_save = False
            logger.debug(f"Saved {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Error saving: {e}")

    def _load(self) -> None:
        if not self.info_file.exists():
            logger.info("No existing memory file")
            return
        try:
            with open(self.info_file, "rb") as f:
                data = f.read()
            if not data.strip():
                return
            store = self._decrypt(data)
            # Handle migration from old format
            if isinstance(store, list):
                self.memories = self._migrate_old_format(store)
                self._pending_save = True
            else:
                self.memories = store.get("memories", [])
                self.superseded = store.get("superseded", [])
            logger.info(f"Loaded {len(self.memories)} memories")
        except (InvalidToken, json.JSONDecodeError) as e:
            logger.error(f"Failed to load: {e}")
            self.memories = []
        except Exception as e:
            logger.error(f"Unexpected load error: {e}")
            self.memories = []

    def _migrate_old_format(self, old_entries: list) -> list:
        """Migrate from old 1-7 relevance format to new persistence model."""
        logger.info(f"Migrating {len(old_entries)} entries from old format")
        migrated = []
        for entry in old_entries:
            # Map old relevance to persistence
            relevance = entry.get("relevance", 4)
            if relevance >= 6:
                persistence = "core"
            elif relevance >= 5:
                persistence = "stable"
            elif relevance >= 4:
                persistence = "situational"
            else:
                persistence = "ephemeral"

            migrated.append({
                "id": str(uuid.uuid4()),
                "created": entry.get("timestamp", datetime.now().strftime("%Y-%m-%d-%H-%M")),
                "updated": entry.get("timestamp", datetime.now().strftime("%Y-%m-%d-%H-%M")),
                "category": entry.get("category", "preferences"),
                "content": entry.get("data", ""),
                "memory_type": "fact",
                "persistence": persistence,
                "status": "active",
                "supersedes": None,
                "reinforced": 0,
            })
        return migrated

    # -------------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------------

    def _extract_memories(self, user_input: str, last_response: Optional[str]) -> list[dict]:
        """Extract structured memories from user input."""
        if len(user_input.strip()) < 10:
            return []

        prompt = f"""Extract memorable personal information from the user's message.

For each piece of information, provide:

1. CONTENT: The information in third person ("User...")

2. CATEGORY: {' | '.join(CATEGORIES)}

3. TYPE:
   - fact: Objective information ("User is an engineer")
   - preference: Opinion/taste ("User dislikes meetings")
   - event: Something that happened ("User moved to Berlin")
   - goal: Aspiration ("User wants to run a marathon")
   - relationship: Connection to person/entity ("User's daughter is 5")

4. PERSISTENCE:
   - core: Defining/permanent (profession, nationality, core beliefs)
   - stable: Long-term but changeable (current job, city, main hobby)
   - situational: Current context (ongoing project, upcoming event)
   - ephemeral: Temporary state (today's mood, current activity)

EXAMPLES:
Input: "I'm a pediatric nurse and I've been feeling overwhelmed lately"
Output: [
  {{"content": "User is a pediatric nurse", "category": "work", "type": "fact", "persistence": "core"}},
  {{"content": "User is feeling overwhelmed with work", "category": "work", "type": "fact", "persistence": "situational"}}
]

Input: "My brother's wedding is next month, still need to write my speech"
Output: [
  {{"content": "User's brother is getting married next month", "category": "relationships", "type": "event", "persistence": "situational"}},
  {{"content": "User needs to write a wedding speech", "category": "tasks", "type": "goal", "persistence": "situational"}}
]

Input: "Hey, what's the weather?"
Output: []

RULES:
- Skip greetings, questions, commands with no personal info
- When uncertain on persistence, choose SHORTER duration
- Prefer specific over vague ("software engineer" not "works in tech")
- Maximum 3 entries per message
- Return ONLY valid JSON array, no other text

Context from assistant's last response: "{last_response or 'None'}"

User input: "{user_input}"
"""
        try:
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.memory_extract_max_tokens
            )
            parsed = extract_json(response)

            if not isinstance(parsed, list):
                return []

            now = datetime.now().strftime("%Y-%m-%d-%H-%M")
            valid = []

            for entry in parsed[:3]:  # Enforce max 3
                # Validate and normalize
                content = entry.get("content", "").strip()
                if not content or len(content) < 5:
                    continue

                category = entry.get("category", "preferences")
                if category not in CATEGORIES:
                    category = "preferences"

                mem_type = entry.get("type", "fact")
                if mem_type not in MEMORY_TYPES:
                    mem_type = "fact"

                persistence = entry.get("persistence", "situational")
                if persistence not in PERSISTENCE_LEVELS:
                    persistence = "situational"

                valid.append({
                    "id": str(uuid.uuid4()),
                    "created": now,
                    "updated": now,
                    "category": category,
                    "content": content,
                    "memory_type": mem_type,
                    "persistence": persistence,
                    "status": "active",
                    "supersedes": None,
                    "reinforced": 0,
                })

            return valid

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return []

    # -------------------------------------------------------------------------
    # Memory Evolution
    # -------------------------------------------------------------------------

    def _find_related(self, new_memory: dict) -> list[dict]:
        """Find potentially related active memories."""
        related = []
        new_content = new_memory["content"].lower()
        new_category = new_memory["category"]

        # Extract key terms (simple approach - no embeddings)
        new_words = set(new_content.split())

        for mem in self.memories:
            if mem["status"] != "active":
                continue
            if mem["category"] != new_category:
                continue

            mem_words = set(mem["content"].lower().split())
            overlap = len(new_words & mem_words)

            # Require at least 2 word overlap or substring match
            if overlap >= 2 or new_content in mem["content"].lower() or mem["content"].lower() in new_content:
                related.append(mem)

        return related

    def _classify_relationship(self, new_memory: dict, existing: dict) -> str:
        """Use LLM to classify relationship between memories."""
        prompt = f"""Compare these two pieces of information about a user:

EXISTING: "{existing['content']}"
NEW: "{new_memory['content']}"

What is the relationship?
- REINFORCEMENT: Same information repeated (e.g., "User likes coffee" and "User enjoys coffee")
- UPDATE: New info replaces old (e.g., "User works at Google" replaces "User works at Microsoft")
- ELABORATION: New info adds detail (e.g., "User is a senior engineer" elaborates "User is an engineer")
- NEW: Different information, both should be kept

Return exactly one word: REINFORCEMENT, UPDATE, ELABORATION, or NEW"""

        try:
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.memory_classify_max_tokens
            )
            result = response.strip().upper()
            if result in ["REINFORCEMENT", "UPDATE", "ELABORATION", "NEW"]:
                return result
            return "NEW"
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "NEW"

    def _integrate_memory(self, new_memory: dict) -> bool:
        """Integrate a new memory, handling evolution. Returns True if memory was added/updated."""
        related = self._find_related(new_memory)

        if not related:
            self.memories.append(new_memory)
            return True

        # Check against most similar (first related)
        existing = related[0]
        relationship = self._classify_relationship(new_memory, existing)

        if relationship == "REINFORCEMENT":
            existing["reinforced"] += 1
            existing["updated"] = new_memory["updated"]
            # Promote ephemeral to situational if reinforced
            if existing["persistence"] == "ephemeral" and existing["reinforced"] >= 2:
                existing["persistence"] = "situational"
            return True

        elif relationship == "UPDATE":
            # Supersede old with new - remove first to ensure atomic operation
            # (if remove fails, we haven't modified anything)
            self.memories.remove(existing)
            existing["status"] = "superseded"
            self.superseded.append(existing)
            new_memory["supersedes"] = existing["id"]
            self.memories.append(new_memory)
            return True

        elif relationship == "ELABORATION":
            # Merge: keep more detailed version
            if len(new_memory["content"]) > len(existing["content"]):
                existing["content"] = new_memory["content"]
            existing["updated"] = new_memory["updated"]
            existing["reinforced"] += 1
            # Potentially upgrade persistence
            persistence_rank = PERSISTENCE_LEVELS.index
            if persistence_rank(new_memory["persistence"]) < persistence_rank(existing["persistence"]):
                existing["persistence"] = new_memory["persistence"]
            return True

        else:  # NEW
            self.memories.append(new_memory)
            return True

    # -------------------------------------------------------------------------
    # Retention & Cleanup
    # -------------------------------------------------------------------------

    def _apply_retention_policy(self) -> None:
        """Remove expired memories based on persistence level."""
        now = datetime.now()
        remove_ids = set()

        for mem in self.memories:
            if mem["status"] != "active":
                remove_ids.add(mem["id"])
                continue

            retention = RETENTION_DAYS.get(mem["persistence"])
            if retention is None:
                continue  # Keep forever

            try:
                created = datetime.strptime(mem["created"], "%Y-%m-%d-%H-%M")
                age_days = (now - created).days

                # Reinforced memories get extended retention
                effective_retention = retention + (
                    mem.get("reinforced", 0) * _config.reinforcement_retention_bonus_days
                )

                if age_days > effective_retention:
                    remove_ids.add(mem["id"])
            except ValueError:
                continue

        if remove_ids:
            expired = [m for m in self.memories if m["id"] in remove_ids]
            self.memories = [m for m in self.memories if m["id"] not in remove_ids]
            self.superseded.extend(expired)
            logger.info(f"Retention policy removed {len(remove_ids)} memories")
            self._pending_save = True

    def _enforce_limits(self) -> None:
        """Ensure we don't exceed MAX_MEMORIES."""
        if len(self.memories) <= MAX_MEMORIES:
            return

        # Sort by priority: persistence level, then reinforced count, then recency
        # We want to KEEP: core (0), high reinforced, RECENT
        # We want to REMOVE: ephemeral (3), low reinforced, OLD
        # Sort so best items are at beginning, pop() removes from end
        def priority(mem):
            pers_rank = PERSISTENCE_LEVELS.index(mem.get("persistence", "situational"))
            reinforced = mem.get("reinforced", 0)
            try:
                updated = datetime.strptime(mem["updated"], "%Y-%m-%d-%H-%M")
                # Negate timestamp so newer dates sort earlier (lower value = keep)
                updated_score = -updated.timestamp()
            except ValueError:
                updated_score = 0  # Treat invalid as neutral
            return (pers_rank, -reinforced, updated_score)

        self.memories.sort(key=priority)

        # Remove lowest priority until under limit
        while len(self.memories) > MAX_MEMORIES:
            removed = self.memories.pop()
            self.superseded.append(removed)

    # -------------------------------------------------------------------------
    # Profile Generation
    # -------------------------------------------------------------------------

    def _generate_profile(self) -> None:
        """Generate structured user profile from memories."""
        if not self.memories:
            self.profile_cache = "No personal information available yet."
            return

        # Group active memories by category
        by_category: dict[str, list] = defaultdict(list)
        for mem in self.memories:
            if mem["status"] == "active":
                by_category[mem["category"]].append(mem)

        # Sort within categories: core first, then stable, then by reinforced
        def mem_priority(m):
            pers_rank = PERSISTENCE_LEVELS.index(m.get("persistence", "situational"))
            return (pers_rank, -m.get("reinforced", 0))

        sections = []

        # Identity & Core Info (core/stable from identity, work, relationships, milestones)
        identity_categories = ["identity", "work", "relationships", "milestones"]
        core_items = []
        for cat in identity_categories:
            for mem in sorted(by_category.get(cat, []), key=mem_priority):
                if mem["persistence"] in ["core", "stable"]:
                    marker = f"({mem['persistence']})"
                    core_items.append(f"- {mem['content']} {marker}")
                    if len(core_items) >= 10:
                        break
            if len(core_items) >= 10:
                break
        if core_items:
            sections.append("IDENTITY & BACKGROUND\n" + "\n".join(core_items))

        # Current Situation (situational from identity/work/relationships/milestones only)
        # Other categories show situational in their own sections to avoid duplication
        situational = []
        for cat in identity_categories:
            for mem in sorted(by_category.get(cat, []), key=mem_priority):
                if mem["persistence"] == "situational":
                    situational.append(f"- {mem['content']}")
        if situational:
            sections.append("CURRENT SITUATION\n" + "\n".join(situational[:10]))

        # Preferences & Interests
        pref_items = []
        for cat in ["preferences", "interests"]:
            for mem in sorted(by_category.get(cat, []), key=mem_priority)[:5]:
                pref_items.append(f"- {mem['content']}")
        if pref_items:
            sections.append("PREFERENCES & INTERESTS\n" + "\n".join(pref_items))

        # Goals & Tasks
        goal_items = []
        for cat in ["goals", "tasks"]:
            for mem in sorted(by_category.get(cat, []), key=mem_priority)[:5]:
                goal_items.append(f"- {mem['content']}")
        if goal_items:
            sections.append("GOALS & TASKS\n" + "\n".join(goal_items))

        # Health (if any) - sorted by priority
        health_mems = sorted(by_category.get("health", []), key=mem_priority)[:3]
        health_items = [f"- {m['content']}" for m in health_mems]
        if health_items:
            sections.append("HEALTH\n" + "\n".join(health_items))

        self.profile_cache = "\n\n".join(sections) if sections else "No notable personal information."

    # -------------------------------------------------------------------------
    # Agent Interface
    # -------------------------------------------------------------------------

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        return True  # Always runs as background agent

    def process(self, user_input: str, last_response: Optional[str] = None) -> str:
        """Extract memories, integrate them, and return user profile."""
        new_memories = self._extract_memories(user_input, last_response)

        added = 0
        for mem in new_memories:
            if self._integrate_memory(mem):
                added += 1
                self._pending_save = True

        if added > 0:
            logger.info(f"Integrated {added} memories")
            self._enforce_limits()
            self._generate_profile()

        # Batch save
        if self._pending_save:
            self._save()

        return self.profile_cache
