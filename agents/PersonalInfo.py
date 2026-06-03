"""
Personal Information Agent.

Always-on enricher. Every turn injects the user profile as background context
for the main LLM, and (when warranted) updates that profile.

Architecture (see agents/Agent.py and agents/_Agents.md for the LAWS):
- should_process: returns True. Always-on agent — decision is degenerate.
- process: contains separate LLM calls.
    1. DECISION call:   "is there memorable info in this turn?" → yes/no
    2. EXECUTION call:  "extract entries" → list of memory candidates
    3. SUB-TASK calls:  per (new, related-existing) pair, "REINFORCEMENT /
                        UPDATE / ELABORATION / NEW?" — drives integration.
  Decision and execution are NEVER merged.

Encrypted at rest when ENCKEY is set (PBKDF2 → Fernet); if no key is configured
the store falls back to plaintext JSON. Three retention buckets: core /
situational / ephemeral.
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

import _config
from .Agent import Agent
from tools.utils import extract_json, derive_encryption_key, encrypt_data, decrypt_data

logger = logging.getLogger(__name__)


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

# Retention buckets, in order of decreasing permanence.
RETENTION_LEVELS = ["core", "situational", "ephemeral"]
RETENTION_DAYS = {
    "core": None,        # forever
    "situational": 30,
    "ephemeral": 2,
}

RELATIONSHIPS = ["REINFORCEMENT", "UPDATE", "ELABORATION", "NEW"]

MAX_MEMORIES = _config.max_personal_entries

# Validation/integration tuning (centralised in _config.py).
MIN_CONTENT_CHARS = _config.memory_min_content_chars
MAX_EXTRACTIONS_PER_TURN = _config.memory_max_extractions_per_turn
RELATEDNESS_OVERLAP = _config.memory_relatedness_overlap
PROMOTION_REINFORCEMENT_THRESHOLD = _config.memory_promotion_reinforcement_threshold

# Profile display caps (total items per section)
PROFILE_CAPS = {
    "identity_background": 10,    # core memories from identity/work/relationships/milestones
    "current_situation": 10,      # situational memories from same 4 categories
    "preferences_interests": 10,  # any retention from preferences/interests
    "goals_tasks": 10,            # any retention from goals/tasks
    "health": 3,                  # any retention from health
}

IDENTITY_CATEGORIES = ["identity", "work", "relationships", "milestones"]


class PersonalInfoAgent(Agent):
    """Always-on. Injects user profile; updates it via decision+execution LLM calls."""

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        logger.info(f"Initializing PersonalInfoAgent for user: {user}")

        self.memories: list[dict[str, Any]] = []
        self.superseded: list[dict[str, Any]] = []

        hashed_user = hashlib.sha1(str(self.user).encode()).hexdigest() if user else None
        self.info_file = Path(_config.data_directory) / (f".pinf_{hashed_user}" if user else ".pinf")
        os.makedirs(self.info_file.parent, exist_ok=True)

        self.secret_key = self._get_secret_key() if ENCRYPTION_ENABLED else None

        self._pending_save = False
        self.profile_cache = ""

        self._load()
        self._apply_retention_policy()
        self._generate_profile()

    # ------------------------------------------------------------------ Encryption
    def _get_secret_key(self) -> Optional[bytes]:
        username = str(self.user) if self.user else "default"
        key = derive_encryption_key(username)
        if not key:
            logger.warning("ENCKEY not set, encryption disabled")
        return key

    def _encrypt(self, data: dict) -> bytes:
        # No key → encryption is opt-out; store plaintext JSON.
        if not ENCRYPTION_ENABLED or not self.secret_key:
            return json.dumps(data).encode()
        # Key present → NEVER silently downgrade to plaintext on error. Let the
        # exception propagate so _save() aborts and leaves the prior (encrypted)
        # file intact rather than overwriting it with unencrypted data.
        return encrypt_data(data, self.secret_key)

    def _decrypt(self, data: bytes) -> dict:
        if not ENCRYPTION_ENABLED or not self.secret_key:
            return json.loads(data.decode())
        return decrypt_data(data, self.secret_key)

    # ------------------------------------------------------------------ Persistence
    def _save(self) -> None:
        try:
            store = {
                "memories": self.memories,
                "superseded": self.superseded[-_config.superseded_history_size:],
            }
            payload = self._encrypt(store)
            # Atomic write: serialise to a sibling temp file, fsync, then
            # os.replace() (atomic on POSIX, same filesystem). A crash mid-write
            # can no longer truncate the live store and wipe every memory on the
            # next load. If _encrypt raised, we never reach here and the existing
            # file is left untouched.
            tmp = self.info_file.parent / (self.info_file.name + ".tmp")
            with open(tmp, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.info_file)
            self._pending_save = False
            logger.debug(f"Saved {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Error saving: {e}")

    def _load(self) -> None:
        if not self.info_file.exists():
            return
        try:
            with open(self.info_file, "rb") as f:
                data = f.read()
            if not data.strip():
                return
            store = self._decrypt(data)

            if isinstance(store, list):
                self.memories = self._migrate_v1(store)
                self._pending_save = True
                logger.info(f"Loaded {len(self.memories)} memories (migrated from v1)")
                return

            raw = store.get("memories", [])
            self.superseded = store.get("superseded", [])
            if raw and ("persistence" in raw[0] or "memory_type" in raw[0]):
                self.memories = self._migrate_v2(raw)
                self._pending_save = True
                logger.info(f"Loaded {len(self.memories)} memories (migrated from v2)")
            else:
                self.memories = raw
                # Backfill reinforced if loading earlier v3 files that dropped it
                for m in self.memories:
                    m.setdefault("reinforced", 0)
                logger.info(f"Loaded {len(self.memories)} memories")
        except (InvalidToken, json.JSONDecodeError) as e:
            logger.error(f"Failed to load: {e}")
            self.memories = []
        except Exception as e:
            logger.error(f"Unexpected load error: {e}")
            self.memories = []

    def _migrate_v1(self, entries: list) -> list:
        """v1 (1-7 relevance) → v3 (3-bucket retention + reinforced counter)."""
        now = datetime.now().strftime("%Y-%m-%d-%H-%M")
        out = []
        for e in entries:
            rel = e.get("relevance", 4)
            # 1-7 → 3 buckets
            retention = "core" if rel >= 5 else ("situational" if rel >= 3 else "ephemeral")
            ts = e.get("timestamp", now)
            out.append({
                "id": str(uuid.uuid4()),
                "created": ts,
                "updated": ts,
                "category": e.get("category", "preferences"),
                "content": e.get("data", ""),
                "retention": retention,
                "status": "active",
                "supersedes": None,
                "reinforced": 0,
            })
        return out

    def _migrate_v2(self, entries: list) -> list:
        """v2 (4-bucket persistence + memory_type) → v3. Preserves reinforced counter."""
        out = []
        for e in entries:
            persistence = e.get("persistence", "situational")
            # core+stable merge into core (the v2 distinction was never load-bearing)
            retention = "core" if persistence in ("core", "stable") else persistence
            if retention not in RETENTION_LEVELS:
                retention = "situational"
            out.append({
                "id": e.get("id", str(uuid.uuid4())),
                "created": e.get("created", ""),
                "updated": e.get("updated", e.get("created", "")),
                "category": e.get("category", "preferences"),
                "content": e.get("content", ""),
                "retention": retention,
                "status": e.get("status", "active"),
                "supersedes": e.get("supersedes"),
                "reinforced": e.get("reinforced", 0),
            })
        return out

    # ------------------------------------------------------------------ Decision (LLM)
    def _has_memorable_info(self, user_input: str) -> bool:
        """DECISION LLM call: does this turn carry memorable personal info?"""
        prompt = f"""Does this message contain memorable personal information about the user — anything across identity, work, preferences, interests, relationships, health, goals, tasks, or milestones?

Answer with exactly one word: YES or NO.

Message: "{user_input}"

Answer:"""
        try:
            resp = self.llm.generate_single_response(
                prompt, max_tokens=_config.memory_classify_max_tokens
            ).strip().upper()
            return resp.startswith("YES")
        except Exception as e:
            logger.error(f"Memorable-info decision error: {e}")
            return False

    # ------------------------------------------------------------------ Execution (LLM)
    def _extract_memories(self, user_input: str, last_response: Optional[str]) -> list[dict]:
        """EXECUTION LLM call: extract structured memory entries."""
        prompt = f"""Extract memorable personal information from the user's message.

For each piece of information, provide:

1. CONTENT: in third person ("User...")
2. CATEGORY: {' | '.join(CATEGORIES)}
3. RETENTION:
   - core: lasting facts (profession, nationality, family, core beliefs)
   - situational: current context (active project, current city, recent event)
   - ephemeral: passing state (today's mood, current activity)

EXAMPLES

Input: "I'm a pediatric nurse and I've been feeling overwhelmed lately"
→ [
  {{"content": "User is a pediatric nurse", "category": "work", "retention": "core"}},
  {{"content": "User is feeling overwhelmed at work", "category": "work", "retention": "situational"}}
]

Input: "My brother's wedding is next month, still need to write my speech"
→ [
  {{"content": "User's brother is getting married next month", "category": "relationships", "retention": "situational"}},
  {{"content": "User needs to write a wedding speech", "category": "tasks", "retention": "situational"}}
]

Input: "Hey what's the weather?"
→ []

RULES
- Skip greetings, questions, commands with no personal info
- If unsure on retention, choose shorter duration
- Prefer specific over vague ("software engineer" not "works in tech")
- Maximum {MAX_EXTRACTIONS_PER_TURN} entries
- Return ONLY a valid JSON array; no other text

PREVIOUS ASSISTANT: "{last_response or 'None'}"
USER: "{user_input}"
"""
        try:
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.memory_extract_max_tokens
            )
            parsed = extract_json(response)
            if not isinstance(parsed, list):
                return []

            now = datetime.now().strftime("%Y-%m-%d-%H-%M")
            out = []
            for entry in parsed[:MAX_EXTRACTIONS_PER_TURN]:
                if not isinstance(entry, dict):
                    continue
                content = (entry.get("content") or "").strip()
                if len(content) < MIN_CONTENT_CHARS:
                    continue
                category = entry.get("category", "preferences")
                if category not in CATEGORIES:
                    category = "preferences"
                retention = entry.get("retention", "situational")
                if retention not in RETENTION_LEVELS:
                    retention = "situational"
                out.append({
                    "id": str(uuid.uuid4()),
                    "created": now,
                    "updated": now,
                    "category": category,
                    "content": content,
                    "retention": retention,
                    "status": "active",
                    "supersedes": None,
                    "reinforced": 0,
                })
            return out
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return []

    # ------------------------------------------------------------------ Integration
    def _find_related(self, new_memory: dict) -> list[dict]:
        """Classifier candidates: same category AND (≥N word overlap OR substring),
        ranked best-match first so _integrate_memory compares the new entry
        against the MOST similar existing memory, not just the first encountered.
        """
        new_content = new_memory["content"].lower()
        new_words = set(new_content.split())
        new_category = new_memory["category"]

        scored = []
        for mem in self.memories:
            if mem["status"] != "active" or mem["category"] != new_category:
                continue
            mem_content = mem["content"].lower()
            overlap = len(new_words & set(mem_content.split()))
            contained = new_content in mem_content or mem_content in new_content
            if overlap >= RELATEDNESS_OVERLAP or contained:
                # (containment, overlap) ordering: substring matches rank above
                # mere word overlap; ties broken by overlap count.
                scored.append((1 if contained else 0, overlap, mem))

        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [mem for _, _, mem in scored]

    def _classify_relationship(self, new_memory: dict, existing: dict) -> str:
        """SUB-TASK LLM call: classify new vs existing — drives integration branch."""
        prompt = f"""Compare these two pieces of information about a user:

EXISTING: "{existing['content']}"
NEW: "{new_memory['content']}"

What is the relationship?
- REINFORCEMENT: same information repeated ("User likes coffee" vs "User enjoys coffee")
- UPDATE: new replaces old ("User works at Google" replaces "User works at Microsoft")
- ELABORATION: new adds detail ("User is a senior engineer" elaborates "User is an engineer")
- NEW: different information, both should be kept

Return exactly one word: REINFORCEMENT, UPDATE, ELABORATION, or NEW."""
        try:
            resp = self.llm.generate_single_response(
                prompt, max_tokens=_config.memory_classify_max_tokens
            ).strip().upper()
            if resp in RELATIONSHIPS:
                return resp
        except Exception as e:
            logger.error(f"Classification error: {e}")
        return "NEW"

    def _integrate_memory(self, new_memory: dict) -> bool:
        """Apply the new memory, evolving an existing one if related."""
        related = self._find_related(new_memory)
        if not related:
            self.memories.append(new_memory)
            return True

        existing = related[0]
        relationship = self._classify_relationship(new_memory, existing)

        if relationship == "REINFORCEMENT":
            existing["reinforced"] = existing.get("reinforced", 0) + 1
            existing["updated"] = new_memory["updated"]
            if (existing["retention"] == "ephemeral"
                    and existing["reinforced"] >= PROMOTION_REINFORCEMENT_THRESHOLD):
                existing["retention"] = "situational"
            return True

        if relationship == "UPDATE":
            self.memories.remove(existing)
            existing["status"] = "superseded"
            self.superseded.append(existing)
            new_memory["supersedes"] = existing["id"]
            self.memories.append(new_memory)
            return True

        if relationship == "ELABORATION":
            if len(new_memory["content"]) > len(existing["content"]):
                existing["content"] = new_memory["content"]
            existing["updated"] = new_memory["updated"]
            existing["reinforced"] = existing.get("reinforced", 0) + 1
            new_rank = RETENTION_LEVELS.index(new_memory["retention"])
            existing_rank = RETENTION_LEVELS.index(existing["retention"])
            if new_rank < existing_rank:
                existing["retention"] = new_memory["retention"]
            return True

        # NEW
        self.memories.append(new_memory)
        return True

    # ------------------------------------------------------------------ Retention / Limits
    def _apply_retention_policy(self) -> None:
        """Drop active memories older than their bucket's retention window (with reinforcement bonus)."""
        now = datetime.now()
        remove_ids = set()
        for mem in self.memories:
            if mem["status"] != "active":
                remove_ids.add(mem["id"])
                continue
            days = RETENTION_DAYS.get(mem.get("retention"))
            if days is None:
                continue
            try:
                updated = datetime.strptime(mem["updated"], "%Y-%m-%d-%H-%M")
            except ValueError:
                continue
            effective_days = days + mem.get("reinforced", 0) * _config.reinforcement_retention_bonus_days
            if (now - updated).days > effective_days:
                remove_ids.add(mem["id"])

        if remove_ids:
            expired = [m for m in self.memories if m["id"] in remove_ids]
            self.memories = [m for m in self.memories if m["id"] not in remove_ids]
            self.superseded.extend(expired)
            self._pending_save = True
            logger.info(f"Retention policy removed {len(remove_ids)} memories")

    def _enforce_limits(self) -> None:
        """Hard cap at MAX_MEMORIES. Drop lowest-priority first."""
        if len(self.memories) <= MAX_MEMORIES:
            return

        def priority(mem):
            rank = RETENTION_LEVELS.index(mem.get("retention", "situational"))
            reinforced = mem.get("reinforced", 0)
            try:
                ts = -datetime.strptime(mem["updated"], "%Y-%m-%d-%H-%M").timestamp()
            except ValueError:
                ts = 0
            return (rank, -reinforced, ts)

        self.memories.sort(key=priority)
        while len(self.memories) > MAX_MEMORIES:
            self.superseded.append(self.memories.pop())

    # ------------------------------------------------------------------ Profile
    def _generate_profile(self) -> None:
        if not self.memories:
            self.profile_cache = "No personal information available yet."
            return

        active_by_cat: dict[str, list] = defaultdict(list)
        for m in self.memories:
            if m["status"] == "active":
                active_by_cat[m["category"]].append(m)

        def mem_key(m):
            return (RETENTION_LEVELS.index(m.get("retention", "situational")),
                    -m.get("reinforced", 0), m["updated"])

        def pool(categories, retention=None):
            items = [m for cat in categories for m in active_by_cat.get(cat, [])]
            if retention:
                items = [m for m in items if m["retention"] == retention]
            items.sort(key=mem_key)
            return items

        sections: list[str] = []
        section_specs = [
            ("IDENTITY & BACKGROUND", pool(IDENTITY_CATEGORIES, retention="core"),
             PROFILE_CAPS["identity_background"]),
            ("CURRENT SITUATION",     pool(IDENTITY_CATEGORIES, retention="situational"),
             PROFILE_CAPS["current_situation"]),
            ("PREFERENCES & INTERESTS", pool(["preferences", "interests"]),
             PROFILE_CAPS["preferences_interests"]),
            ("GOALS & TASKS",         pool(["goals", "tasks"]),
             PROFILE_CAPS["goals_tasks"]),
            ("HEALTH",                pool(["health"]),
             PROFILE_CAPS["health"]),
        ]
        for title, items, cap in section_specs:
            if items:
                lines = [f"- {m['content']}" for m in items[:cap]]
                sections.append(f"{title}\n" + "\n".join(lines))

        self.profile_cache = "\n\n".join(sections) if sections else "No notable personal information."

    # ------------------------------------------------------------------ Agent Interface
    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Always-on. Decision is degenerate; per-turn judgement lives inside process()."""
        return True

    def process(self, user_input: str, last_response: Optional[str] = None) -> str:
        """Run decision + execution (separate LLM calls). Always returns the profile."""
        self.metadata = {"new_memories": 0, "memorable": False}

        # DECISION LLM call
        memorable = self._has_memorable_info(user_input)
        self.metadata["memorable"] = memorable

        if memorable:
            # EXECUTION LLM call
            new_memories = self._extract_memories(user_input, last_response)
            added = 0
            for mem in new_memories:
                # SUB-TASK LLM call(s): classify against related existing
                if self._integrate_memory(mem):
                    added += 1
            if added > 0:
                self.metadata["new_memories"] = added
                self._enforce_limits()
                self._generate_profile()
                self._pending_save = True

        if self._pending_save:
            self._save()

        return self.profile_cache
