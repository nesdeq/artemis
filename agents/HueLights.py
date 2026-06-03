"""
HueLights agent: control Philips Hue lights via natural language.

Architecture (see agents/Agent.py and agents/_Agents.md for the LAWS):
- should_process: cheap pre-filter (bridge connected + keyword regex; both
  binary, allowed under rule 5), THEN the DECISION LLM call
  ("is this a light-control command?").
- process: the EXECUTION LLM call returns a structured action JSON;
  code then applies the action via the Hue Bridge.

Decision and execution are SEPARATE LLM calls. NEVER merged.
"""

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from phue import Bridge

from .Agent import Agent
from tools.utils import extract_json
import _config

logger = logging.getLogger(__name__)

# Cheap pre-filter — narrows the input space before the decision LLM call.
_LIGHT_KEYWORDS = re.compile(
    r'\b(lights?|lamps?|bright|dim|dark|hue|bulbs?|(?:turn|switch)\s+(?:on|off))\b',
    re.IGNORECASE,
)

_VALID_ACTIONS = {"on", "off", "adjust"}


class HueLightsAgent(Agent):
    """Controls Philips Hue lights via natural language. Decision + execution as separate LLM calls."""

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)
        self.bridge: Optional[Bridge] = None
        self.lights: Dict[str, Any] = {}
        self.groups: Dict[Any, Any] = {}
        self._id_to_name: Dict[Any, str] = {}
        try:
            self._connect_to_bridge()
        except Exception as e:
            logger.error(f"Error connecting to Hue Bridge: {e}")

    def _connect_to_bridge(self) -> None:
        self.bridge = Bridge(_config.hueip)
        self.lights = self.bridge.get_light_objects('name')
        # id → name map, so group members resolve from cache instead of a
        # per-light HTTP round-trip at action time.
        self._id_to_name = {light.light_id: name for name, light in self.lights.items()}
        self.groups = self.bridge.get_group()
        if not self.lights and not self.groups:
            logger.warning("No lights or groups found.")
        else:
            logger.info(f"Found {len(self.lights)} lights, {len(self.groups)} groups")

    # ----------------------------------------------------------------- Decision
    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Cheap pre-filter, then DECISION LLM call."""
        if not self.lights and not self.groups:
            return False
        if not _LIGHT_KEYWORDS.search(user_input):
            return False
        return self._is_light_command(user_input)

    def _is_light_command(self, user_input: str) -> bool:
        """DECISION LLM call: is this an actual light-control command?"""
        prompt = f"""Is this a command to control Philips Hue lights (turn on/off, dim, brighten, change color)?

Answer with exactly one word: YES or NO.

Message: "{user_input}"

Answer:"""
        try:
            resp = self.llm.generate_single_response(
                prompt, max_tokens=_config.search_decision_max_tokens
            ).strip().upper()
            return resp.startswith("YES")
        except Exception as e:
            logger.error(f"Light-command decision error: {e}")
            return False

    # ----------------------------------------------------------------- Execution
    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        self.metadata = {}
        action = self._plan_action(user_input)
        if not action:
            return None
        self.metadata = {
            "action": action["action"],
            "targets": action["targets"],
            "description": action["description"],
        }
        self._apply_action(action)
        return f"Action executed: {action['description']}"

    def _plan_action(self, user_input: str) -> Optional[Dict[str, Any]]:
        """EXECUTION LLM call: produce a structured action JSON."""
        light_names = list(self.lights.keys())
        group_names = [g['name'] for g in self.groups.values()]

        prompt = f"""Translate this user request into a JSON action for Philips Hue lights.

Available lights: {', '.join(light_names) or '(none)'}
Available groups: {', '.join(group_names) or '(none)'}
Current time: {datetime.now().strftime("%H:%M")}

Output a JSON object with this schema:
{{
  "action": "on" | "off" | "adjust",
  "targets": ["light_or_group_name", ...] OR "all",
  "brightness": <number 1-254>,    // optional; required for "adjust" if dimming
  "color": [x, y] OR null,          // optional CIE xy
  "description": "short description of the action"
}}

User: "{user_input}"

Return ONLY the JSON object."""
        try:
            response = self.llm.generate_single_response(
                prompt, max_tokens=_config.hue_action_max_tokens
            )
            action = extract_json(response)
            if isinstance(action, dict) and self._validate_action(action):
                return action
        except Exception as e:
            logger.error(f"Action planning error: {e}")
        return None

    def _validate_action(self, action: Dict[str, Any]) -> bool:
        if action.get("action") not in _VALID_ACTIONS:
            return False
        if "description" not in action:
            return False
        # targets must be "all" or a non-empty list, else the action silently
        # resolves to zero lights yet reports success.
        targets = action.get("targets")
        if targets != "all" and not (isinstance(targets, list) and targets):
            return False
        # brightness/color come from the LLM and are written straight to
        # hardware — bound them to the Hue API's documented ranges.
        brightness = action.get("brightness")
        if brightness is not None and not (
            isinstance(brightness, int) and 1 <= brightness <= 254  # Hue brightness range
        ):
            return False
        color = action.get("color")
        if color is not None and not (
            isinstance(color, (list, tuple)) and len(color) == 2
            and all(isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in color)  # CIE xy
        ):
            return False
        return True

    def _apply_action(self, action: Dict[str, Any]) -> None:
        for light in self._resolve_targets(action.get("targets", [])):
            try:
                self._apply_to_light(light, action)
            except Exception as e:
                logger.error(f"Error applying action to light: {e}")

    def _resolve_targets(self, targets) -> List[Any]:
        """Resolve 'all' / light-names / group-names to a flat unique list of light objects."""
        if targets == "all":
            return list(self.lights.values())
        if not isinstance(targets, list):
            return []

        seen: set = set()
        resolved: List[Any] = []
        for target in targets:
            for light in self._lights_for_target(target):
                key = id(light)
                if key not in seen:
                    seen.add(key)
                    resolved.append(light)
        return resolved

    def _lights_for_target(self, target: str) -> List[Any]:
        if target in self.lights:
            return [self.lights[target]]
        group = next((g for g in self.groups.values() if g.get('name') == target), None)
        if not group:
            return []
        # Resolve member ids via the cached id→name map (built at connect time)
        # instead of a blocking bridge.get_light() HTTP call per member.
        out = []
        for light_id in group.get('lights', []):
            try:
                key = int(light_id)
            except (TypeError, ValueError):
                continue
            name = self._id_to_name.get(key)
            if name and name in self.lights:
                out.append(self.lights[name])
        return out

    def _apply_to_light(self, light: Any, action: Dict[str, Any]) -> None:
        if action["action"] == "off":
            light.on = False
            return
        light.on = True
        if action.get("brightness") is not None:
            light.brightness = action["brightness"]
        if action.get("color"):
            light.xy = action["color"]
