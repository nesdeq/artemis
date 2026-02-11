# HueLights.py
import re
import logging
from datetime import datetime
from typing import Dict, Optional, Any

from phue import Bridge

from .Agent import Agent
from tools.utils import extract_json
import _config

logger = logging.getLogger(__name__)

# Fast keyword check for should_process (avoids LLM call for irrelevant input)
_LIGHT_KEYWORDS = re.compile(
    r'\b(lights?|lamps?|bright|dim|dark|hue|bulbs?|turn\s+on|turn\s+off|switch\s+on|switch\s+off)\b',
    re.IGNORECASE
)


class HueLightsAgent(Agent):
    """Agent for controlling Philips Hue lights.
    
    This agent interprets user commands to control Philips Hue lights and groups,
    handling actions like turning lights on/off and adjusting brightness/color.
    """
    
    def __init__(self, name: str, user: Optional[str] = None) -> None:
        """Initialize the HueLightsAgent.
        
        Args:
            name: The name of the agent
            user: Optional user identifier
        """
        super().__init__(name, user)
        self.lights = None
        self.groups = None
        
        try:
            self._connect_to_bridge()
        except Exception as e:
            logger.error(f"Error connecting to Hue Bridge: {str(e)}")

    def _connect_to_bridge(self) -> None:
        """Connect to the Philips Hue Bridge and get lights and groups."""
        self.bridge = Bridge(_config.hueip)
        # self.bridge.connect()  # Uncomment if you need interactive connection
        
        self.lights = self.bridge.get_light_objects('name')
        self.groups = self.bridge.get_group()
        
        if not self.lights and not self.groups:
            logger.warning("No lights or groups found. Check if the lights are connected and configured properly.")
        else:
            logger.info(f"Found {len(self.lights)} lights and {len(self.groups)} groups:")
            
            for light_name, light_object in self.lights.items():
                logger.info(f"- Light: {light_name} ({light_object.type})")
                
            for group_id, group_data in self.groups.items():
                logger.info(f"- Group: {group_data['name']} (ID: {group_id})")

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Fast keyword check - LLM classification deferred to process()."""
        if not self.lights and not self.groups:
            return False
        return bool(_LIGHT_KEYWORDS.search(user_input))

    def process(self, user_input: str, last_response: Optional[str] = None) -> Optional[str]:
        """Determine and execute light control action."""
        action = self.determine_action(user_input)
        if not action:
            return None

        self.metadata = {
            "action": action["action"],
            "targets": action["targets"],
            "description": action["description"]
        }

        self.execute_action(action)
        return f"Action executed: {action['description']}"

    def determine_action(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Determine the light control action from user input.

        Args:
            user_input: The text input from the user

        Returns:
            Optional[Dict[str, Any]]: Dictionary with action details or None if no action
        """
        light_names = list(self.lights.keys()) if self.lights else []
        group_names = [group_data['name'] for group_data in self.groups.values()] if self.groups else []

        prompt = f"""
Analyze the following user input to determine if it is a command to control Hue lights.
If it is, return a valid JSON object with this structure:
{{
    "action": "on" | "off" | "adjust",
    "targets": ["light_name1", "group_name1"] or "all",
    "brightness": number,  // 0 for off, or a value between 1 and 254,
    "color": [x, y] or null,
    "description": "A brief description of the action"
}}
If it is not related to lighting control, return null.

User Input: {user_input}
Available lights: {', '.join(light_names)}
Available groups: {', '.join(group_names)}
Current Time: {datetime.now().strftime("%H:%M")}

Ensure that the output is valid JSON.
"""
        try:
            response = self.llm.generate_single_response(prompt)
            action = extract_json(response)

            # Validate the parsed action
            if action and isinstance(action, dict) and self._validate_action(action):
                return action

        except Exception as e:
            logger.error(f"Error determining light action: {str(e)}")

        return None

    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate that an action dictionary has the required fields.
        
        Args:
            action: Dictionary with action details
            
        Returns:
            bool: True if the action is valid, False otherwise
        """
        required_fields = ["action", "targets", "description"]
        return all(field in action for field in required_fields)

    def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute a light control action.
        
        Args:
            action: Dictionary with action details
        """
        targets = action['targets']
        
        # Handle 'all' targets
        if targets == 'all':
            targets = list(self.lights.keys()) + [group['name'] for group in self.groups.values()]
            
        for target in targets:
            try:
                # Check if target is a light
                if target in self.lights:
                    self.execute_light_action(self.lights[target], action)
                else:
                    # Check if target is a group
                    group = next((group for group in self.groups.values() if group['name'] == target), None)
                    if group:
                        for light_id in group['lights']:
                            light = self.bridge.get_light(int(light_id))
                            if light and light['name'] in self.lights:
                                self.execute_light_action(self.lights[light['name']], action)
            except Exception as e:
                logger.error(f"Error executing action on target {target}: {str(e)}")

    def execute_light_action(self, light: Any, action: Dict[str, Any]) -> None:
        """Execute an action on a specific light.
        
        Args:
            light: Light object to control
            action: Dictionary with action details
        """
        try:
            if action['action'] == 'on':
                light.on = True
            elif action['action'] == 'off':
                light.on = False
            elif action['action'] == 'adjust':
                light.on = True
                
                if 'brightness' in action:
                    light.brightness = action['brightness']
                    
                if action.get('color'):
                    light.xy = action['color']
                    
        except Exception as e:
            logger.error(f"Error executing light action: {str(e)}")