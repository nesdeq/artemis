"""LangDetect agent: detect the user's language and direct the main LLM to match it."""
import logging
from typing import Optional

import _config
from .Agent import Agent

logger = logging.getLogger(__name__)


class DetectLanguageAgent(Agent):
    """Agent for detecting the language of user input text."""

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        return True

    def process(self, user_input: str, last_response: Optional[str] = None) -> str:
        """Detect language and return instruction."""
        self.metadata = {"detected_language": None}

        if not user_input.strip():
            self.metadata["detected_language"] = "en"
            return "Your response/answer MUST use the language (ISO 639-1): en."

        prompt = (
            "Analyze the following text and determine its language.\n"
            "Respond with only the ISO 639-1 two-letter language code.\n"
            "If you cannot determine the language, respond with 'un' for unknown.\n\n"
            f"Text to analyze: \"{user_input}\"\n\n"
            "Language code:"
        )

        try:
            detected = self.llm.generate_single_response(
                prompt, max_tokens=_config.lang_detect_max_tokens
            ).strip().lower()

            # 'un' is the model's "unknown" sentinel — fall back to en rather than
            # emitting a nonsensical "respond in language: un" directive.
            if (not detected or not detected.isalpha()
                    or len(detected) != 2 or detected == "un"):
                logger.warning(f"Invalid/unknown language code: {detected}, defaulting to 'en'")
                detected = "en"

            self.metadata["detected_language"] = detected
            return f"Your response/answer MUST use the language (ISO 639-1): {detected}."

        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            self.metadata["detected_language"] = "en"
            return "Your response/answer MUST use the language (ISO 639-1): en."
