# LangDetect.py
import logging
from typing import Optional

from .Agent import Agent

# Configure logger
logger = logging.getLogger(__name__)


class DetectLanguageAgent(Agent):
    """Agent for detecting the language of user input text."""

    def __init__(self, name: str, user: Optional[str] = None) -> None:
        super().__init__(name, user)

    def should_process(self, user_input: str, last_response: Optional[str] = None) -> bool:
        """Always runs."""
        return True

    def process(self, user_input: str, last_response: Optional[str] = None) -> str:
        """Detect language and return instruction."""
        self.metadata = {"detected_language": None}

        if not user_input.strip():
            self.metadata["detected_language"] = "en"
            return "Your response/answer MUST use the language (ISO 639-1): en."

        prompt = """Analyze the following text and determine its language.
Respond with only the ISO 639-1 two-letter language code.
If you cannot determine the language, respond with 'un' for unknown.

Text to analyze: "{}"

Language code:""".format(user_input)

        try:
            detected_lang = self.llm.generate_single_response(prompt, max_tokens=64).strip().lower()

            if not detected_lang or not detected_lang.isalpha() or len(detected_lang) != 2:
                logger.warning(f"Invalid language code: {detected_lang}, defaulting to 'en'")
                detected_lang = "en"

            self.metadata["detected_language"] = detected_lang
            return f"Your response/answer MUST use the language (ISO 639-1): {detected_lang}."

        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            self.metadata["detected_language"] = "en"
            return "Your response/answer MUST use the language (ISO 639-1): en."