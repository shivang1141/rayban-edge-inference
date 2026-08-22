import logging

logger = logging.getLogger(__name__)

# Basic heuristic for intent classification
VISION_KEYWORDS = [
    "this", "that", "these", "those",
    "look", "looking at", "see",
    "sign", "translate", "read",
    "front of me", "what is", "who is",
    "what color", "how many"
]

class IntentRouter:
    @staticmethod
    def classify_intent(text: str) -> str:
        """
        Returns 'VISION_REQUIRED' if spatial/visual indicators are found,
        otherwise 'TEXT_ONLY'.
        """
        text_lower = text.lower()
        
        for keyword in VISION_KEYWORDS:
            if keyword in text_lower:
                logger.info(f"Intent Classifier: Match found for keyword '{keyword}' -> VISION_REQUIRED")
                return "VISION_REQUIRED"
        
        logger.info("Intent Classifier: No visual keywords found -> TEXT_ONLY")
        return "TEXT_ONLY"
