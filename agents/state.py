from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ChatState:
    stage: str = "fallback"                      # Current high-level stage
    user_message: str = ""                       # Latest user utterance
    resume: Optional[Dict[str, Any]] = None      # Parsed résumé
    preferences: Optional[Dict[str, Any]] = None # User’s filters
    positions: List[Dict[str, Any]] = field(default_factory=list)
    confirmation: Optional[bool] = None          # Did the user say “yes”?
    emails_sent: bool = False
    replies: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_done: bool = False                 # After follow-up completed
