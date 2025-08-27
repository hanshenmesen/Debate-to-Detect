from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
import os

# ---------------------------------------------------------------------------
# 0) RoleConfig dataclass (original role_config.py content)
@dataclass
class RoleConfig:
    name: str
    side: Literal["Affirmative", "Negative", ""]
    duty: str  # Opening / Judge_Facts ...
    meta_prompt: str

# ---------------------------------------------------------------------------
# 1) API 配置
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
# ---------------------------------------------------------------------------
# 2) Agent / LLM general constants
# ---------------------------------------------------------------------------
# Supported model list
SUPPORT_MODELS: List[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "gpt-4-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301"
]

# Token and memory management configuration
MAX_COMPLETION_TOKENS: int = 1024      # Maximum tokens generated per response
MEMORY_SUMMARIZE_THRESHOLD: int = 20   # Summarize after this many conversations
MEMORY_KEEP_RECENT: int = 10          # Keep this many recent conversations after summarizing

# ---------------------------------------------------------------------------
# 3) Debate-to-Detect specific configuration
# ---------------------------------------------------------------------------
DETECTION_TASK: bool = True    # Whether to execute "fake news detection" task
FREE_ROUNDS: int = 1          # Free debate rounds (each round includes one speech from A_Free + N_Free)

# ---- Evidence system configuration ----
ENABLE_EVIDENCE: bool = True   # Whether to enable evidence system
EVIDENCE_PHASE: str = "Free"   # In which phase to present evidence ("Free", "Rebuttal", "Opening")

# ---- Role table -------------------------------------------------------------
DEBATE_ROLES = ["Opening", "Rebuttal", "Free", "Closing"]

# Keep only scoring judges, summary judge handled separately
SCORING_JUDGE_ROLES = [
    ("Accuracy", "evaluate **factual accuracy**"),
    ("SourceReliability", "evaluate **source reliability**"),
    ("Reasoning", "evaluate **reasoning and internal consistency**"),
    ("Clarity", "evaluate **clarity and neutrality of language**"),
    ("Ethics", "evaluate **ethical responsibility and potential harm**"),
]

# Summary judge defined separately
SUMMARY_JUDGE_CONFIG = (
    "Summary", 
    """summarize the entire debate in a structured, concise, and objective manner: 
    "Please use the following format:\n\n"
    "**Summary**\n"
    1. **Main Event**: [Briefly describe the news being debated]
    2. **Affirmative Position**: [Key arguments supporting the news as true]
    3. **Negative Position**: [Key arguments claiming the news is false]
    4. **Evidence Presented**: [Summary of any external evidence mentioned]
    5. **Key Points of Contention**: [Main areas of disagreement]
    6. **Conclusion**: [State the verdict and explain the reasoning based on the debate arguments, evidence, and scoring results]  
    Keep each section Concise and Comprehensive."""
)

ROLES: Dict[str, List[str] | List[Tuple[str, str]]] = {
    "Affirmative": DEBATE_ROLES,
    "Negative": DEBATE_ROLES,
    "Judge": SCORING_JUDGE_ROLES,  # 只包含评分法官
}

# ---- Process templates -----------------------------------------------------------
PHASE_TEMPLATES = {
    "Opening": (
        "The news is:\n\"\"\"{news}\"\"\"\n"
        "Give your opening statement defending your fixed stance. Concise and Comprehensive"
    ),
    "Rebuttal": "Please rebut your opponent's opening statement above. Concise and Comprehensive",
    "Free": (
        "Free-debate round {turn}. "
        "Your opponent just said:\n\"{opp}\"\nRespond accordingly. Concise and Comprehensive"
    ),
    "Free_Evidence": (
        "Free-debate round {turn} with evidence support. "
        "Your opponent just said:\n\"{opp}\"\n\n"
        "Available evidence:\n{evidence}\n\n"
        "Use the evidence above to support your arguments and respond accordingly. Concise and Comprehensive"
    ),
    "Closing": "Summarise your team's arguments and present your closing statement. Concise and Comprehensive",
}

# ---- Process (phase name, speaking order, prompt template) -------------------------------
PHASES = [
    ("Opening", ["Affirmative_Opening", "Negative_Opening"], PHASE_TEMPLATES["Opening"]),
    ("Rebuttal", ["Affirmative_Rebuttal", "Negative_Rebuttal"], PHASE_TEMPLATES["Rebuttal"]),
    ("Free", ["Affirmative_Free", "Negative_Free"], PHASE_TEMPLATES["Free"]),
    ("Closing", ["Affirmative_Closing", "Negative_Closing"], PHASE_TEMPLATES["Closing"]),
]

# ---- Scoring dimensions -----------------------------------------------------------
DIMENSIONS = {
    "Accuracy": "factual accuracy",
    "SourceReliability": "source reliability",
    "Reasoning": "reasoning consistency",
    "Clarity": "clarity and neutrality of language",
    "Ethics": "ethical responsibility and potential harm",
}

# ---------------------------------------------------------------------------
# 4) Save settings
# ---------------------------------------------------------------------------
AUTO_SAVE: bool = True         # Auto save after completion
SAVE_DIR: str = "Results"      # Save root directory
SAVE_FMT: str = "json"         # Support "json" or "txt"