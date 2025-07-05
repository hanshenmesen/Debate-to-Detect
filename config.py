from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
import os

# ---------------------------------------------------------------------------
# 0) RoleConfig 数据类 (原 role_config.py 内容)
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
# 2) Agent / LLM 通用常量
# ---------------------------------------------------------------------------
# 支持的模型列表
SUPPORT_MODELS: List[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "gpt-4-0314",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301"
]

# Token和内存管理配置
MAX_COMPLETION_TOKENS: int = 1024      # 每次回复最长生成的token数量
MEMORY_SUMMARIZE_THRESHOLD: int = 20   # 超过多少条对话后进行摘要
MEMORY_KEEP_RECENT: int = 10          # 摘要后保留最近多少条对话

# ---------------------------------------------------------------------------
# 3) Debate-to-Detect 专用配置
# ---------------------------------------------------------------------------
DETECTION_TASK: bool = True    # 是否执行"虚假信息检测"任务
FREE_ROUNDS: int = 1          # 自由辩论回合数（每回合包含 A_Free + N_Free 各一次发言）

# ---- 角色表 -------------------------------------------------------------
DEBATE_ROLES = ["Opening", "Rebuttal", "Free", "Closing"]

JUDGE_ROLES = [
    ("Summary", "summarize the entire debate fairly and objectively"),
    ("Accuracy", "evaluate **factual accuracy**"),
    ("SourceReliability", "evaluate **source reliability**"),
    ("Reasoning", "evaluate **reasoning and internal consistency**"),
    ("Clarity", "evaluate **clarity and neutrality of language**"),
    ("Ethics", "evaluate **ethical responsibility and potential harm**"),
]

ROLES: Dict[str, List[str] | List[Tuple[str, str]]] = {
    "Affirmative": DEBATE_ROLES,
    "Negative": DEBATE_ROLES,
    "Judge": JUDGE_ROLES,
}

# ---- 流程模板 -----------------------------------------------------------
PHASE_TEMPLATES = {
    "Opening": (
        "The news is:\n\"\"\"{news}\"\"\"\n"
        "Give your opening statement defending your fixed stance."
    ),
    "Rebuttal": "Please rebut your opponent's opening statement above.",
    "Free": (
        "Free-debate round {turn}. "
        "Your opponent just said:\n\"{opp}\"\nRespond accordingly."
    ),
    "Closing": "Summarise your team's arguments and present your closing statement.",
}

# ---- 流程（phase 名、发言顺序、提示模板） -------------------------------
PHASES = [
    ("Opening", ["Affirmative_Opening", "Negative_Opening"], PHASE_TEMPLATES["Opening"]),
    ("Rebuttal", ["Affirmative_Rebuttal", "Negative_Rebuttal"], PHASE_TEMPLATES["Rebuttal"]),
    ("Free", ["Affirmative_Free", "Negative_Free"], PHASE_TEMPLATES["Free"]),
    ("Closing", ["Affirmative_Closing", "Negative_Closing"], PHASE_TEMPLATES["Closing"]),
]

# ---- 评分维度 -----------------------------------------------------------
DIMENSIONS = {
    "Accuracy": "factual accuracy",
    "SourceReliability": "source reliability",
    "Reasoning": "reasoning consistency",
    "Clarity": "clarity and neutrality of language",
    "Ethics": "ethical responsibility and potential harm",
}

# ---------------------------------------------------------------------------
# 4) 保存设置
# ---------------------------------------------------------------------------
AUTO_SAVE: bool = True         # 运行完自动保存
SAVE_DIR: str = "Results"      # 保存根目录
SAVE_FMT: str = "json"         # 支持 "json" 或 "txt"