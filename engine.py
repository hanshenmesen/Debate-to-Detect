import os
import re
import json
import itertools
import time
from pathlib import Path
from typing import List, Dict

from config import ROLES, PHASES, DIMENSIONS, FREE_ROUNDS, SAVE_DIR, SAVE_FMT, AUTO_SAVE, RoleConfig
from agent import build_agent, Agent

class Debate:
    def __init__(self, *, model_name="gpt-4o", T=1, sleep=1):
        self.model_name, self.T, self.sleep = model_name, T, sleep
        self.shared: List[Dict] = []       # 传给 LLM 的完整上下文
        self.transcript: List[Dict] = []   # 保存时用的简洁对话
        self.domain: str = ""              # 新闻领域
        self.profiles: Dict[str, str] = {}  # 每个角色的领域相关简介
        self.agents = self._init_agents()   # 初始化角色代理

    def _detect_domain(self, news_text: str) -> str:
        """检测新闻所属领域"""
        detector = Agent(self.model_name, "DomainDetector", temperature=0.0)
        detector.set_meta_prompt(
            "Classify the domain of the following news in one or two words "
            "(e.g., 'politics', 'finance', 'sports', 'technology', 'health')."
        )
        return detector.ask([], news_text).strip()

    def _generate_profiles(self, domain: str) -> Dict[str, str]:
        """为每个角色生成领域相关的专业简介"""
        profiles = {}
        for role_name, agent in self.agents.items():
            prompt = (
                f"The news domain is '{domain}'. "
                f"Provide a brief professional profile (1 sentence) for a '{role_name}' "
                f"role relevant to this domain."
            )
            profiles[role_name] = agent.ask([], prompt, temperature=1).strip()
        return profiles

    def _create_role_configs(self) -> List[RoleConfig]:
        """创建角色配置列表"""
        cfgs = []
        for side, duties in ROLES.items():
            if side == "Judge":
                cfgs.extend(self._create_judge_configs(duties))
            else:
                cfgs.extend(self._create_debate_configs(side, duties))
        return cfgs

    def _create_judge_configs(self, duties: List[tuple]) -> List[RoleConfig]:
        """创建评判角色配置"""
        return [
            RoleConfig(
                name=f"Judge_{duty}",
                side="",
                duty=duty,
                meta_prompt=f"You are a judge; please {brief}."
            )
            for duty, brief in duties
        ]

    def _create_debate_configs(self, side: str, duties: List[str]) -> List[RoleConfig]:
        """创建辩论角色配置"""
        stance = (
            "You believe the news is true and need to argue in its favor."
            if side == "Affirmative"
            else "You believe the news is false and need to argue against it."
        )
        return [
            RoleConfig(
                name=f"{side}_{duty}",
                side=side,
                duty=duty,
                meta_prompt=(
                    f"You are the {duty.lower()} speaker on the {side.lower()} side.\n"
                    f"{stance}"
                )
            )
            for duty in duties
        ]

    def _init_agents(self) -> Dict[str, Agent]:
        """初始化所有角色代理"""
        cfgs = self._create_role_configs()
        return {c.name: build_agent(c, self.model_name, self.T, self.sleep) for c in cfgs}

    def _get_fixed_stance(self, speaker: str) -> str:
        """根据发言者身份返回固定立场提醒"""
        stance_map = {
            "Affirmative": "**Your fixed stance is that the news is true.**",
            "Negative": "**Your fixed stance is that the news is false.**"
        }
        return stance_map.get(speaker.split('_')[0], "")  # 更简洁的写法，避免遍历

    def _record(self, role: str, prompt: str, reply: str):
        """记录对话到共享上下文和转录本"""
        # 给 LLM 的上下文
        self.shared.extend([
            {"role": "user", "content": f"{role}: {prompt}"},
            {"role": "assistant", "content": f"{role}: {reply}"}
        ])
        
        # 保存用简洁格式
        if self.transcript and self.transcript[-1]["speaker"] == role:
            self.transcript[-1]["text"] += "\n\n" + reply
        else:
            self.transcript.append({"speaker": role, "text": reply})

    def _ask(self, role: str, prompt: str) -> str:
        """向指定角色提问并记录对话"""
        agent = self.agents[role]
        reply = agent.ask(self.shared, prompt, temperature=self.T)
        self._record(role, prompt, reply)
        print(f"{role}:\n{reply}\n")
        return reply

    def _last(self, role: str) -> str:
        """获取指定角色的最后一次对话内容"""
        for m in reversed(self.shared):
            if m["role"] == "assistant" and m["content"].startswith(f"{role}:"):
                return m["content"].split(":", 1)[1].strip()
        return ""

    def _setup_domain_context(self, news_text: str):
        """设置领域上下文和角色简介"""
        self.domain = self._detect_domain(news_text)
        self.profiles = self._generate_profiles(self.domain)
        
        # 将简介加入各角色 system_prompt
        for role_name, agent in self.agents.items():
            original = agent.system_prompt
            agent.set_meta_prompt(
                f"{original}\nDomain: {self.domain}\n"
                f"Profile: {self.profiles.get(role_name, '')}"
            )

    def _run_debate_phases(self, news_text: str):
        """执行辩论各阶段"""
        for phase, speakers, tpl in PHASES:
            print(f"\n--- {phase} ---")
            seq = self._get_speakers_sequence(phase, speakers)
            for turn, sp in enumerate(seq, 1):
                prompt = self._build_prompt(sp, tpl, news_text, turn)
                self._ask(sp, prompt)

    def _get_speakers_sequence(self, phase: str, speakers: List[str]):
        """获取发言者序列"""
        return speakers if phase != "Free" else itertools.islice(itertools.cycle(speakers), 2 * FREE_ROUNDS)

    def _build_prompt(self, speaker: str, template: str, news_text: str, turn: int) -> str:
        """构建提示语"""
        stance_reminder = self._get_fixed_stance(speaker)
        base_prompt = template.format(
            news=news_text,
            turn=turn,
            opp=self._last(self._opponent(speaker))
        )
        return f"{stance_reminder}\n\n{base_prompt}" if stance_reminder else base_prompt

    def run(self, *, news_text: str, news_path: Path):
        """运行完整的辩论流程"""
        assert news_text, "news_text 不能为空"
        self.news_stem = news_path.stem

        # 设置领域上下文
        self._setup_domain_context(news_text)
        print(f"\n=== Debate-to-Detect: 真伪新闻辨析 | Domain: {self.domain} ===")

        # 执行辩论阶段
        self._run_debate_phases(news_text)
        
        # 评判阶段
        self._judge(news_text)

    def _judge(self, news_text: str):
        """执行评判和判决"""
        summary = self._get_debate_summary()
        scores = self._calculate_scores(news_text, summary)
        verdict = self._determine_verdict(scores)
        
        print(f"\n[Total] A={scores['Affirmative']}  N={scores['Negative']}")
        print("Verdict:", verdict)

        if AUTO_SAVE:
            self._save(news_text, summary, scores, verdict)

    def _get_debate_summary(self) -> str:
        """获取辩论总结"""
        summary_judge = self.agents["Judge_Summary"]
        body = "\n".join(self._last(n) for n in self.agents if not n.startswith("Judge"))
        summary = summary_judge.ask(self.shared, body, temperature=0.2)
        print(f"\n[Debate Summary]\n{summary}\n")
        return summary

    def _calculate_scores(self, news_text: str, summary: str) -> Dict[str, int]:
        """计算各方得分"""
        scores = {"Affirmative": 0, "Negative": 0}
        
        judge_prompt_template = (
            f"The original news article is:\n{news_text}\n"
            f"Debate Summary:\n{summary}\n"
            "Your task is to assess the factual authenticity of the original news.\n"
            "Affirmatives defend the news as factual.\n"
            "Negatives argue that the news is misleading.\n"
            "Base your judgment both on the news content and the arguments from the debate.\n"
            "Assign an integer score to each side based on how convincingly they support "
            "their position relative to the truth.\n"
            "The two scores must add up to exactly 7.\n"
            "Return ONLY the following JSON format and nothing else:\n"
            '{"Affirmative": X, "Negative": Y}'
        )

        for key in DIMENSIONS.keys():
            judge = self.agents[f"Judge_{key}"]
            res = self._ask(judge.name, judge_prompt_template)
            scores["Affirmative"] += self._extract(res, "Affirmative")
            scores["Negative"] += self._extract(res, "Negative")

        return scores

    def _determine_verdict(self, scores: Dict[str, int]) -> str:
        """确定最终判决"""
        if scores["Affirmative"] > scores["Negative"]:
            return "REAL"
        elif scores["Negative"] > scores["Affirmative"]:
            return "FAKE"
        else:
            return "UNCERTAIN"

    def _save(self, news_text: str, summary: str, scores: Dict[str, int], verdict: str):
        """保存辩论结果"""
        os.makedirs(SAVE_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        out = Path(SAVE_DIR) / f"{self.news_stem}_{timestamp}.{SAVE_FMT.lower()}"

        if SAVE_FMT.lower() == "json":
            self._save_json(out, news_text, summary, scores, verdict)
        else:
            self._save_text(out, news_text, summary, scores, verdict)

        print(f"💾 Saved to {out}")

    def _save_json(self, path: Path, news_text: str, summary: str, 
                   scores: Dict[str, int], verdict: str):
        """保存为JSON格式"""
        data = {
            "news_text": news_text,
            "domain": self.domain,
            "profiles": self.profiles,
            "summary": summary,
            "scores": scores,
            "verdict": verdict,
            "transcript": self.transcript
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")

    def _save_text(self, path: Path, news_text: str, summary: str, 
                   scores: Dict[str, int], verdict: str):
        """保存为文本格式"""
        with path.open("w", encoding="utf8") as f:
            f.write(f"Verdict: {verdict}\nScores: {scores}\nDomain: {self.domain}\n\n")
            f.write("Profiles:\n")
            for r, p in self.profiles.items():
                f.write(f"{r}: {p}\n")
            f.write(f"\n=== NEWS ===\n{news_text}\n\n=== SUMMARY ===\n{summary}\n\n"
                    "=== TRANSCRIPT ===\n")
            for line in self.transcript:
                f.write(f"{line['speaker']}: {line['text']}\n\n")

    @staticmethod
    def _extract(text: str, side: str) -> int:
        """从文本中提取分数"""
        try:
            block = re.search(r"\{.*?\}", text, re.S)
            if block:
                data = json.loads(block.group(0))
                return int(data.get(side, 0))
        except Exception:
            pass

        abbr = side[0]
        m = re.search(fr"(?:{side}|{abbr})\s*[:=]\s*\[?\s*(\d)", text, re.I)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _opponent(role: str) -> str:
        """获取对手角色名称"""
        return (role.replace("Affirmative", "Negative")
                if "Affirmative" in role
                else role.replace("Negative", "Affirmative"))
