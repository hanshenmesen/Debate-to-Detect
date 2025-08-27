import os
import re
import json
import itertools
import time
from pathlib import Path
from typing import List, Dict, Optional

from config import (ROLES, PHASES, DIMENSIONS, FREE_ROUNDS, SAVE_DIR, SAVE_FMT, 
                    AUTO_SAVE, RoleConfig, ENABLE_EVIDENCE, EVIDENCE_PHASE, 
                    PHASE_TEMPLATES, SCORING_JUDGE_ROLES, SUMMARY_JUDGE_CONFIG)
from agent import build_agent, Agent
from evidence_system import EvidenceSystem

class Debate:
    def __init__(self, *, model_name="gpt-4o", T=1, sleep=1):
        self.model_name, self.T, self.sleep = model_name, T, sleep
        self.shared: List[Dict] = []       # Complete context for LLM
        self.transcript: List[Dict] = []   # Concise dialogue for saving
        self.domain: str = ""              # News domain
        self.profiles: Dict[str, str] = {}  # Domain-related profile for each role
        self.agents = self._init_agents()   # Initialize role agents
        self.evidence_system = EvidenceSystem(model_name, T) if ENABLE_EVIDENCE else None
        self.evidence_data: Optional[Dict] = None  # Store all evidence data
        self.affirmative_evidence: Optional[Dict] = None  # Available evidence for affirmative
        self.negative_evidence: Optional[Dict] = None     # Available evidence for negative

    def _detect_domain(self, news_text: str) -> str:
        """Detect the domain of the news"""
        detector = Agent(self.model_name, "DomainDetector", temperature=0.0)
        detector.set_meta_prompt(
            "Classify the domain of the following news in one or two words "
            "(e.g., 'politics', 'finance', 'sports', 'technology', 'health')."
        )
        return detector.ask([], news_text).strip()

    def _generate_profiles(self, domain: str) -> Dict[str, str]:
        """Generate domain-related professional profiles for each role"""
        profiles = {}
        for role_name, agent in self.agents.items():
            if not role_name.startswith("Judge"):  # Only generate profiles for debate roles
                prompt = (
                    f"The news domain is '{domain}'. "
                    f"Provide a brief professional profile (1 sentence) for a '{role_name}' "
                    f"role relevant to this domain."
                )
                profiles[role_name] = agent.ask([], prompt, temperature=1).strip()
        return profiles

    def _create_role_configs(self) -> List[RoleConfig]:
        """Create role configuration list"""
        cfgs = []
        for side, duties in ROLES.items():
            if side == "Judge":
                cfgs.extend(self._create_judge_configs(duties))
            else:
                cfgs.extend(self._create_debate_configs(side, duties))
        
        # Add summary judge separately
        cfgs.append(self._create_summary_judge_config())
        return cfgs

    def _create_judge_configs(self, duties: List[tuple]) -> List[RoleConfig]:
        """Create scoring judge configurations"""
        return [
            RoleConfig(
                name=f"Judge_{duty}",
                side="",
                duty=duty,
                meta_prompt=f"You are a judge; please {brief}."
            )
            for duty, brief in duties
        ]

    def _create_summary_judge_config(self) -> RoleConfig:
        """Create summary judge configuration"""
        duty, brief = SUMMARY_JUDGE_CONFIG
        return RoleConfig(
            name=f"Judge_{duty}",
            side="",
            duty=duty,
            meta_prompt=f"You are a judge; please {brief}."
        )

    def _create_debate_configs(self, side: str, duties: List[str]) -> List[RoleConfig]:
        """Create debate role configurations"""
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
                    f"{stance}\n"
                    "When evidence is provided, analyze it carefully and use it to support your arguments. "
                    "If the evidence doesn't support your position, you may choose to focus on other aspects of your argument."
                )
            )
            for duty in duties
        ]

    def _init_agents(self) -> Dict[str, Agent]:
        """Initialize all role agents"""
        cfgs = self._create_role_configs()
        return {c.name: build_agent(c, self.model_name, self.T, self.sleep) for c in cfgs}

    def _get_fixed_stance(self, speaker: str) -> str:
        """Return fixed stance reminder based on speaker identity"""
        stance_map = {
            "Affirmative": "**Your fixed stance is that the news is true.**",
            "Negative": "**Your fixed stance is that the news is false.**"
        }
        return stance_map.get(speaker.split('_')[0], "")

    def _record(self, role: str, prompt: str, reply: str):
        """Record conversation to shared context and transcript"""
        # Context for LLM
        self.shared.extend([
            {"role": "user", "content": f"{role}: {prompt}"},
            {"role": "assistant", "content": f"{role}: {reply}"}
        ])
        
        # Save in concise format
        if self.transcript and self.transcript[-1]["speaker"] == role:
            self.transcript[-1]["text"] += "\n\n" + reply
        else:
            self.transcript.append({"speaker": role, "text": reply})

    def _ask(self, role: str, prompt: str) -> str:
        """Ask specified role and record conversation"""
        agent = self.agents[role]
        reply = agent.ask(self.shared, prompt, temperature=self.T)
        self._record(role, prompt, reply)
        print(f"{role}:\n{reply}\n")
        return reply

    def _last(self, role: str) -> str:
        """Get the last conversation content of specified role"""
        for m in reversed(self.shared):
            if m["role"] == "assistant" and m["content"].startswith(f"{role}:"):
                return m["content"].split(":", 1)[1].strip()
        return ""

    def _opponent(self, role: str) -> str:
        """Get opponent role name"""
        return (role.replace("Affirmative", "Negative")
                if "Affirmative" in role
                else role.replace("Negative", "Affirmative"))

    def _setup_domain_context(self, news_text: str):
        """Set up domain context and role profiles"""
        self.domain = self._detect_domain(news_text)
        self.profiles = self._generate_profiles(self.domain)
        
        # Add profiles to each role's system_prompt
        for role_name, agent in self.agents.items():
            if not role_name.startswith("Judge"):  # Only set profiles for debate roles
                original = agent.system_prompt
                agent.set_meta_prompt(
                    f"{original}\nDomain: {self.domain}\n"
                    f"Profile: {self.profiles.get(role_name, '')}"
                )

    def _gather_evidence(self, news_text: str):
        """Collect evidence and classify by stance"""
        if self.evidence_system and ENABLE_EVIDENCE:
            self.evidence_data = self.evidence_system.gather_evidence(news_text)
            
            # Filter evidence separately for affirmative and negative
            self.affirmative_evidence = self.evidence_system.filter_evidence_by_stance(
                self.evidence_data, 'SUPPORTS_TRUE'
            )
            self.negative_evidence = self.evidence_system.filter_evidence_by_stance(
                self.evidence_data, 'SUPPORTS_FALSE'
            )
            
            # Record evidence collection status
            total_evidence = len(self.evidence_data['evidence'])
            aff_evidence = len(self.affirmative_evidence['evidence'])
            neg_evidence = len(self.negative_evidence['evidence'])
            
            evidence_summary = (
                f"Evidence gathered: {total_evidence} Wikipedia entries found. "
                f"Affirmative favorable: {aff_evidence}, Negative favorable: {neg_evidence}"
            )
            self.transcript.append({"speaker": "Evidence_System", "text": evidence_summary})
            print(f"📊 {evidence_summary}")

    def _run_debate_phases(self, news_text: str):
        """Execute debate phases"""
        for phase, speakers, tpl in PHASES:
            print(f"\n--- {phase} ---")
            
            # Present evidence in specified phase
            if phase == EVIDENCE_PHASE and ENABLE_EVIDENCE and not self.evidence_data:
                self._gather_evidence(news_text)
            
            seq = self._get_speakers_sequence(phase, speakers)
            for turn, sp in enumerate(seq, 1):
                prompt = self._build_prompt(sp, tpl, news_text, turn, phase)
                self._ask(sp, prompt)

    def _get_speakers_sequence(self, phase: str, speakers: List[str]):
        """Get speakers sequence"""
        return speakers if phase != "Free" else itertools.islice(itertools.cycle(speakers), 2 * FREE_ROUNDS)

    def _get_speaker_stance(self, speaker: str) -> str:
        """Get speaker stance"""
        if speaker.startswith("Affirmative"):
            return "SUPPORTS_TRUE"
        elif speaker.startswith("Negative"):
            return "SUPPORTS_FALSE"
        return "NEUTRAL"

    def _get_evidence_for_speaker(self, speaker: str) -> Optional[Dict]:
        """Get favorable evidence for speaker"""
        if not ENABLE_EVIDENCE or not self.evidence_data:
            return None
        
        if speaker.startswith("Affirmative"):
            # Check if there is favorable evidence
            if self.evidence_system.has_favorable_evidence(self.affirmative_evidence, 'SUPPORTS_TRUE'):
                return self.affirmative_evidence
            else:
                print(f"🚫 {speaker} chooses not to present evidence (no favorable evidence found)")
                return None
        elif speaker.startswith("Negative"):
            # Check if there is favorable evidence
            if self.evidence_system.has_favorable_evidence(self.negative_evidence, 'SUPPORTS_FALSE'):
                return self.negative_evidence
            else:
                print(f"🚫 {speaker} chooses not to present evidence (no favorable evidence found)")
                return None
        
        return None

    def _build_prompt(self, speaker: str, template: str, news_text: str, turn: int, phase: str) -> str:
        """Build prompt"""
        stance_reminder = self._get_fixed_stance(speaker)
        
        # If it's free debate phase and has evidence, check whether to provide evidence
        if phase == "Free" and ENABLE_EVIDENCE and self.evidence_data:
            speaker_evidence = self._get_evidence_for_speaker(speaker)
            
            if speaker_evidence and speaker_evidence.get('evidence'):
                # Has favorable evidence, use template with evidence
                evidence_text = self.evidence_system.format_evidence_for_debate(speaker_evidence)
                template = PHASE_TEMPLATES["Free_Evidence"]
                base_prompt = template.format(
                    news=news_text,
                    turn=turn,
                    opp=self._last(self._opponent(speaker)),
                    evidence=evidence_text
                )
                print(f"📋 {speaker} presenting evidence (found {len(speaker_evidence['evidence'])} favorable entries)")
            else:
                # No favorable evidence, use normal template
                base_prompt = template.format(
                    news=news_text,
                    turn=turn,
                    opp=self._last(self._opponent(speaker))
                )
        else:
            # Use standard template for other cases
            base_prompt = template.format(
                news=news_text,
                turn=turn,
                opp=self._last(self._opponent(speaker))
            )
        
        return f"{stance_reminder}\n\n{base_prompt}" if stance_reminder else base_prompt

    def run(self, *, news_text: str, news_path: Path):
        """Run complete debate process"""
        assert news_text, "news_text cannot be empty"
        self.news_stem = news_path.stem

        # Set up domain context
        self._setup_domain_context(news_text)
        print(f"\n=== Debate-to-Detect: Truth/Fake News Analysis | Domain: {self.domain} ===")

        # If presenting evidence in opening phase
        if EVIDENCE_PHASE == "Opening" and ENABLE_EVIDENCE:
            self._gather_evidence(news_text)

        # Execute debate phases
        self._run_debate_phases(news_text)
        
        # Judging phase
        self._judge(news_text)

    def _judge(self, news_text: str):
        """Execute evaluation and verdict"""
        print(f"\n--- Scoring Phase ---")
        
        # First calculate scores for five dimensions
        scores = self._calculate_scores(news_text)
        
        # Determine final verdict
        verdict = self._determine_verdict(scores)
        
        print(f"\n--- Summary Phase ---")
        
        # Then generate summary (including verdict and explanation)
        summary = self._get_debate_summary_with_verdict(news_text, scores, verdict)
        
        print(f"\n[Total] A={scores['Affirmative']}  N={scores['Negative']}")
        print("Verdict:", verdict)

        if AUTO_SAVE:
            self._save(news_text, summary, scores, verdict)

    def _calculate_scores(self, news_text: str) -> Dict[str, int]:
        """Calculate scores for each side - only use scoring judges"""
        scores = {"Affirmative": 0, "Negative": 0}
        
        # Build judge prompt including all evidence information (regardless of stance)
        evidence_context = ""
        if self.evidence_data and ENABLE_EVIDENCE:
            evidence_context = f"\n\nEvidence from Wikipedia:\n{self.evidence_system.format_evidence_for_debate(self.evidence_data)}"
        
        judge_prompt_template = (
            f"The original news article is:\n{news_text}\n"
            f"Debate content:\n{self._get_debate_content()}\n"
            f"{evidence_context}\n"
            "Your task is to assess the factual authenticity of the original news.\n"
            "Affirmatives defend the news as factual.\n"
            "Negatives argue that the news is misleading.\n"
            "Base your judgment on the news content, the debate arguments, and the external evidence provided.\n"
            "Assign an integer score to each side based on how convincingly they support their position relative to the truth.\n"
            "The two scores must add up to exactly 7.\n"
            "Return ONLY the following JSON format and nothing else:\n"
            '{"Affirmative": X, "Negative": Y}'
        )

        # Only use scoring judges
        for duty, _ in SCORING_JUDGE_ROLES:
            judge = self.agents[f"Judge_{duty}"]
            res = self._ask(judge.name, judge_prompt_template)
            scores["Affirmative"] += self._extract(res, "Affirmative")
            scores["Negative"] += self._extract(res, "Negative")

        return scores

    def _get_debate_content(self) -> str:
        """Get debate content"""
        debate_content = []
        for entry in self.transcript:
            # Only include debate roles' speeches, exclude evidence system and judges
            if not entry["speaker"].startswith("Judge") and entry["speaker"] != "Evidence_System":
                debate_content.append(f"{entry['speaker']}: {entry['text']}")
        return "\n\n".join(debate_content)

    def _get_debate_summary_with_verdict(self, news_text: str, scores: Dict[str, int], verdict: str) -> str:
        """Get debate summary with verdict - use summary judge"""
        summary_judge = self.agents["Judge_Summary"]
        
        # Build prompt including score information
        debate_content = self._get_debate_content()
        evidence_context = ""
        if self.evidence_data and ENABLE_EVIDENCE:
            evidence_context = f"\n\nEvidence from Wikipedia:\n{self.evidence_system.format_evidence_for_debate(self.evidence_data)}"
        
        summary_prompt = (
            f"The original news article is:\n{news_text}\n"
            f"Debate content:\n{debate_content}\n"
            f"{evidence_context}\n"
            f"Scoring results: Affirmative={scores['Affirmative']}, Negative={scores['Negative']}\n"
            f"Final verdict: {verdict}\n\n"
            "Please provide your summary based on all the above information."
        )
        
        summary = self._ask(summary_judge.name, summary_prompt)
        return summary

    def _determine_verdict(self, scores: Dict[str, int]) -> str:
        """Determine final verdict"""
        if scores["Affirmative"] > scores["Negative"]:
            return "REAL"
        elif scores["Negative"] > scores["Affirmative"]:
            return "FAKE"
        else:
            return "UNCERTAIN"

    def _save(self, news_text: str, summary: str, scores: Dict[str, int], verdict: str):
        """Save debate results"""
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
        """Save as JSON format"""
        data = {
            "news_text": news_text,
            "domain": self.domain,
            "profiles": self.profiles,
            "evidence_data": self.evidence_data if ENABLE_EVIDENCE else None,
            "summary": summary,
            "scores": scores,
            "verdict": verdict,
            "transcript": self.transcript
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


    def _save_text(self, path: Path, news_text: str, summary: str, 
                   scores: Dict[str, int], verdict: str):
        """Save as text format"""
        with path.open("w", encoding="utf8") as f:
            f.write(f"Verdict: {verdict}\nScores: {scores}\nDomain: {self.domain}\n\n")
            f.write("Profiles:\n")
            for r, p in self.profiles.items():
                f.write(f"{r}: {p}\n")
            
            if self.evidence_data and ENABLE_EVIDENCE:
                f.write(f"\n=== EVIDENCE ===\n")
                f.write(f"Keywords: {self.evidence_data['keywords']}\n")
                f.write(f"Total evidence: {len(self.evidence_data['evidence'])}\n")
                f.write(f"Affirmative favorable: {len(self.affirmative_evidence['evidence'])}\n")
                f.write(f"Negative favorable: {len(self.negative_evidence['evidence'])}\n\n")
                
                for kw, info in self.evidence_data['evidence'].items():
                    f.write(f"{kw} [{info.get('stance', 'NEUTRAL')}]: {info['title']}\n")
                    f.write(f"{info['extract'][:200]}...\n\n")
            
            f.write(f"\n=== NEWS ===\n{news_text}\n\n=== SUMMARY ===\n{summary}\n\n"
                    "=== TRANSCRIPT ===\n")
            for line in self.transcript:
                f.write(f"{line['speaker']}: {line['text']}\n\n")

    @staticmethod
    def _extract(text: str, side: str) -> int:
        """Extract score from text"""
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