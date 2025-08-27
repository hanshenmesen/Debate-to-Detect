import re
import json
import requests
from typing import List, Dict, Optional
from agent import Agent

class EvidenceSystem:
    """Evidence system responsible for keyword extraction and Wikipedia retrieval"""
    
    def __init__(self, model_name: str, temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.keyword_extractor = self._create_keyword_extractor()
        self.evidence_evaluator = self._create_evidence_evaluator()
        
    def _create_keyword_extractor(self) -> Agent:
        """Create keyword extractor"""
        extractor = Agent(self.model_name, "KeywordExtractor", self.temperature)
        extractor.set_meta_prompt(
            "Extract the most important entities and key concepts from the given news text. "
            "Return ONLY a JSON array of keywords, like: [\"keyword1\", \"keyword2\", \"keyword3\"]. "
            "Focus on proper nouns and factual claims that can be verified."
        )
        return extractor
    
    def _create_evidence_evaluator(self) -> Agent:
        """Create evidence evaluator"""
        evaluator = Agent(self.model_name, "EvidenceEvaluator", self.temperature)
        evaluator.set_meta_prompt(
            "You are an evidence evaluator. Given a news claim and supporting evidence from Wikipedia, "
            "decide whether the evidence indicates the claim is TRUE, FALSE, or inconclusive. "
            "Respond with only one label: 'SUPPORTS_TRUE', 'SUPPORTS_FALSE', or 'NEUTRAL'."
        )
        return evaluator
    
    def extract_keywords(self, news_text: str) -> List[str]:
        """Extract keywords from news text"""
        try:
            response = self.keyword_extractor.ask([], news_text, self.temperature)
            # Try to parse JSON
            keywords = self._parse_keywords_response(response)
            return keywords[:5]  # Limit to maximum 5 keywords
        except Exception as e:
            print(f"[⚠️ Keyword extraction failed] {e}")
            return []
    
    def _parse_keywords_response(self, response: str) -> List[str]:
        """Parse keyword extraction response"""
        try:
            # Try to parse JSON directly
            if response.strip().startswith('['):
                return json.loads(response.strip())
            
            # Try to find JSON array
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: split by lines
            lines = [line.strip(' -•*') for line in response.split('\n') if line.strip()]
            return [line for line in lines if line and not line.startswith('[')][:4]
            
        except Exception:
            # Final fallback option
            return []
    
    def search_wikipedia(self, keyword: str) -> Optional[Dict]:
        """Search Wikipedia for keyword"""
        try:
            # Search API
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + keyword.replace(' ', '_')
            headers = {'User-Agent': 'DebateBot/1.0'}
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', keyword),
                    'extract': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'thumbnail': data.get('thumbnail', {}).get('source', '') if data.get('thumbnail') else ''
                }
            else:
                return None
                
        except Exception as e:
            print(f"[⚠️ Wikipedia search failed for '{keyword}'] {e}")
            return None
    
    def evaluate_evidence_stance(self, news_text: str, evidence_info: Dict) -> str:
        """Evaluate evidence stance on news authenticity"""
        evaluation_prompt = (
            f"News claim: {news_text}\n\n"
            f"Evidence from Wikipedia:\n"
            f"Title: {evidence_info['title']}\n"
            f"Content: {evidence_info['extract']}\n\n"
            f"Does this evidence support the news being TRUE, FALSE, or is it NEUTRAL?"
        )
        
        try:
            response = self.evidence_evaluator.ask([], evaluation_prompt, self.temperature)
            response = response.strip().upper()
            
            if 'SUPPORTS_TRUE' in response or 'TRUE' in response:
                return 'SUPPORTS_TRUE'
            elif 'SUPPORTS_FALSE' in response or 'FALSE' in response:
                return 'SUPPORTS_FALSE'
            else:
                return 'NEUTRAL'
        except Exception as e:
            print(f"[⚠️ Evidence evaluation failed] {e}")
            return 'NEUTRAL'
    
    def gather_evidence(self, news_text: str) -> Dict[str, any]:
        """Collect evidence: extract keywords and search Wikipedia"""
        print("\n--- Evidence Gathering Phase ---")
        
        # 1. Extract keywords
        keywords = self.extract_keywords(news_text)
        print(f"📝 Extracted keywords: {keywords}")
        
        # 2. Search Wikipedia and evaluate evidence stance
        evidence = {}
        for keyword in keywords:
            print(f"🔍 Searching Wikipedia for: {keyword}")
            wiki_result = self.search_wikipedia(keyword)
            if wiki_result:
                # Evaluate evidence stance
                stance = self.evaluate_evidence_stance(news_text, wiki_result)
                wiki_result['stance'] = stance
                evidence[keyword] = wiki_result
                print(f"✅ Found: {wiki_result['title']} (Stance: {stance})")
            else:
                print(f"❌ No Wikipedia entry found for: {keyword}")
        
        return {
            'keywords': keywords,
            'evidence': evidence
        }
    
    def filter_evidence_by_stance(self, evidence_data: Dict, desired_stance: str) -> Dict:
        """Filter evidence by stance"""
        if not evidence_data.get('evidence'):
            return evidence_data
        
        filtered_evidence = {}
        for keyword, info in evidence_data['evidence'].items():
            evidence_stance = info.get('stance', 'NEUTRAL')
            
            # Keep evidence if it supports desired stance or is neutral
            if evidence_stance == desired_stance or evidence_stance == 'NEUTRAL':
                filtered_evidence[keyword] = info
        
        return {
            'keywords': evidence_data['keywords'],
            'evidence': filtered_evidence
        }
    
    def format_evidence_for_debate(self, evidence_data: Dict) -> str:
        """Format evidence for debate"""
        if not evidence_data.get('evidence'):
            return "No external evidence was found to support the claims."
        
        evidence_text = "**Available Evidence from Wikipedia:**\n\n"
        
        for keyword, info in evidence_data['evidence'].items():
            evidence_text += f"**{info['title']}:**\n"
            evidence_text += f"{info['extract'][:300]}{'...' if len(info['extract']) > 300 else ''}\n"
            evidence_text += f"Source: {info['url']}\n\n"
        
        return evidence_text
    
    def has_favorable_evidence(self, evidence_data: Dict, stance: str) -> bool:
        """Check if there is favorable evidence"""
        if not evidence_data.get('evidence'):
            return False
        
        for keyword, info in evidence_data['evidence'].items():
            evidence_stance = info.get('stance', 'NEUTRAL')
            if evidence_stance == stance:
                return True
        
        return False