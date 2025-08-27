# Debate-to-Detect (D2D)

## 📌 Introduction
This project implements a **multi-agent debate framework for fake news detection**.  
The core idea is inspired by "truth emerges from debate":  
- **Affirmative Agents**: always argue that the news is true  
- **Negative Agents**: always argue that the news is false  
- **Judge Agents**: evaluate from multiple dimensions (accuracy, source reliability, reasoning consistency, clarity, and ethics), and deliver the final verdict  

Additionally, an **evidence retrieval module** is integrated to automatically extract supporting evidence from Wikipedia, enhancing both reliability and interpretability of the detection results.

---

## ⚙️ Architecture

The system consists of the following components:

1. **`agent.py`**  
   - Defines the `Agent` class that wraps interaction with the OpenAI API.  
   - Provides memory management (summarization + recent context preservation).  
   - Supports role-specific prompts (system prompts).  

2. **`config.py`**  
   - Contains global configuration:  
     - OpenAI API settings (key, base URL)  
     - Supported model list  
     - Memory management (max tokens, summarization threshold)  
     - Debate phases (Opening, Rebuttal, Free Debate, Closing)  
     - Judge role settings  

3. **`engine.py`**  
   - Core `Debate` class that orchestrates the debate workflow:  
     - Detects the news domain and generates role-specific profiles  
     - Initializes all role agents (Affirmative, Negative, Judges)  
     - Runs the debate phases sequentially  
     - Integrates evidence from Wikipedia during designated stages  
     - Produces scores, verdict (REAL / FAKE / UNCERTAIN), summary, and transcript  
   - Results can be saved in `json` or `txt` format.  

4. **`evidence_system.py`**  
   - Evidence collection module:  
     - Extracts keywords from news text  
     - Queries Wikipedia for relevant entries  
     - Uses `EvidenceEvaluator` to assess whether evidence supports TRUE, FALSE, or is NEUTRAL  
   - Injects evidence into debate rounds to improve persuasiveness and interpretability.  

5. **`openai_utils.py`**  
   - Utility functions for token counting and model context limits  
   - Custom exceptions (e.g., quota exceeded, access terminated).  

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

Dependencies include:
- `openai`
- `tiktoken`
- `requests`
- `backoff`
- `dataclasses` (built-in for Python 3.7+)

### 2. Configure environment variables
Before running, set your OpenAI API parameters:
```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

### 3. Run a debate
Example usage:
```python
from pathlib import Path
from engine import Debate

news_text = "Apple will release a new quantum computer next year."
news_path = Path("sample_news.txt")

debate = Debate(model_name="gpt-4o", T=1, sleep=1)
debate.run(news_text=news_text, news_path=news_path)
```

The output will include:
- **Verdict** (REAL / FAKE / UNCERTAIN)  
- **Score distribution** (Affirmative vs Negative)  
- **Debate summary** (with evidence references)  
- **Transcript**  

Results will be automatically saved under the `Results/` directory.

---

## 📂 Example Output

A typical `json` output looks like:
```json
{
  "news_text": "Apple will release a new quantum computer next year.",
  "domain": "technology",
  "profiles": {...},
  "evidence_data": {...},
  "summary": "...",
  "scores": {"Affirmative": 12, "Negative": 8},
  "verdict": "REAL",
  "transcript": [...]
}
```

---

## 🧠 Features
- **Multi-agent structured debate**: enhances interpretability by simulating adversarial reasoning  
- **Evidence retrieval and stance evaluation**: improves factual reliability with Wikipedia evidence  
- **Multi-dimensional judge scoring**: evaluates persuasiveness and truthfulness across five independent aspects  
- **Extensibility**: supports adding more debate rounds, judge dimensions, or swapping different LLMs  

---

## 📜 License
MIT License

---

## 📖 Reference
This project is associated with the following research paper:  
[Debate-to-Detect: Reformulating Misinformation Detection as a Real-World Debate with Large Language Models](http://arxiv.org/abs/2505.18596)
