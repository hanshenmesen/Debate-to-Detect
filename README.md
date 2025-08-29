# Debate-to-Detect (D2D) 
Version 2.0 (Evidence-Based)

## 📌 Introduction
This project implements a **multi-agent debate framework for fake news detection**.  
The core idea is inspired by "the truth becomes clearer from debate":  
- **Affirmative Agents**: always argue that the news is true  
- **Negative Agents**: always argue that the news is false  
- **Judge Agents**: evaluate from multiple dimensions (accuracy, source reliability, reasoning consistency, clarity, and ethics), and deliver the final verdict  

Additionally, an **evidence retrieval module** is integrated to automatically extract supporting evidence from Wikipedia, enhancing both reliability and interpretability of the detection results.

The code only include simplified prompts for demonstration.
For achieving more comprehensive results, it is recommended to use stronger models together with more detailed and structured prompts. By enriching prompts with role-specific instructions, evaluation criteria, and constraints, the system can deliver higher reliability, interpretability, and overall performance.


---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
Before running, set your OpenAI API parameters:
```bash
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
```

### 3. Run a debate
Example usage:
```python
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

## 📜 License
MIT License

---

## 📖 Reference
This project is associated with the following research paper:  
[Debate-to-Detect: Reformulating Misinformation Detection as a Real-World Debate with Large Language Models](http://arxiv.org/abs/2505.18596)
