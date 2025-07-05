from pathlib import Path
from engine import Debate

if __name__ == "__main__":
    news_file = Path(r"example.txt")
    news_text = news_file.read_text(encoding="utf8").strip()

    Debate(model_name="gpt-4o-mini", T=1, sleep=0.5).run(
        news_text=news_text,
        news_path=news_file
    )
 