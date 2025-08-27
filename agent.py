import time
import openai
from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError
from config import OPENAI_API_KEY, OPENAI_API_BASE, SUPPORT_MODELS, MAX_COMPLETION_TOKENS, MEMORY_SUMMARIZE_THRESHOLD, MEMORY_KEEP_RECENT
from openai_utils import num_tokens_from_string, model2max_context 
import backoff 

# ---- OpenAI Authentication ----
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

# Agent class, responsible for interacting with OpenAI API
class Agent:
    def __init__(self, model_name: str, name: str,
                 temperature: float, sleep_time: float = 0) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.sleep_time = sleep_time
        self.system_prompt = ""

    def _validate_model(self):
        """Validate if the model is supported"""
        if self.model_name not in SUPPORT_MODELS:
            raise ValueError(f"Model {self.model_name} not in {SUPPORT_MODELS}")

    def _limit_tokens(self, max_tokens: int) -> int:
        """Limit token count within reasonable range"""
        return max(1, min(max_tokens, MAX_COMPLETION_TOKENS))

    def _make_openai_request(self, messages: list, max_tokens: int, temperature: float) -> str:
        """Send OpenAI request and handle response"""
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError("API returned unexpected response format.")

    # OpenAI call: query function with retry mechanism
    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError),
        max_tries=20
    )
    def query(self, messages: list, max_tokens: int, temperature: float) -> str:
        time.sleep(self.sleep_time)
        self._validate_model()
        limited_tokens = self._limit_tokens(max_tokens)
        return self._make_openai_request(messages, limited_tokens, temperature)

    # Set system prompt
    def set_meta_prompt(self, prompt: str):
        self.system_prompt = prompt

    # Summarize memory content
    def summarize_memory(self, memory: list) -> str:
        summarizer_prompt = [
            {"role": "system", "content": "Summarize the following debate history into a concise paragraph."},
            *memory,
            {"role": "user", "content": "Please provide the summary."}
        ]
        try:
            return self._make_openai_request(summarizer_prompt, 256, 0.3)
        except Exception as e:
            print(f"[⚠️ Summarization Failed] {e}")
            return "[Summary unavailable]"

    def _prepare_memory_context(self, shared_memory: list) -> list:
        """Prepare memory context, summarize when necessary"""
        if len(shared_memory) <= MEMORY_SUMMARIZE_THRESHOLD:
            return shared_memory
        
        recent = shared_memory[-MEMORY_KEEP_RECENT:]
        summary = self.summarize_memory(shared_memory[:-MEMORY_KEEP_RECENT])
        return [{"role": "system", "content": f"[Debate Summary]: {summary}"}] + recent

    def _calculate_max_tokens(self, messages: list) -> int:
        """Calculate maximum available tokens"""
        ctx_tokens = sum(num_tokens_from_string(m["content"], self.model_name) for m in messages)
        max_context = model2max_context.get(self.model_name, 128_000)
        available_tokens = max_context - ctx_tokens
        return max(64, min(available_tokens, MAX_COMPLETION_TOKENS))

    # Ask function with memory and summarization support
    def ask(self, shared_memory: list, prompt: str, temperature: float = None):
        memory_ctx = self._prepare_memory_context(shared_memory)
        
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + memory_ctx
            + [{"role": "user", "content": f"{self.name}: {prompt}"}]
        )

        max_tokens = self._calculate_max_tokens(messages)
        effective_temperature = temperature if temperature is not None else self.temperature
        
        return self.query(messages, max_tokens, effective_temperature)

# Merged build_agent function
def build_agent(cfg, model_name: str, T: float, sleep: float):
    agent = Agent(model_name, cfg.name, T, sleep)
    agent.set_meta_prompt(cfg.meta_prompt)
    return agent