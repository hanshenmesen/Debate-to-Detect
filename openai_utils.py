import tiktoken

# Maximum context window
model2max_context = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4": 7_900,
    "gpt-4-0314": 7_900,
    "gpt-3.5-turbo": 3_900,
    "gpt-3.5-turbo-0301": 3_900
}

class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Return token count; fallback when model tokenizer is unknown."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(string))
