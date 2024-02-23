import tiktoken

GPT4_TURBO_PRICE_1K_INPUT = 0.070
GPT4_TURBO_PRICE_1K_OUTPUT = 0.208

ENCODER = tiktoken.encoding_for_model("gpt-4-turbo")


def count_1k_tokens(text: str) -> float:
    return len(ENCODER.encode(text)) / 1000


def calculate_price(full_text: str, full_output: str) -> float:
    return GPT4_TURBO_PRICE_1K_INPUT * count_1k_tokens(
        full_text,
    ) + GPT4_TURBO_PRICE_1K_OUTPUT * count_1k_tokens(full_output)


calculate_price(full_text="This is a test", full_output="This is a test.")
