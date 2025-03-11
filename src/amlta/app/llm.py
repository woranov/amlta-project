from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from amlta.app import config


def get_ollama(model: str | None = None, base_url: str | None = None) -> ChatOllama:
    if model is None:
        model = config.ollama_model
    if base_url is None:
        base_url = config.ollama_base_url

    return ChatOllama(
        model=model, base_url=base_url, num_ctx=2**14, temperature=0.3, seed=42
    )


def get_openai(model="chatgpt-4o-mini", temperature=0.3):
    from dotenv import load_dotenv

    load_dotenv()

    return ChatOpenAI(
        model=model,
        temperature=temperature,
    )
