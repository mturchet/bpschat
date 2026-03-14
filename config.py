import os
from dotenv import load_dotenv

# Load from .env file. Store your HF token in the .env file.
load_dotenv()


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Other options:
# MODEL = "HuggingFaceTB/SmolLM3-3B"
# MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# If you finetune the model or change it in any way, save it to huggingface hub, then set MY_MODEL to your model ID. The model ID is in the format "your-username/your-model-name".
MY_MODEL = None

HF_TOKEN = os.getenv("HF_TOKEN")

# Provider switch:
# - "openai" for local dev/testing
# - "huggingface" for HF inference providers
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

# OpenAI settings (used when LLM_PROVIDER=openai)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
