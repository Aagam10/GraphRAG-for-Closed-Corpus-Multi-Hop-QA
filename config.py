import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

for d in [DATA_RAW, DATA_PROCESSED, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM Provider ─────────────────────────────────────────────────────
# Set LLM_PROVIDER in .env to one of: "gemini", "groq", "ollama", "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

if LLM_PROVIDER == "gemini":
    LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    EXTRACTION_MODEL = "gemini-2.0-flash"
    GENERATION_MODEL = "gemini-2.0-flash"

elif LLM_PROVIDER == "groq":
    LLM_API_KEY = os.getenv("GROQ_API_KEY", "")
    LLM_BASE_URL = "https://api.groq.com/openai/v1"
    EXTRACTION_MODEL = "llama-3.3-70b-versatile"
    GENERATION_MODEL = "llama-3.3-70b-versatile"

elif LLM_PROVIDER == "ollama":
    LLM_API_KEY = "ollama"  # dummy, ollama doesn't need a key
    LLM_BASE_URL = "http://localhost:11434/v1"
    EXTRACTION_MODEL = "llama3.1"
    GENERATION_MODEL = "llama3.1"

elif LLM_PROVIDER == "openai":
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_BASE_URL = "https://api.openai.com/v1"
    EXTRACTION_MODEL = "gpt-4o-mini"
    GENERATION_MODEL = "gpt-4o"

elif LLM_PROVIDER == "bedrock":
    LLM_API_KEY = "bedrock"          # not used directly, boto3 handles auth
    LLM_BASE_URL = ""                # not used
    _bedrock_model = os.getenv("ANTHROPIC_MODEL", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    EXTRACTION_MODEL = _bedrock_model
    GENERATION_MODEL = _bedrock_model

else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use gemini, groq, ollama, openai, or bedrock.")


def get_llm_client():
    """Return the appropriate LLM client for the configured provider."""
    if LLM_PROVIDER == "bedrock":
        from src.bedrock_client import BedrockClient
        return BedrockClient(model=EXTRACTION_MODEL)
    else:
        from openai import OpenAI
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ── Embedding (always local, free) ───────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers, runs locally

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # tokens
CHUNK_OVERLAP = 64        # tokens

# ── Graph ─────────────────────────────────────────────────────────────
GRAPH_PATH = DATA_PROCESSED / "knowledge_graph.gpickle"
TRIPLES_CACHE = DATA_PROCESSED / "triples.json"
MAX_EXTRACTION_RETRIES = 5

# ── Retrieval ─────────────────────────────────────────────────────────
K_HOPS = 2                # traversal depth
TOP_N_PATHS = 30          # max paths to keep after pruning
FUZZY_MATCH_THRESHOLD = 0.75  # entity linking similarity cutoff

# ── Evaluation ────────────────────────────────────────────────────────
HOTPOTQA_SAMPLES = 500
MUSIQUE_SAMPLES = 300
VANILLA_TOP_K = 5         # chunks to retrieve in baseline RAG
