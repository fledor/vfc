from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CLAIMS_DIR = DATA_DIR / "claims"
FACTCHECKS_DIR = DATA_DIR / "factchecks"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Per-language checkworthiness models (all 3-class ClaimBuster taxonomy).
# Set CW_MODEL_NAME to override language routing with a single model.
CW_MODELS: dict[str, str] = {
    "en": "whispAI/ClaimBuster-DeBERTaV2",
    "de": "Nithiwat/mdeberta-v3-base_claimbuster",
}
CW_MODEL_FALLBACK = "Nithiwat/mdeberta-v3-base_claimbuster"
CW_MODEL_NAME = os.getenv("CW_MODEL_NAME", "")
CW_MODEL_PATH = os.getenv("CW_MODEL_PATH", "")
CW_CONFIDENCE_THRESHOLD = float(os.getenv("CW_CONFIDENCE_THRESHOLD", "0.5"))

# Deduplication
DEDUP_MODEL_NAME = os.getenv(
    "DEDUP_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))

# Fact-check knowledge base API
KB_API_URL = os.getenv(
    "KB_API_URL",
    "https://your-kb.example.com/api/search",
)
KB_API_KEY = os.getenv("KB_API_KEY", "")
KB_SEARCH_LIMIT = int(os.getenv("KB_SEARCH_LIMIT", "10"))
KB_INCLUDE_EVIDENCE = os.getenv("KB_INCLUDE_EVIDENCE", "false").lower() == "true"

for d in (AUDIO_DIR, TRANSCRIPTS_DIR, CLAIMS_DIR, FACTCHECKS_DIR):
    d.mkdir(parents=True, exist_ok=True)
