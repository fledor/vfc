from pathlib import Path
import logging
import os

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CLAIMS_DIR = DATA_DIR / "claims"
FACTCHECKS_DIR = DATA_DIR / "factchecks"
RESULTS_DIR = DATA_DIR / "results"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Optional stronger model for claim extraction (better discourse reasoning)
OPENAI_EXTRACT_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", OPENAI_MODEL)
# Stronger model for FC relevance judgment — nuanced multi-factor reasoning benefits from better model
OPENAI_RELEVANCE_MODEL = os.getenv("OPENAI_RELEVANCE_MODEL", "gpt-4.1-mini")

# Per-request timeout (seconds) and retry count for OpenAI calls. The SDK
# retries transient failures (429, 5xx, connection errors) with exponential
# backoff up to max_retries.
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))

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

# Optional private fact-check knowledge base API (bring-your-own endpoint).
# Both URL and key must be set for this source to be queried; otherwise the
# pipeline skips it and relies on Google Fact Check + web evidence.
PRIVATE_KB_API_URL = os.getenv("PRIVATE_KB_API_URL", "")
PRIVATE_KB_API_KEY = os.getenv("PRIVATE_KB_API_KEY", "")
PRIVATE_KB_SEARCH_LIMIT = int(os.getenv("PRIVATE_KB_SEARCH_LIMIT", "10"))
PRIVATE_KB_INCLUDE_EVIDENCE = (
    os.getenv("PRIVATE_KB_INCLUDE_EVIDENCE", "false").lower() == "true"
)

# Google Fact Check Tools API
GOOGLE_FC_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
GOOGLE_FC_MAX_AGE_DAYS = int(os.getenv("GOOGLE_FC_MAX_AGE_DAYS", "0"))
GOOGLE_FC_PAGE_SIZE = int(os.getenv("GOOGLE_FC_PAGE_SIZE", "10"))

# Web evidence search (authoritative sources via Tavily)
ENABLE_WEB_EVIDENCE = os.getenv("ENABLE_WEB_EVIDENCE", "false").lower() == "true"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EVIDENCE_MAX_RESULTS = int(os.getenv("EVIDENCE_MAX_RESULTS", "5"))

EVIDENCE_DOMAINS: list[str] = [
    # German government / official
    "bundesregierung.de",
    "destatis.de",
    "bundestag.de",
    "bundesnetzagentur.de",
    "bmwk.de",
    "bafa.de",
    "bmvg.de",
    "auswaertiges-amt.de",
    # EU institutions
    "europa.eu",
    "ec.europa.eu",
    "consilium.europa.eu",
    "eurostat.ec.europa.eu",
    # International organisations
    "nato.int",
    "iea.org",
    "imf.org",
    "worldbank.org",
    "un.org",
    "who.int",
    # Wire services / high-trust news
    "reuters.com",
    "apnews.com",
    "dpa.com",
    # German public broadcasters
    "tagesschau.de",
    "zdf.de",
    "deutschlandfunk.de",
    # Research / data
    "statista.com",
    "sipri.org",
]

_extra = os.getenv("EVIDENCE_DOMAINS_EXTRA", "")
if _extra:
    EVIDENCE_DOMAINS = EVIDENCE_DOMAINS + [d.strip() for d in _extra.split(",") if d.strip()]

for d in (AUDIO_DIR, TRANSCRIPTS_DIR, CLAIMS_DIR, FACTCHECKS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_openai_client():
    """Return an OpenAI client with a bounded timeout and automatic retries.

    Centralizes client construction so every pipeline stage shares the same
    timeout and exponential-backoff retry behaviour for transient failures.
    """
    from openai import OpenAI

    return OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT,
        max_retries=OPENAI_MAX_RETRIES,
    )


def validate_config() -> None:
    """Fail fast on missing required config; log optional-integration status.

    ``OPENAI_API_KEY`` is required (claim extraction + relevance judgment).
    The other integrations are optional and degrade gracefully — their status
    is logged so the user knows which sources are active for this run.
    """
    if not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set. It is required for claim extraction and "
            "relevance judgment. Set it in your environment or a .env file."
        )

    logger = logging.getLogger(__name__)
    integrations = [
        ("Private KB", bool(PRIVATE_KB_API_URL and PRIVATE_KB_API_KEY),
         "PRIVATE_KB_API_URL + PRIVATE_KB_API_KEY"),
        ("Google Fact Check", bool(GOOGLE_FC_API_KEY), "GOOGLE_FACTCHECK_API_KEY"),
        ("Web evidence (Tavily)", bool(TAVILY_API_KEY), "TAVILY_API_KEY"),
    ]
    for name, enabled, env_names in integrations:
        label = f"{name}:"
        if enabled:
            logger.info("%-24s enabled", label)
        else:
            logger.info("%-24s disabled (%s not set)", label, env_names)
