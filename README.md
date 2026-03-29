# vfc — Video Fact-Checking Pipeline

Automated fact-checking pipeline for YouTube videos. Scrapes audio, transcribes
with Whisper (auto-detecting language), detects checkworthy claims using a hybrid
classifier + LLM approach, deduplicates, cross-references against fact-check
databases, and searches authoritative web sources for supporting or refuting
evidence.

## Pipeline

```
1. Scrape          yt-dlp downloads audio (skips cached files)
2. Transcribe      faster-whisper (large-v3) with automatic language detection
3. Topic extract   LLM summarises topic from video title, channel, and transcript (in transcript language)
4. Detect claims   ClaimBuster classifier → GPT-4o-mini extraction with CoT decontextualisation
5. Deduplicate     Sentence embeddings + LLM subsumption check
6. Fact-check      Query reformulation → KB + Google Fact Check API → concurrent CoT relevance judgment
7. Web evidence    Tavily search on authoritative domains → LLM evidence assessment (SUPPORTS/REFUTES/INSUFFICIENT)
8. Report          JSON + markdown with matched claims, verdicts, and evidence summaries
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Required configuration

Create a `.env` file (or export the variables):

```bash
OPENAI_API_KEY=sk-...             # claim extraction, dedup, query rewrite, relevance judgment
KB_API_KEY=...          # fact-check knowledge base
GOOGLE_FACTCHECK_API_KEY=...     # Google Fact Check Tools API
TAVILY_API_KEY=tvly-...          # web evidence search (required for --web-evidence)
```

### Optional environment variables

```bash
WHISPER_LANGUAGE=auto                  # auto-detect (default) or force e.g. "de"
CW_MODEL_NAME=                        # override per-language model routing
CW_CONFIDENCE_THRESHOLD=0.5           # CFS classification threshold
DEDUP_SIMILARITY_THRESHOLD=0.85       # hard cosine threshold for dedup
KB_SEARCH_LIMIT=10                    # max results per KB query
WHISPER_MODEL=large-v3                # whisper model size
OPENAI_MODEL=gpt-4o-mini             # LLM model for most steps
OPENAI_EXTRACT_MODEL=                 # dedicated model for claim extraction (defaults to OPENAI_MODEL)
OPENAI_RELEVANCE_MODEL=gpt-4.1-mini  # model for FC relevance judgment (defaults to gpt-4.1-mini)
ENABLE_WEB_EVIDENCE=false             # enable web evidence search by default
EVIDENCE_MAX_RESULTS=5                # max Tavily results per query
EVIDENCE_DOMAINS_EXTRA=               # comma-separated additional trusted domains
```

## Usage

```bash
# Single video with web evidence
vfc "https://www.youtube.com/watch?v=VIDEO_ID" --web-evidence

# Whole playlist
vfc "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-videos 10

# Skip fact-check step (detection + dedup only)
vfc --skip-factcheck

# Debug logging
vfc --log-level DEBUG
```

## Design choices and references

### Language detection and model routing

Whisper auto-detects the transcript language, which routes downstream model
selection for the ClaimBuster classifier:

| Language | Model | Source |
|----------|-------|--------|
| English  | `whispAI/ClaimBuster-DeBERTaV2` | English-only, highest accuracy |
| German   | `Nithiwat/mdeberta-v3-base_claimbuster` | Multilingual mDeBERTa |
| Other    | `Nithiwat/mdeberta-v3-base_claimbuster` | Multilingual fallback |

### Checkworthiness detection (hybrid)

Two-stage approach: fine-tuned classifiers outperform LLMs on in-domain claim
detection, but LLMs generalise better and handle decontextualisation.

**Stage 1 — Classifier:** ClaimBuster model
([Arslan et al., 2020](https://arxiv.org/abs/2004.14425)) classifies sentences
into NFS/UFS/CFS taxonomy, filtering conversational filler.

**Stage 2 — LLM extraction:** GPT-4o-mini extracts self-contained claims using
chain-of-thought decontextualisation — resolving pronouns, naming actors,
specifying events and conflicts, anchoring statistics to their real-world context
— inspired by the Claimify framework
([Chen et al., 2025](https://aclanthology.org/2025.acl-long.348.pdf)).

### Deduplication

Semantic deduplication with `paraphrase-multilingual-mpnet-base-v2`:
- **Hard threshold** (≥0.85 cosine): auto-merge
- **Soft threshold** (0.55–0.85): LLM subsumption check

### Fact-check retrieval

Query reformulation + concurrent search across the KB and Google Fact
Check API. Relevance judged via structured multi-criteria CoT prompt (MEDIA gate,
SUBJECT, ASSERTION, TEMPORAL alignment) with two safeguards: a consistency check
that rejects matches where the reasoning contradicts the verdict, and a
confirmation call that requires two independent LLM judgments to agree.
Concurrent thread pool (3 workers).

### Web evidence search

Tavily API searches authoritative domains (government sites, EU institutions,
IEA, WHO, Reuters, AP, etc.) for supporting or refuting evidence. LLM assesses
retrieved snippets as SUPPORTS / REFUTES / INSUFFICIENT / UNRELATED. Graceful
handling of API credit exhaustion.

## Project structure

```
config.py          Configuration and environment variables
scraper.py         YouTube audio download (yt-dlp), caches audio files
transcriber.py     Speech-to-text (faster-whisper), auto language detection
topic.py           LLM-based topic extraction from video metadata + transcript
claim_detector.py  Hybrid checkworthiness detection (classifier + LLM)
deduplicator.py    Two-stage deduplication (embeddings + LLM subsumption)
fact_checker.py    Query reformulation + KB/Google search + concurrent CoT relevance judgment
evidence.py        Tavily web evidence search + LLM assessment
reporter.py        Generate JSON + markdown result reports
pipeline.py        Orchestration and CLI entry point
data/
  audio/           Downloaded WAV files + metadata JSON
  transcripts/     Transcript JSON + TXT files
  claims/          Detected claims JSON
  factchecks/      Full fact-check results JSON
  results/         Reports (JSON + markdown) with matched claims and evidence
```

## References

- Arslan, F., et al. (2020). *A Benchmark Dataset of Check-Worthy Factual Claims.* [arXiv:2004.14425](https://arxiv.org/abs/2004.14425)
- Chen, Y., et al. (2025). *Claimify: LLM-Based Claim Extraction from Real-World Sources.* [ACL 2025](https://aclanthology.org/2025.acl-long.348.pdf)
