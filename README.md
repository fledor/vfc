# vfc -- Video Fact-Checking Pipeline

Automated fact-checking pipeline for German talk show videos (Markus Lanz / ZDF). Scrapes audio from YouTube, transcribes with Whisper, detects checkworthy claims using a fine-tuned transformer, deduplicates them, and cross-references against a fact-check knowledge base.

## Pipeline

```
1. Scrape        yt-dlp downloads audio from a YouTube playlist
2. Transcribe    faster-whisper (large-v3) produces German transcripts
3. Detect        XLM-R classifier flags checkworthy factual claims
4. Deduplicate   Sentence embeddings remove near-duplicate claims
5. Fact-check    KB API retrieval + LLM relevance judgment
```

### Claim detection model

The default checkworthiness classifier is [`Sami92/XLM-R-Large-ClaimDetection`](https://huggingface.co/Sami92/XLM-R-Large-ClaimDetection) -- an XLM-R Large model fine-tuned on German data (GermEval 2021 + Telegram posts). It classifies sentences as *factual* (verifiable claim) or *non-factual*.

Alternative models (set `CW_MODEL_NAME` or `CW_MODEL_PATH`):

| Model | Source | Languages | Notes |
|-------|--------|-----------|-------|
| `Sami92/XLM-R-Large-ClaimDetection` | HuggingFace | German (multilingual backbone) | Default. Trained on German social media. |
| MultiCW mDeBERTa-v3 | [kinit-sk/MultiCW](https://github.com/kinit-sk/MultiCW) | 16 languages incl. German | Best benchmark results (92%+ accuracy). Download from repo. |
| `sergiopinto/deberta-v3-large-claim-checkworthiness-seed0` | HuggingFace | English | 80% F1. English only -- requires translation step. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Required configuration

Create a `.env` file (or export the variables):

```bash
OPENAI_API_KEY=sk-...                              # for claim normalisation + relevance judgment
KB_API_KEY=...                            # fact-check knowledge base
```

### Optional environment variables

```bash
CW_MODEL_NAME=Sami92/XLM-R-Large-ClaimDetection   # checkworthiness model
CW_MODEL_PATH=                                      # local path to fine-tuned checkpoint
CW_CONFIDENCE_THRESHOLD=0.5                          # classification threshold
DEDUP_SIMILARITY_THRESHOLD=0.85                      # cosine similarity for dedup
KB_SEARCH_LIMIT=10                                   # max results per KB query
KB_INCLUDE_EVIDENCE=false                             # set true for evidence (slow, cached)
WHISPER_MODEL=large-v3                                # whisper model size
OPENAI_MODEL=gpt-4o-mini                              # LLM for normalisation/relevance
```

## Usage

```bash
# Full pipeline (1 video)
vfc

# Process 5 videos
vfc --max-videos 5

# Custom playlist/video URL
vfc --playlist-url "https://www.youtube.com/watch?v=..."

# Skip fact-check step (detection + dedup only)
vfc --skip-factcheck

# Debug logging
vfc --log-level DEBUG
```

## Project structure

```
config.py          Configuration and environment variables
scraper.py         YouTube audio download (yt-dlp)
transcriber.py     Speech-to-text (faster-whisper)
claim_detector.py  Checkworthiness classification + LLM normalisation
deduplicator.py    Semantic deduplication (sentence-transformers)
fact_checker.py    KB API search + LLM relevance judgment
pipeline.py        Orchestration and CLI entry point
data/
  audio/           Downloaded WAV files
  transcripts/     Transcript JSON + TXT files
  claims/          Detected claims JSON
  factchecks/      Fact-check results JSON
```
