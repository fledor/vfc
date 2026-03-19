# vfc — Video Fact-Checking Pipeline

Universal fact-checking pipeline for YouTube videos in any language.
Scrapes audio from any YouTube video or playlist, transcribes with Whisper
(auto-detecting language), detects checkworthy claims using a hybrid
classifier + LLM approach, deduplicates them, and cross-references against
a fact-check knowledge base.

## Pipeline

```
1. Scrape          yt-dlp downloads audio (skips cached files)
2. Transcribe      faster-whisper (large-v3) with automatic language detection
3. Topic extract   LLM summarises topic from video title, channel, and transcript opening
4. Detect claims   Language-routed ClaimBuster classifier filters check-worthy sentences
                   GPT-4o-mini extracts & decontextualises each claim (topic-aware prompt)
5. Deduplicate     Sentence embeddings + LLM subsumption check
6. Fact-check      Query reformulation → KB API retrieval → CoT relevance judgment
```

## Design choices and references

### Language detection and model routing

Whisper auto-detects the transcript language, which then routes downstream
model selection. The ClaimBuster classifier is chosen per language:

| Language | Model | Source |
|----------|-------|--------|
| English  | `whispAI/ClaimBuster-DeBERTaV2` | English-only, highest accuracy |
| German   | `Nithiwat/mdeberta-v3-base_claimbuster` | Multilingual mDeBERTa |
| Other    | `Nithiwat/mdeberta-v3-base_claimbuster` | Multilingual fallback |

Set `CW_MODEL_NAME` to override language routing with a single model.

### Topic extraction

Before claim detection, the pipeline extracts a one-line topic summary from
the YouTube video title, channel name, and first ~2 minutes of transcript.
This topic is injected into the claim extraction prompt to improve
decontextualisation and reduce hallucination — the model can resolve
ambiguous references (e.g. "he" → specific politician) when it knows the
video discusses a specific election or event.

### Checkworthiness detection (hybrid)

The pipeline uses a two-stage approach combining a fine-tuned classifier with
an LLM, following findings from CheckThat! 2025 that "fine-tuned BERT-based
models outperform LLMs on in-domain claim detection" while "LLMs generalize
better to out-of-domain data" [(UNH at CheckThat! 2025)](https://arxiv.org/pdf/2509.06883).

**Stage 1 — Classifier pre-filter:**
The language-appropriate ClaimBuster model (see above) classifies each
sentence using a 3-class taxonomy (NFS/UFS/CFS) that distinguishes
*check-worthy factual* statements from *unimportant factual* and *non-factual*
ones — critical for filtering conversational speech where most statements
contain some factual element
[(Arslan et al., 2020)](https://arxiv.org/abs/2004.14425).

**Stage 2 — LLM extraction & decontextualisation:**
Each CFS sentence is sent to GPT-4o-mini with surrounding context and the
video topic to extract a concise, self-contained claim. The prompt enforces
*decontextualisation* — resolving pronouns, naming actors, and specifying
events — following best practices from the Claimify framework
[(Chen et al., 2025)](https://aclanthology.org/2025.acl-long.348.pdf) and
CLEF 2025 claim normalisation task
[(Nakov et al., 2025)](https://arxiv.org/abs/2503.14828).

### Deduplication (two-stage)

Semantic deduplication uses `paraphrase-multilingual-mpnet-base-v2` embeddings
with a two-tier strategy:

- **Hard threshold** (>= 0.85 cosine similarity): auto-merge without LLM.
- **Soft threshold** (0.65-0.85): LLM-based subsumption check determines
  whether one claim contains the other.

This addresses a limitation of pure embedding similarity where paraphrased
or subsuming claims can fall below fixed thresholds
[(NVIDIA NeMo SemDedup)](https://docs.nvidia.com/nemo-framework/user-guide/25.07/datacuration/semdedup.html).

### Fact-check retrieval

**Query reformulation:** Before querying the KB API, each claim is rewritten
into a clean search query (fixing residual transcription errors, stripping
attributions). This follows findings from hybrid fact-checking systems that
query reformulation significantly improves retrieval recall
[(Jiang et al., 2025)](https://arxiv.org/abs/2511.03217).

**Relevance judgment (chain-of-thought):** Retrieved fact-checks are judged for
relevance using a structured multi-criteria prompt that requires reasoning
through SUBJECT, EVENT, and ASSERTION alignment before a verdict. This
follows the multi-criteria decomposition approach from
[(Stureborg et al., 2025)](https://arxiv.org/html/2507.09488v1).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Required configuration

Create a `.env` file (or export the variables):

```bash
OPENAI_API_KEY=sk-...          # for claim extraction, dedup, query rewrite, relevance
KB_API_KEY=...       # fact-check knowledge base
```

### Optional environment variables

```bash
WHISPER_LANGUAGE=auto                                  # auto-detect (default) or force e.g. "de"
CW_MODEL_NAME=                                         # override per-language model routing
CW_CONFIDENCE_THRESHOLD=0.5                            # CFS classification threshold
DEDUP_SIMILARITY_THRESHOLD=0.85                        # hard cosine threshold for dedup
KB_SEARCH_LIMIT=10                                     # max results per KB query
KB_INCLUDE_EVIDENCE=false                              # set true for evidence (slow)
WHISPER_MODEL=large-v3                                 # whisper model size
OPENAI_MODEL=gpt-4o-mini                               # LLM model
```

## Usage

```bash
# Full pipeline (default playlist, 1 video)
vfc

# Single video
vfc "https://www.youtube.com/watch?v=VIDEO_ID"

# Whole playlist
vfc "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-videos 10

# Skip fact-check step (detection + dedup only)
vfc --skip-factcheck

# Debug logging
vfc --log-level DEBUG
```

## Project structure

```
config.py          Configuration and environment variables
scraper.py         YouTube audio download (yt-dlp), caches audio files
transcriber.py     Speech-to-text (faster-whisper), auto language detection
topic.py           LLM-based topic extraction from video metadata + transcript
claim_detector.py  Hybrid checkworthiness detection (classifier + LLM)
deduplicator.py    Two-stage deduplication (embeddings + LLM subsumption)
fact_checker.py    Query reformulation + KB search + CoT relevance judgment
pipeline.py        Orchestration and CLI entry point
data/
  audio/           Downloaded WAV files + metadata JSON
  transcripts/     Transcript JSON + TXT files
  claims/          Detected claims JSON
  factchecks/      Fact-check results JSON
```

## References

- Arslan, F., et al. (2020). *A Benchmark Dataset of Check-Worthy Factual Claims.* [arXiv:2004.14425](https://arxiv.org/abs/2004.14425)
- Chen, Y., et al. (2025). *Claimify: LLM-Based Claim Extraction.* [ACL 2025](https://aclanthology.org/2025.acl-long.348.pdf)
- Jiang, A., et al. (2025). *Hybrid Fact-Checking with KG, LLM, and Search Agents.* [arXiv:2511.03217](https://arxiv.org/abs/2511.03217)
- Nakov, P., et al. (2025). *The CLEF-2025 CheckThat! Lab.* [arXiv:2503.14828](https://arxiv.org/abs/2503.14828)
- Stureborg, R., et al. (2025). *Criteria-Based LLM Relevance Judgments.* [arXiv:2507.09488](https://arxiv.org/abs/2507.09488)
- UNH at CheckThat! 2025. *Fine-tuning vs Prompting in Claim Extraction.* [arXiv:2509.06883](https://arxiv.org/abs/2509.06883)
