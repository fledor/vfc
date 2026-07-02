# Case study: fact-checking German political talk shows

A worked example of running `vfc` end-to-end over a batch of real videos. It
documents the pipeline stages, the aggregate results, and what we learned about
where automated fact-checking works and where it breaks down. All numbers below
come from a single run; web-evidence results are non-deterministic (see
[Limitations](#limitations)), so a re-run will shift individual claims.

## Pipeline overview

| Step                    | Component                    | Purpose                                                                                                                                                                                                |
| ----------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Scrape               | `yt-dlp`                     | Download audio from YouTube videos/playlists; caches to avoid re-downloads                                                                                                                             |
| 2. Transcribe           | `faster-whisper` (large-v3)  | Speech-to-text with automatic language detection; necessary because talk show speech is unstructured                                                                                                   |
| 3. Topic extraction     | GPT-4o-mini                  | Summarise the video topic from title + transcript in the transcript's language; injected into claim extraction to resolve ambiguous references ("he" → specific politician)                            |
| 4. Claim detection      | ClaimBuster + GPT-4o-mini    | Two-stage hybrid: classifier pre-filters checkworthy sentences (most transcript sentences are conversational filler), then LLM extracts self-contained claims via chain-of-thought decontextualisation |
| 5. Deduplication        | Sentence embeddings + LLM    | Removes near-duplicate and subsuming claims; needed because talk shows revisit the same point multiple times                                                                                           |
| 6. Fact-check retrieval | Private KB + Google FC API   | Query reformulation (GPT-4o-mini) then concurrent search across an optional private knowledge base and the Google Fact Check API                                                                        |
| 7. Relevance judgment   | GPT-4.1-mini                 | Structured CoT prompt (MEDIA / SUBJECT / ASSERTION / TEMPORAL) with consistency check and confirmation call; 3 concurrent threads                                                                      |
| 8. Web evidence         | Tavily + GPT-4o-mini         | Searches authoritative domains (government, EU, IEA, wire services) for primary-source evidence; fills coverage gaps where no fact-check article exists                                                |
| 9. Report               | JSON + Markdown              | Structured output with claims, verdicts, evidence summaries, and source links                                                                                                                          |

## Evaluation results

Dataset: 10 episodes of a German political talk-show format (public-broadcaster
political discussion, aired March 2026).

| Video       | Title                                              | Claims  | FC          | Conclusive WE | Any Verified  | Missed        |
| ----------- | -------------------------------------------------- | ------- | ----------- | ------------- | ------------- | ------------- |
| 2ddWqGhfcrE | Antisemitismus-Vorwürfe gegen Die Linke...         | 4       | 0 (0%)      | 0 (0%)        | 0 (0%)        | 4 (100%)      |
| 39tvI-oZTaw | Wehrpflicht und Kriegseinsätze...                  | 17      | 1 (6%)      | 12 (71%)      | 13 (76%)      | 4 (24%)       |
| U_zMkV-DCK0 | Sachsen-Anhalt: Regiert die AfD hier bald allein?  | 27      | 6 (22%)     | 9 (33%)       | 13 (48%)      | 14 (52%)      |
| VyB1nZ4KMbI | "Rehbraune Augen": Hat das Video der CDU...        | 16      | 2 (13%)     | 5 (31%)       | 7 (44%)       | 9 (56%)       |
| _qyEIFJPwNA | Wehrdienst: Würden junge Menschen...               | 15      | 0 (0%)      | 4 (27%)       | 4 (27%)       | 11 (73%)      |
| hghNZjGnqhI | Steigende Spritpreise: Müssen wir im Iran-Krieg... | 41      | 6 (15%)     | 25 (61%)      | 29 (71%)      | 12 (29%)      |
| l9D75ga3-A8 | AfD-Chef Chrupalla über Trump und Iran-Krieg...    | 28      | 0 (0%)      | 13 (46%)      | 13 (46%)      | 15 (54%)      |
| rcxMskKdsPk | AfD-Verbotsverfahren: CDU-Politiker warnt...       | 11      | 2 (18%)     | 5 (45%)       | 6 (55%)       | 5 (45%)       |
| yRcRU92AF-o | Krieg in Iran: Droht jetzt der Flächenbrand?       | 61      | 4 (7%)      | 23 (38%)      | 24 (39%)      | 37 (61%)      |
| zZVAFSqfzBY | CDU vs. Linke: Bringt uns die Wehrpflicht...       | 22      | 0 (0%)      | 6 (27%)       | 6 (27%)       | 16 (73%)      |
| **Total**   |                                                    | **242** | **21 (9%)** | **102 (42%)** | **115 (48%)** | **127 (52%)** |

- **FC** = claims matched to existing fact-check articles (private KB + Google Fact Check API); relevance judged by `gpt-4.1-mini` with structured CoT prompt (MEDIA / SUBJECT / ASSERTION / TEMPORAL checks), consistency verification, and confirmation call
- **Conclusive WE** = claims with SUPPORTS or REFUTES assessment from authoritative web sources (69 SUPPORTS / 33 REFUTES)
- **Missed** = claims with no fact-check match and either INSUFFICIENT web evidence (topically related but not conclusive) or no web evidence match at all

## Key statistics

- **Pipeline funnel:** around 6,300 transcript segments → 275 checkworthy claims (classifier) → **242 after deduplication** (12% reduction). Only around 4% of transcript content is identified as checkworthy
- Claim density varies widely: 0.3–1.9 claims/min depending on video content
- **9% fact-check coverage** from databases alone → **48% with web evidence** added
- **~2:1 ratio** of SUPPORTS (69) to REFUTES (33) in web evidence — most claims are broadly accurate, but around 1/3 of verifiably checked claims contain factual errors
- CFS scores (checkworthiness confidence from the ClaimBuster classifier) do not predict verifiability (mean 0.89 for verified, 0.87 for missed)
- Longer claims are slightly harder to verify: INSUFFICIENT claims average 140 characters vs ~130 for conclusive ones

## Insights and challenges

- **Topic sets the ceiling, speaker behaviour determines how much of it you reach:** Topics with heavy public reporting (Iran/geopolitics 51%, domestic politics 48%) verify better than those that invite personal narratives (Wehrpflicht 43%, antisemitism 0%). But within-topic variance is larger: Wehrpflicht ranges from 27% to 76% depending on whether the speaker cites publicly-documented statistics (troop counts, polling data, parliamentary records) or shares personal military experiences and operational details that lack public records. The most consistent group is AfD/domestic politics (44–55%), where all speakers make similar types of claims grounded in election data and party positions
- **Fact-check databases have limited coverage for current events:** Only 9% of claims matched existing fact-checks. These databases excel for viral misinformation but miss most claims from live political discussion. Of the 21 matches, manual review found ~60% clearly relevant (e.g. Germany's CO2 share, gas storage capacity, Nord Stream), ~30% tangential (vague claims matched to loosely related FCs), and ~10% incorrect/wrongly attributed
- **Web evidence is the primary verification driver:** Authoritative web sources (government sites, IEA, Eurostat, wire services) provide conclusive evidence for 42% of claims, more than quadrupling the coverage of fact-check databases alone
- **REFUTES findings are high-value:** 33 claims refuted across diverse categories — wrong LNG import percentages, incorrect troop/warhead counts, exaggerated election results, inaccurate oil price calculations, and mischaracterised parliamentary compositions
- **127 claims remained unverified or with insufficient evidence:** These are typically insider knowledge ("intelligence reports say..."), real-time geopolitical developments, or highly specific numbers that authoritative sources discuss tangentially but never directly confirm
- **Top web evidence sources:** AP News (119), Reuters (118), Statista (112), Bundestag (101), tagesschau.de (96) — 1,070 total Tavily sources from 80 unique domains across all claims
- **Runtime and cost:** Roughly comparable to video length (15-20 min for a typical 20-40 min episode); scales with claim count, not duration. Per-video: ~2k OpenAI requests (gpt-4o-mini for extraction/evidence, gpt-4.1-mini for relevance) + ~50 Tavily credits. OpenAI requests could be reduced with fewer reformulations/topic detection, but these measures improved claim quality and detection significantly

## Limitations

- LLM assessments (SUPPORTS/REFUTES) are **not manually validated** — precision and recall of these judgments are unknown
- All videos are from a single show format (German political talk show); coverage rates may differ for other languages, topics, or video types
- Web evidence depends on Tavily search quality and the curated domain allowlist; claims about topics not covered by these domains will be missed
- Claim extraction occasionally produces English-language text for German transcripts (3 of 242 claims) despite language-matched topic generation
- Web evidence results are **non-deterministic** — re-running the pipeline can shift individual claims between SUPPORTS/REFUTES and INSUFFICIENT due to Tavily search variability and LLM assessment variance
