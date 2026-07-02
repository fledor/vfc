"""Generate concise result files from fact-checked claims."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import RESULTS_DIR

logger = logging.getLogger(__name__)


def _has_evidence(claim: dict) -> bool:
    """True if any topically-relevant web evidence was attached (incl. inconclusive)."""
    we = claim.get("web_evidence")
    return bool(we and we.get("assessment") in ("SUPPORTS", "REFUTES", "INSUFFICIENT"))


def _is_conclusive_evidence(claim: dict) -> bool:
    """True if web evidence conclusively supports or refutes the claim."""
    we = claim.get("web_evidence")
    return bool(we and we.get("assessment") in ("SUPPORTS", "REFUTES"))


def _is_verified(claim: dict) -> bool:
    """A claim is *verified* if a fact-check matched or web evidence was conclusive.

    A claim that is shown but not verified (only INSUFFICIENT evidence) is
    *inconclusive* — surfaced for the reader, but not counted as verified.
    """
    return bool(claim.get("relevant_factchecks")) or _is_conclusive_evidence(claim)


def _fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def generate_results(
    claims: list[dict],
    video_id: str,
    video_title: str = "",
    channel: str = "",
    language: str = "",
    topic: str = "",
    output_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    """Write a concise JSON and markdown report for claims with relevant fact-checks.

    Returns (json_path, md_path).
    """
    matched = [
        c for c in claims
        if c.get("relevant_factchecks") or _has_evidence(c)
    ]

    results_json = _build_results_json(
        matched, claims, video_id, video_title, channel, language, topic,
    )
    json_path = output_dir / f"{video_id}.json"
    json_path.write_text(json.dumps(results_json, ensure_ascii=False, indent=2))

    md_text = _build_markdown(results_json)
    md_path = output_dir / f"{video_id}.md"
    md_path.write_text(md_text)

    verified = sum(1 for c in matched if _is_verified(c))
    inconclusive = len(matched) - verified
    logger.info(
        "Results: %d verified, %d inconclusive (%d claims examined) → %s",
        verified,
        inconclusive,
        len(claims),
        output_dir,
    )
    return json_path, md_path


def _build_results_json(
    matched: list[dict],
    all_claims: list[dict],
    video_id: str,
    video_title: str,
    channel: str,
    language: str,
    topic: str,
) -> dict:
    summary = {
        "video_id": video_id,
        "video_title": video_title,
        "channel": channel,
        "language": language,
        "topic": topic,
        "total_claims": len(all_claims),
        "verified_claims": sum(1 for c in matched if _is_verified(c)),
        "inconclusive_claims": sum(1 for c in matched if not _is_verified(c)),
        "claims": [],
    }

    for claim in matched:
        entry: dict = {
            "claim_text": claim["claim_text"],
            "claim_original": claim.get("claim_original", ""),
            "timestamp": f"{_fmt_timestamp(claim.get('start', 0))}–{_fmt_timestamp(claim.get('end', 0))}",
            "cfs_score": round(claim.get("cfs_score", 0), 3),
            "status": "verified" if _is_verified(claim) else "inconclusive",
            "factchecks": [],
        }
        for fc in claim.get("relevant_factchecks", []):
            entry["factchecks"].append({
                "source": fc.get("source", ""),
                "title": fc.get("title"),
                "claim": fc.get("claim", ""),
                "verdict": fc.get("verdict", ""),
                "url": fc.get("url", ""),
            })

        we = claim.get("web_evidence")
        if _has_evidence(claim) and we:
            entry["web_evidence"] = {
                "assessment": we["assessment"],
                "summary": we.get("summary", ""),
                "sources": [
                    {"title": s.get("title", ""), "url": s.get("url", "")}
                    for s in we.get("sources", [])
                ],
            }

        summary["claims"].append(entry)

    return summary


def _build_markdown(results: dict) -> str:
    lines: list[str] = []

    title = results.get("video_title") or results["video_id"]
    lines.append(f"# Fact-Check Results: {title}\n")

    lines.append(f"**Video:** `{results['video_id']}`")
    if results.get("channel"):
        lines.append(f"**Channel:** {results['channel']}")
    if results.get("language"):
        lines.append(f"**Language:** {results['language']}")
    if results.get("topic"):
        lines.append(f"**Topic:** {results['topic']}")
    lines.append("")
    lines.append(
        f"**{results['verified_claims']} verified**, "
        f"{results['inconclusive_claims']} with inconclusive evidence "
        f"({results['total_claims']} claims examined).\n"
    )
    lines.append("---\n")

    for i, claim in enumerate(results["claims"], 1):
        status_label = "Verified" if claim.get("status") == "verified" else "Inconclusive"
        lines.append(
            f"### Claim {i} — {status_label} "
            f"({claim['timestamp']}, CFS {claim['cfs_score']:.2f})\n"
        )
        lines.append(f"> {claim['claim_text']}\n")

        if claim.get("claim_original") and claim["claim_original"] != claim["claim_text"]:
            lines.append(f"*Original transcript:* {claim['claim_original']}\n")

        if claim.get("factchecks"):
            lines.append("| Source | Verdict | Fact-Check Claim | Link |")
            lines.append("|--------|---------|-----------------|------|")
            for fc in claim["factchecks"]:
                source = fc.get("source", "")
                verdict = fc.get("verdict") or "—"
                fc_claim = (fc.get("claim") or fc.get("title") or "").replace("|", "\\|")
                if len(fc_claim) > 120:
                    fc_claim = fc_claim[:117] + "..."
                url = fc.get("url", "")
                link = f"[link]({url})" if url else "—"
                lines.append(f"| {source} | {verdict} | {fc_claim} | {link} |")
            lines.append("")

        we = claim.get("web_evidence")
        if we:
            lines.append(
                f"**Web Evidence** [{we['assessment']}]: {we.get('summary', '')}\n"
            )
            sources = we.get("sources", [])
            if sources:
                for s in sources:
                    title = s.get("title", s.get("url", ""))
                    url = s.get("url", "")
                    if url:
                        lines.append(f"- [{title}]({url})")
                    else:
                        lines.append(f"- {title}")
                lines.append("")

        lines.append("")

    return "\n".join(lines)
