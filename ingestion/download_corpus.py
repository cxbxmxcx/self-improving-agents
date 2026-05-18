"""Download the HelixAgent reference corpus.

Pulls 33 arXiv papers plus the AlphaEvolve DeepMind tech report PDF and saves
them to ingestion/pdfs/ with stable filenames. Non-PDF blog posts (Anthropic
"Building effective agents", Cognition "Don't Build Multi-Agents") get rendered
to PDF separately or fetched as HTML; we save them as .html alongside the PDFs
and let the ingestion pipeline pick up whichever extracts cleanly.

Idempotent: skips files already present. Polite 3-second delay between fetches
to keep arxiv.org happy. Reports successes and failures at the end.

Usage:
    python ingestion/download_corpus.py
    python ingestion/download_corpus.py --skip-failures   # don't abort on a single 404
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PDFS_DIR = REPO_ROOT / "ingestion" / "pdfs"

USER_AGENT = "HelixAgent-Corpus-Builder/0.1 (book companion repo; contact: cxbxmxcx@gmail.com)"
POLITE_DELAY_SEC = 3.0


@dataclass
class Source:
    filename: str          # final name under ingestion/pdfs/
    url: str               # canonical download URL
    kind: str = "arxiv"    # "arxiv" | "tech_report" | "blog"
    notes: str = ""


# ---------------------------------------------------------------------------
# Confirmed arXiv papers (33 entries).
# Filename convention: <year>_<firstauthor>_<shorttitle>.pdf
# ---------------------------------------------------------------------------
ARXIV_PAPERS: list[Source] = [
    # FOUNDATIONS
    Source("2003_schmidhuber_godel_machines.pdf",       "https://arxiv.org/pdf/cs/0309048"),
    Source("2022_yao_react.pdf",                        "https://arxiv.org/pdf/2210.03629"),
    Source("2022_wei_chain_of_thought.pdf",             "https://arxiv.org/pdf/2201.11903"),
    Source("2023_schick_toolformer.pdf",                "https://arxiv.org/pdf/2302.04761"),

    # PROMPT-LAYER SEARCH
    Source("2022_zhou_ape.pdf",                         "https://arxiv.org/pdf/2211.01910"),
    Source("2023_pryzant_protegi_apo.pdf",              "https://arxiv.org/pdf/2305.03495"),
    Source("2025_xiang_spo.pdf",                        "https://arxiv.org/pdf/2502.06855"),
    Source("2025_agrawal_gepa.pdf",                     "https://arxiv.org/pdf/2507.19457"),
    Source("2024_opsahl_ong_miprov2.pdf",               "https://arxiv.org/pdf/2406.11695"),
    Source("2024_zhang_aflow.pdf",                      "https://arxiv.org/pdf/2410.10762"),

    # MEMORY
    Source("2023_park_generative_agents.pdf",           "https://arxiv.org/pdf/2304.03442"),
    Source("2023_shinn_reflexion.pdf",                  "https://arxiv.org/pdf/2303.11366"),
    Source("2024_zhao_expel.pdf",                       "https://arxiv.org/pdf/2308.10144"),
    Source("2023_packer_memgpt.pdf",                    "https://arxiv.org/pdf/2310.08560"),
    Source("2026_memrl.pdf",                            "https://arxiv.org/pdf/2601.03192",
        notes="MemTensor team Jan 2026 — confidence MEDIUM, verify on abstract page"),
    Source("2025_memory_r1.pdf",                        "https://arxiv.org/pdf/2508.19828",
        notes="alternate memory+RL angle, kept alongside MemRL per user request"),
    Source("2025_chhikara_mem0.pdf",                    "https://arxiv.org/pdf/2504.19413"),

    # METACOGNITION / REFLECTION CRITIQUE
    Source("2024_huang_llms_cannot_self_correct.pdf",   "https://arxiv.org/pdf/2310.01798"),
    Source("2025_self_correction_bench.pdf",            "https://arxiv.org/pdf/2507.02778"),
    Source("2023_lightman_lets_verify_step_by_step.pdf","https://arxiv.org/pdf/2305.20050"),
    Source("2023_wang_math_shepherd.pdf",               "https://arxiv.org/pdf/2312.08935"),
    Source("2025_agentprm.pdf",                         "https://arxiv.org/pdf/2511.08325",
        notes="second AgentPRM (Nov 2025, Web Conf 2026) per user choice"),

    # SKILLS / TOOLS
    Source("2023_wang_voyager.pdf",                     "https://arxiv.org/pdf/2305.16291"),
    Source("2023_qin_toolllm.pdf",                      "https://arxiv.org/pdf/2307.16789"),

    # ARCHITECTURAL / FRONTIER
    Source("2025_zhang_darwin_godel_machine.pdf",       "https://arxiv.org/pdf/2505.22954"),
    Source("2025_lange_shinkaevolve.pdf",               "https://arxiv.org/pdf/2509.19349"),
    Source("2026_zhang_hyperagents.pdf",                "https://arxiv.org/pdf/2603.19461"),

    # EVAL / JUDGING
    Source("2023_liu_g_eval.pdf",                       "https://arxiv.org/pdf/2303.16634"),
    Source("2023_zheng_mt_bench.pdf",                   "https://arxiv.org/pdf/2306.05685"),
    Source("2023_zhu_judgelm.pdf",                      "https://arxiv.org/pdf/2310.17631"),

    # MULTI-AGENT
    Source("2023_wu_autogen.pdf",                       "https://arxiv.org/pdf/2308.08155"),
    Source("2023_hong_metagpt.pdf",                     "https://arxiv.org/pdf/2308.00352"),

    # FAILURE MODES / DRIFT / REWARD HACKING
    Source("2024_pan_feedback_loops_icrh.pdf",          "https://arxiv.org/pdf/2402.06627"),
    Source("2022_skalse_reward_hacking.pdf",            "https://arxiv.org/pdf/2209.13085"),
    Source("2025_cemri_mast.pdf",                       "https://arxiv.org/pdf/2503.13657"),
]


# ---------------------------------------------------------------------------
# Non-arXiv sources are intentionally excluded from the shipped corpus.
# AlphaEvolve (DeepMind tech report), Anthropic "Building effective agents",
# and Cognition "Don't Build Multi-Agents" are referenced narratively in the
# book but not retrievable. Eval questions are scoped to the arXiv corpus.
# ---------------------------------------------------------------------------
NON_ARXIV_SOURCES: list[Source] = []


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------
def fetch(source: Source, out_dir: Path, skip_failures: bool) -> tuple[bool, str]:
    target = out_dir / source.filename
    if target.exists() and target.stat().st_size > 1024:
        return True, f"skip (already present, {target.stat().st_size} bytes)"

    req = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 1024:
            msg = f"suspiciously small response ({len(data)} bytes)"
            if not skip_failures:
                raise RuntimeError(msg)
            return False, msg
        target.write_bytes(data)
        return True, f"downloaded {len(data):,} bytes"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run(skip_failures: bool) -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    all_sources = ARXIV_PAPERS + NON_ARXIV_SOURCES
    print(f"Downloading {len(all_sources)} sources into {PDFS_DIR}")
    print(f"Polite delay: {POLITE_DELAY_SEC}s between requests\n")

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for i, src in enumerate(all_sources, 1):
        print(f"[{i:>2}/{len(all_sources)}] {src.filename}")
        print(f"    {src.url}")
        ok, msg = fetch(src, PDFS_DIR, skip_failures)
        print(f"    -> {msg}")
        if src.notes:
            print(f"    note: {src.notes}")
        if ok:
            successes.append(src.filename)
        else:
            failures.append((src.filename, msg))
            if not skip_failures:
                print(f"\nAborting on failure. Re-run with --skip-failures to continue past errors.")
                break

        if i < len(all_sources):
            time.sleep(POLITE_DELAY_SEC)

    print("\n" + "=" * 60)
    print(f"Done. {len(successes)} ok, {len(failures)} failed.")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        print("\nFix failures, then re-run. Successful files are skipped on rerun.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the HelixAgent reference corpus.")
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Continue past individual download failures instead of aborting.",
    )
    args = parser.parse_args()
    run(args.skip_failures)


if __name__ == "__main__":
    main()
