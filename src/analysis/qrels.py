import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import DATA_RAW

FIGURES = Path("reports/figures")
LABELS = {0: "not relevant", 1: "excluded", 2: "eligible"}


def load_qrels(path: Path) -> list[tuple[int, str, int]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        topic, _, doc, judgement = line.split()
        rows.append((int(topic), doc, int(judgement)))
    return rows


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for year in (2021, 2022):
        rows = load_qrels(DATA_RAW / "trec-ct" / f"qrels{year}.txt")

        # ---- global counts -------------------------------------------------
        by_judgement = Counter(j for _, _, j in rows)
        topics = sorted({t for t, _, _ in rows})
        docs = {d for _, d, _ in rows}

        print(f"=== TREC CT {year} ===")
        print(f"judgements:      {len(rows)}")
        print(f"topics:          {len(topics)}")
        print(f"distinct trials: {len(docs)}")
        print()
        for j in (0, 1, 2):
            n = by_judgement[j]
            print(f"  {j} {LABELS[j]:<14} {n:>7}  ({n / len(rows):6.1%})")

        # ---- per-topic breakdown -------------------------------------------
        per_topic: dict[int, Counter] = defaultdict(Counter)
        for topic, _, judgement in rows:
            per_topic[topic][judgement] += 1

        pool = [sum(c.values()) for c in per_topic.values()]
        eligible = [per_topic[t][2] for t in topics]
        excluded = [per_topic[t][1] for t in topics]

        print("\n--- per topic ---")
        print(
            f"pool depth     median {statistics.median(pool):.0f}   "
            f"min {min(pool)}   max {max(pool)}"
        )
        print(
            f"eligible (2)   median {statistics.median(eligible):.0f}   "
            f"min {min(eligible)}   max {max(eligible)}"
        )
        print(
            f"excluded (1)   median {statistics.median(excluded):.0f}   "
            f"min {min(excluded)}   max {max(excluded)}"
        )

        # Topics with no eligible trial contribute nothing to nDCG: they are
        # dead weight in the evaluation and must be reported.
        dead = [t for t in topics if per_topic[t][2] == 0]
        print(f"\ntopics with 0 eligible: {len(dead)}  {dead if dead else ''}")
        thin = [t for t in topics if per_topic[t][1] == 0]
        print(f"topics with 0 excluded: {len(thin)}  {thin if thin else ''}")

        # ---- figure 1: global distribution ---------------------------------
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = [by_judgement[j] for j in (0, 1, 2)]
        bars = ax.bar(
            [LABELS[j] for j in (0, 1, 2)], counts, color=["#bbbbbb", "#e8833a", "#3a7ce8"]
        )
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=8)
        ax.set_ylabel("judgements")
        ax.set_title(f"TREC CT {year} — judgement distribution")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"qrels{year}_distribution.png", dpi=150)

        # ---- figure 2: composition per topic -------------------------------
        order = sorted(topics, key=lambda t: -per_topic[t][2])
        fig, ax = plt.subplots(figsize=(10, 3.5))
        x = range(len(order))
        e2 = [per_topic[t][2] for t in order]
        e1 = [per_topic[t][1] for t in order]
        e0 = [per_topic[t][0] for t in order]
        ax.bar(x, e2, label="eligible", color="#3a7ce8")
        ax.bar(x, e1, bottom=e2, label="excluded", color="#e8833a")
        ax.bar(
            x,
            e0,
            bottom=[a + b for a, b in zip(e2, e1)],
            label="not relevant",
            color="#dddddd",
        )
        ax.set_xlabel("topics (sorted by eligible count)")
        ax.set_ylabel("judged trials")
        ax.set_title(f"TREC CT {year} — pool composition per topic")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"qrels{year}_per_topic.png", dpi=150)

        # ---- figure 3: how many eligible trials per topic -------------------
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(eligible, bins=20, color="#3a7ce8")
        ax.set_xlabel("eligible trials in a topic")
        ax.set_ylabel("topics")
        ax.set_title(f"TREC CT {year} — eligible trials per topic")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"qrels{year}_eligible_hist.png", dpi=150)

        print(f"\nfigures written to {FIGURES}/")


if __name__ == "__main__":
    main()
