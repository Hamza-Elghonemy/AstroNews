from pathlib import Path
import matplotlib.pyplot as plt


def plot_breakdown(title: str, sem: float, kw: float, rec: float,
                   sem_w: float, kw_w: float, rec_w: float, out_path: Path):
    """
    Save a horizontal stacked bar showing each component's weighted contribution.
    All inputs sem/kw/rec are normalized in [0,1]; weights sum to 1.
    """
    # contributions (already weighted)
    c_sem = sem_w * sem
    c_kw  = kw_w  * kw
    c_rec = rec_w * rec
    total = c_sem + c_kw + c_rec

    fig, ax = plt.subplots(figsize=(7, 2.6))
    # One bar, stack three segments
    left = 0.0
    for val, label in [(c_sem, "Semantic"), (c_kw, "Keyword"), (c_rec, "Recency")]:
        ax.barh([title], [val], left=left)  # default colors; one chart per figure
        # center label on the segment if it's wide enough
        if val > 0.04:
            ax.text(left + val/2, 0, f"{label}\n{val:.2f}", ha="center", va="center", color="white", fontweight="bold")
        left += val

    ax.set_xlim(0, max(1.0, total))  # keep it readable; total often <= 1
    ax.set_title(f"Hybrid Score Breakdown (final={total:.2f})")
    ax.set_ylabel("")  # tidy
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_debug_plots(candidates_sorted, top_n: int, out_dir: Path,
                     sem_w: float, kw_w: float, rec_w: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(candidates_sorted[:top_n], 1):
        fname = out_dir / f"{i:02d}.png"
        plot_breakdown(
            title=c["title"][:48] + ("…" if len(c["title"]) > 48 else ""),
            sem=c["semantic_norm"],
            kw=c["keyword_norm"],
            rec=c["recency"],
            sem_w=sem_w, kw_w=kw_w, rec_w=rec_w,
            out_path=fname
        )
        print(f"Saved plot: {fname}")