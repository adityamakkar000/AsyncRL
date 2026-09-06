# /// script
# dependencies = ["wandb", "pandas", "matplotlib"]
# ///
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

TRAIN_FIGURES = {
    "mean_reward": "train/mean_reward",
    "is_ratio": "train/is_ratio",
    "seq_len": "train/mean_length",
    "weight_sync_time": "train/weight_sync_time",
    "total_step_time": "train/step_time",
    "rollout_wait_time": "train/rollout_queue_wait_time",
    "tps": "inference/worker_0/decode_tps",
}
EVAL_DATASETS = ["aime_2025", "aime_2026", "amc_25", "math_500"]
EVAL_METRICS = {"avg_at_8": "avg@8", "pass_at_32": "pass@k=32"}

WANDB_COLORS = [
    "#5387DD",
    "#DA4C4C",
    "#479A5F",
    "#7D54B2",
    "#E87B9F",
    "#E57439",
    "#87CEBF",
    "#229487",
    "#A0C75C",
    "#A46750",
]
TEXT = "#3B3B44"
MUTED = "#8B8B96"
GRID = "#E6E6EA"
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
    }
)


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def find_run(api, entity, project, name):
    try:
        return api.run(f"{entity}/{project}/{name}")
    except wandb.errors.CommError:
        pass
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": name}))
    if not runs:
        raise SystemExit(f"no run named {name!r} in {entity}/{project}")
    if len(runs) > 1:
        print(f"warning: {len(runs)} runs named {name!r}, using most recent")
        runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs[0]


def fetch(run, key):
    if key not in run.summary.keys():
        return None
    df = run.history(keys=[key], samples=100000, pandas=True)
    if key not in df.columns:
        return None
    return df[["_step", key]].dropna().sort_values("_step")


def plot(series, key, title, out, window):
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("white")
    for i, (label, df) in enumerate(series):
        c = WANDB_COLORS[i % len(WANDB_COLORS)]
        x, y = df["_step"].to_numpy(), df[key]
        if len(df) >= 2 * window:
            ax.plot(x, y.to_numpy(), color=c, lw=1, alpha=0.2)
            ax.plot(x, y.rolling(window, min_periods=1).mean().to_numpy(), color=c, lw=1.6, label=label)
        else:
            ax.plot(x, y.to_numpy(), color=c, lw=1.6, marker="o", ms=4, label=label)
    ax.set_title(title, fontsize=12, fontweight="semibold", color=TEXT, pad=12)
    ax.set_xlabel("Step", fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    ax.margins(x=0.01)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=1 if len(series) > 3 else len(series),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make(runs, key, title, out, window):
    series = [(r.name, df) for r in runs if (df := fetch(r, key)) is not None and not df.empty]
    if not series:
        print(f"skip {title}: no data")
        return
    plot(series, key, title, out, window)
    print(f"wrote {out}")


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "results"))
    ap.add_argument("--smooth", type=int, default=25)
    args = ap.parse_args()

    api = wandb.Api()
    runs = [find_run(api, args.entity, args.project, n) for n in args.runs]
    for r in runs:
        print(f"run {r.name} ({r.id}) state={r.state}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, key in TRAIN_FIGURES.items():
        make(runs, key, name, out_dir / f"{name}.png", args.smooth)
    for ds in EVAL_DATASETS:
        for mname, metric in EVAL_METRICS.items():
            make(runs, f"eval/{ds}/{metric}", f"{ds} {metric}", out_dir / f"{ds}_{mname}.png", args.smooth)


if __name__ == "__main__":
    main()
