# /// script
# dependencies = ["wandb", "pandas", "matplotlib"]
# ///
import argparse
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

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

FIGURES = {
    "mean_reward": ["train/mean_reward"],
    "avg_at_8": ["eval/{ds}/avg@8"],
    "pass_at_32": ["eval/{ds}/pass@k=32"],
    "is_ratio": ["train/is_ratio"],
    "seq_len": ["train/mean_length", "train/median_length"],
    "weight_sync_time": ["train/weight_sync_time"],
    "total_step_time": ["train/step_time"],
    "rollout_wait_time": ["train/rollout_queue_wait_time"],
    "tps": ["inference/worker_0/decode_tps"],
}


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


def expand_keys(patterns, available):
    keys = []
    for p in patterns:
        if "{ds}" in p:
            rx = re.compile("^" + re.escape(p).replace(re.escape("{ds}"), "([^/]+)") + "$")
            keys += sorted(k for k in available if rx.match(k))
        elif p in available:
            keys.append(p)
    return keys


def plot(df, keys, title, out, window):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.set_facecolor("white")
    for i, k in enumerate(keys):
        c = WANDB_COLORS[i % len(WANDB_COLORS)]
        sub = df[["_step", k]].dropna().sort_values("_step")
        if sub.empty:
            continue
        x, y = sub["_step"].to_numpy(), sub[k].to_numpy()
        if len(sub) >= 2 * window:
            ax.plot(x, y, color=c, lw=1, alpha=0.2)
            ax.plot(x, sub[k].rolling(window, min_periods=1).mean().to_numpy(), color=c, lw=1.6, label=k)
        else:
            ax.plot(x, y, color=c, lw=1.6, marker="o", ms=4, label=k)
    ax.set_title(title, loc="center", fontsize=12, fontweight="semibold", color=TEXT, pad=12)
    ax.set_xlabel("Step", fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)
    ax.margins(x=0.01)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(len(keys), 4), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("run")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--out", default=str(Path.home() / "Desktop"))
    ap.add_argument("--smooth", type=int, default=25)
    args = ap.parse_args()

    api = wandb.Api()
    run = find_run(api, args.entity, args.project, args.run)
    print(f"run {run.name} ({run.id}) state={run.state}")

    available = set(run.summary.keys())
    out_dir = Path(args.out) / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pats in FIGURES.items():
        keys = expand_keys(pats, available)
        if not keys:
            print(f"skip {name}: no matching keys")
            continue
        df = run.history(keys=keys, samples=100000, pandas=True)
        keys = [k for k in keys if k in df.columns]
        path = out_dir / f"{name}.png"
        plot(df, keys, name, path, args.smooth)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
