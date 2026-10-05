"""Plot training/validation curves from TensorBoard event files without a TensorBoard server."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def main(runs, output):
    data = {}
    for spec in runs:
        name, path = spec.split("=", 1)
        data[name] = load(path)

    tags = ["training_loss", "training_loss_running_mean", "validation_loss", "learning_rate"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, tag in zip(axs.flat, tags):
        for name, scalars in data.items():
            if tag in scalars and scalars[tag]:
                x, y = zip(*scalars[tag])
                ax.plot(x, y, label=name)
        ax.set_title(tag)
        ax.set_xlabel("samples")
        if tag != "learning_rate":
            ax.set_yscale("log")
        ax.legend()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=120)
    print("saved", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="NAME=path/to/events.file (repeat)")
    p.add_argument("--output", default="plots/training_curves.png")
    a = p.parse_args()
    main(a.run, a.output)
