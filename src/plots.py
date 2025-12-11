import os
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os

def plot_mia_curves(mia_runner, settings, save_root="plots/results"):
    """
    Plot MIA AUC + victim train/test accuracy per round
    """
    rounds = list(range(1, len(mia_runner.aucs) + 1))

    # Filter out None values
    auc_rounds  = [r for r, v in zip(rounds, mia_runner.aucs)        if v is not None]
    auc_vals    = [v for v in mia_runner.aucs                        if v is not None]

    train_rounds = [r for r, v in zip(rounds, mia_runner.train_accs) if v is not None]
    train_vals   = [v for v in mia_runner.train_accs                 if v is not None]

    test_rounds  = [r for r, v in zip(rounds, mia_runner.test_accs)  if v is not None]
    test_vals    = [v for v in mia_runner.test_accs                  if v is not None]

    fig, ax1 = plt.subplots(figsize=(10,6))

    # Left y-axis: accuracies (0–100)
    ax1.plot(train_rounds, train_vals, label="Train accuracy (MIA)", linestyle="--")
    ax1.plot(test_rounds,  test_vals,  label="Test accuracy (MIA)", linestyle="-")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True)

    # Right y-axis: AUC (0–1)
    ax2 = ax1.twinx()
    ax2.plot(auc_rounds, auc_vals, label="MIA AUC", linestyle="-.", marker="o", alpha=0.8)
    ax2.set_ylabel("AUC")

    # Build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title(
        f"MIA AUC & accuracy over rounds\n"
        f"dataset={settings.dataset}, model={settings.model}, topology={settings.topology}"
    )
    plt.tight_layout()

    # Build filename
    filename = (
        f"mia_curves_"
        f"{settings.dataset}_"
        f"{settings.model}_"
        f"{settings.topology}_"
        f"peers{settings.participants}_"
        f"rounds{settings.rounds}.png"
    )
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, filename)
    plt.savefig(save_path)
    print(f"Saved MIA plot to {save_path}")

    plt.show()


def draw_topology(save_root, topology, topology_name):
    os.makedirs(save_root, exist_ok=True)  
    topology.draw(save_path= save_root + topology_name)