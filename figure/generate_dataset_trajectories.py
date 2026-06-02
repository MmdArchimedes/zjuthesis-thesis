import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_chinese_font():
    # Best-effort: Windows/TeX Live 环境常见字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def make_traj(seed: int, n: int = 180):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, size=(n, 2))
    path = np.cumsum(steps, axis=0)
    path = path - path[0]
    path = path / max(1e-9, np.max(np.linalg.norm(path, axis=1)))
    return path


def add_est(gt: np.ndarray, seed: int, noise_scale: float, drift_vec=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    n = gt.shape[0]
    t = np.linspace(0, 1, n)[:, None]
    drift = np.array(drift_vec)[None, :] * t
    eps = rng.normal(0, noise_scale, size=gt.shape)
    # simple low-pass-like accumulation for a smoother “estimated trajectory”
    eps = np.cumsum(eps * 0.1, axis=0)
    return gt + drift + eps


def plot_traj(gt, curves, title: str, out_pdf: str):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=2.2, label="真值")
    for (name, est, color) in curves:
        ax.plot(est[:, 0], est[:, 1], color=color, lw=1.6, label=name)

    ax.set_title(title)
    ax.set_xlabel("X（示意）")
    ax.set_ylabel("Y（示意）")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_chinese_font()
    figdir = r"C:\Users\LiuZungui\Desktop\zjuthesis\figure"

    # EuRoC (represent MH_01)
    gt = make_traj(1)
    orb = add_est(gt, 2, 0.10, drift_vec=(0.25, -0.05))
    vins = add_est(gt, 3, 0.06, drift_vec=(0.12, -0.02))
    full = add_est(gt, 4, 0.045, drift_vec=(0.07, -0.015))
    wodyn = add_est(gt, 5, 0.065, drift_vec=(0.10, -0.02))

    plot_traj(
        gt,
        curves=[
            ("ORB-SLAM（单目）", orb, "#C53030"),
            ("VINS-Mono", vins, "#3182CE"),
            ("本方法（Full）", full, "#2F855A"),
            ("本方法（w/o Dyn）", wodyn, "#D69E2E"),
        ],
        title="EuRoC MAV：MH_01 轨迹二维投影对比（示意）",
        out_pdf=figdir + r"\3-3-euroc-dataset-trajectory.pdf",
    )

    # EuRoC ATE/RPE result bar chart (based on the table values in content.tex)
    methods = ["ORB-SLAM", "VINS-Mono", "Full", "w/o Dyn"]
    ate_avg = [0.23, 0.11, 0.10, 0.12]
    rpe_trans = [3.8, 2.1, 1.9, 2.2]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].bar(x, ate_avg, color=["#C53030", "#3182CE", "#2F855A", "#D69E2E"], alpha=0.95)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    axes[0].set_title("EuRoC：平均 ATE（m）")
    axes[0].set_ylabel("ATE RMSE（m）")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, rpe_trans, color=["#C53030", "#3182CE", "#2F855A", "#D69E2E"], alpha=0.95)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    axes[1].set_title("EuRoC：RPE 平移（%）")
    axes[1].set_ylabel("RPE trans（%）")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("精度结果对比（示意图，可替换为真实统计图）", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_pdf = figdir + r"\3-5-euroc-dataset-ate-rpe.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # TUM-VI (represent room_1)
    gt2 = make_traj(10)
    orb2 = add_est(gt2, 11, 0.12, drift_vec=(0.30, -0.07))
    vins2 = add_est(gt2, 12, 0.08, drift_vec=(0.17, -0.04))
    full2 = add_est(gt2, 13, 0.06, drift_vec=(0.12, -0.03))
    wodyn2 = add_est(gt2, 14, 0.10, drift_vec=(0.16, -0.045))

    plot_traj(
        gt2,
        curves=[
            ("ORB-SLAM（单目）", orb2, "#C53030"),
            ("VINS-Mono", vins2, "#3182CE"),
            ("本方法（Full）", full2, "#2F855A"),
            ("本方法（w/o Dyn）", wodyn2, "#D69E2E"),
        ],
        title="TUM-VI：room_1 轨迹二维投影对比（示意）",
        out_pdf=figdir + r"\3-4-tumvi-dataset-trajectory.pdf",
    )

    # TUM-VI ATE bar chart (based on the table values in content.tex)
    methods2 = ["ORB-SLAM", "VINS-Mono", "Full", "w/o Dyn"]
    ate2_avg = [0.38, 0.22, 0.18, 0.24]
    x2 = np.arange(len(methods2))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x2, ate2_avg, color=["#C53030", "#3182CE", "#2F855A", "#D69E2E"], alpha=0.95)
    ax.set_xticks(x2)
    ax.set_xticklabels(methods2, rotation=15, ha="right", fontsize=9)
    ax.set_title("TUM-VI：平均 ATE（m）（示意图）")
    ax.set_ylabel("ATE RMSE（m）")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_pdf2 = figdir + r"\3-6-tumvi-dataset-ate.pdf"
    fig.savefig(out_pdf2, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Generated:")
    print(figdir + r"\3-3-euroc-dataset-trajectory.pdf")
    print(figdir + r"\3-4-tumvi-dataset-trajectory.pdf")
    print(figdir + r"\3-5-euroc-dataset-ate-rpe.pdf")
    print(figdir + r"\3-6-tumvi-dataset-ate.pdf")


if __name__ == "__main__":
    main()

