"""
make_p3_figs.py -- draws the two data figures of Paper 3 straight from the result files (no typed numbers).

  fig_gaps.pdf   : tokenizer gap and generator gap on absolute HF energy, per tokenizer, CIFAR-10 and CelebA
                   (runs_p3/<ds>_<tag>/h_check.json: tok_gap_q5_abs_hf, gen_gap_q5_abs_hf)
  fig_shares.pdf : share of generated images per real complexity quintile for base / a_gan / b_gan, 3 settings
                   (runs_p3_dec/<arm>/<setting>/gen_eval.json: Q1..Q5 share_gen)

Run from D:\\new_vae_research:   python make_p3_figs.py
Change the tag lists below if your folder names differ. A missing file is skipped and reported.
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOK = {"cifar10": [("A", "A_small_mse"), ("B", "B_strong_mse"), ("G", "G_small_lpips_gan"), ("D", "D_strong_gan_v2")],
       "celeba": [("A", "A_small_mse"), ("B", "B_strong_mse"), ("G", "G_small_lpips_gan"), ("D", "D_strong_gan")]}
SETTINGS = [("CelebA, seed 0", "celeba_D_strong_gan"), ("CelebA, seed 1", "celeba_D_strong_gan_seed1"),
            ("CIFAR-10", "cifar10_D_strong_gan_v2")]
ARMS = [("base", "Base", "#9a9a9a"), ("a_gan", "a_gan (anchor only)", "#e0a458"), ("b_gan", "b_gan (anchor + spectrum)", "#2b6cb0")]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})


def load(p):
    if not os.path.exists(p):
        print("missing:", p); return None
    return json.load(open(p))


def fig_gaps(a):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    for ax, (ds, title) in zip(axs, (("cifar10", "CIFAR-10"), ("celeba", "CelebA"))):
        names, tg, gg = [], [], []
        for lab, tag in TOK[ds]:
            H = load(os.path.join(a.p3, f"{ds}_{tag}", "h_check.json"))
            if H is None or "tok_gap_q5_abs_hf" not in H:
                continue
            names.append(lab); tg.append(H["tok_gap_q5_abs_hf"]); gg.append(H["gen_gap_q5_abs_hf"])
        x = range(len(names))
        ax.bar([i - 0.19 for i in x], tg, 0.38, color="#e0a458", label="tokenizer gap")
        ax.bar([i + 0.19 for i in x], gg, 0.38, color="#2b6cb0", label="generator gap")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(list(x)); ax.set_xticklabels(names); ax.set_title(title, fontsize=8)
    axs[0].set_ylabel("Q5 share points (abs. HF)")
    axs[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_gaps.pdf")); plt.close(fig)
    print("saved", os.path.join(a.out, "fig_gaps.pdf"))


def fig_shares(a):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.9), sharey=True)
    for ax, (title, sub) in zip(axs, SETTINGS):
        for k, (arm, lab, col) in enumerate(ARMS):
            G = load(os.path.join(a.dec, arm, sub, "gen_eval.json"))
            if G is None:
                continue
            ys = [100 * G[f"Q{j}"]["share_gen"] for j in range(1, 6)]
            ax.bar([j + (k - 1) * 0.27 for j in range(5)], ys, 0.27, color=col, label=lab)
        ax.axhline(20, color="k", ls="--", lw=0.6)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{j}" for j in range(1, 6)]); ax.set_title(title, fontsize=8)
    axs[0].set_ylabel("generated share (%)")
    axs[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_shares.pdf")); plt.close(fig)
    print("saved", os.path.join(a.out, "fig_shares.pdf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", default="runs_p3"); ap.add_argument("--dec", default="runs_p3_dec")
    ap.add_argument("--out", default="paper3_figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_gaps(a); fig_shares(a)
