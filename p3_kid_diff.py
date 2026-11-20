"""
p3_kid_diff.py -- is the KID difference between two decoder arms larger than its noise?

The pass tests compare single KID values (e.g. b_gan 7.63 vs base 7.77). This script gives the uncertainty of that
difference so the paper can say "no measurable change" or "worse by X [CI]" instead of reading a bare inequality.
Paired subsets: in every repetition the SAME random real subset is used for both arms, and each arm gets its own random
generated subset of the same size. Reported: mean KID(arm) - KID(ref) and its 2.5-97.5% range, for all images and for
the most complex quintile (h quintile cuts from the real test images, as in the paper). x1000.

Run from D:\\new_vae_research after `eval` (needs gen_images.npz in each arm folder). GPU ~5-10 min per setting.
  python p3_kid_diff.py --dataset celeba  --root D:\\new_vae_research\\celeba64 --tag D_strong_gan
  python p3_kid_diff.py --dataset celeba  --root D:\\new_vae_research\\celeba64 --tag D_strong_gan --seed 1
  python p3_kid_diff.py --dataset cifar10 --root D:\\new_vae_research\\data     --tag D_strong_gan_v2
"""
import argparse, json, os
import numpy as np
import torch

from nested_vae_spectral import get_datasets
from p3_latent_gen import inception_feats, kid_np
from p3_h_check import spectra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dec_root", default="runs_p3_dec"); ap.add_argument("--ref", default="base")
    ap.add_argument("--arms", nargs="+", default=["a_gan", "b_gan"])
    ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--m", type=int, default=2000, help="subset size, all images")
    ap.add_argument("--m_q5", type=int, default=500, help="subset size, Q5")
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sub = f"{a.dataset}_{a.tag}" + (f"_seed{a.seed}" if a.seed else "")
    _, ev = get_datasets(a.dataset, a.root, 64, n_eval=a.n_eval)
    real = torch.stack([ev[i][0] for i in range(len(ev))])
    _, _, hr = spectra(real)
    cuts = np.quantile(hr, [.2, .4, .6, .8])
    print("Inception features of the real test images ...", flush=True)
    Fr = inception_feats(real, dev)
    r5 = np.where(np.digitize(hr, cuts) == 4)[0]
    F, Q5 = {}, {}
    for arm in [a.ref] + a.arms:
        f = os.path.join(a.dec_root, arm, sub, "gen_images.npz")
        if not os.path.exists(f):
            print(f"skip {arm}: {f} not found"); continue
        fake = torch.from_numpy(np.load(f)["x"]).float() / 255
        print(f"{arm}: Inception features ...", flush=True)
        F[arm] = inception_feats(fake, dev)
        Q5[arm] = np.where(np.digitize(spectra(fake)[2], cuts) == 4)[0]
    if a.ref not in F:
        raise SystemExit(f"reference arm {a.ref} missing")
    rng = np.random.default_rng(0)
    out = {"setting": sub, "ref": a.ref, "reps": a.reps, "arms": {}}
    print(f"\n{sub}: KID x1000, arm minus {a.ref}, mean [95% range] over {a.reps} paired subsets")
    print(f"{'arm':>8s} {'all images':>26s} {'Q5 (most complex)':>28s}")
    for arm in a.arms:
        if arm not in F:
            continue
        m = min(a.m, len(Fr), len(F[arm]), len(F[a.ref]))
        mq = min(a.m_q5, len(r5), len(Q5[arm]), len(Q5[a.ref]))
        d_all, d_q5 = [], []
        for _ in range(a.reps):
            R = Fr[rng.choice(len(Fr), m, replace=False)]
            d_all.append(kid_np(R, F[arm][rng.choice(len(F[arm]), m, replace=False)])
                         - kid_np(R, F[a.ref][rng.choice(len(F[a.ref]), m, replace=False)]))
            if mq >= 50:
                R5 = Fr[rng.choice(r5, mq, replace=False)]
                d_q5.append(kid_np(R5, F[arm][rng.choice(Q5[arm], mq, replace=False)])
                            - kid_np(R5, F[a.ref][rng.choice(Q5[a.ref], mq, replace=False)]))
        s = lambda d: [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if d else [float("nan")] * 3
        out["arms"][arm] = {"all": s(d_all), "q5": s(d_q5), "m": m, "m_q5": mq}
        A, B = out["arms"][arm]["all"], out["arms"][arm]["q5"]
        print(f"{arm:>8s}   {A[0]:7.2f} [{A[1]:6.2f}, {A[2]:6.2f}]      {B[0]:7.2f} [{B[1]:6.2f}, {B[2]:6.2f}]  (m={m}, m_q5={mq})")
    print("A range that contains 0 = no measurable difference from the reference; above 0 = worse than the reference.")
    json.dump(out, open(os.path.join(a.dec_root, f"kid_diff_{sub}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
