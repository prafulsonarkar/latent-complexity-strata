"""
p3_minority.py -- does the complexity measure h(x) pick out the same images as the minority score of
Um, Lee & Ye (ICLR 2024, arXiv 2301.12334), and does the generator under-produce high-minority images too?

Minority score (Um et al., Eq. for l(x0)): E_eps[ LPIPS(x0, x0_hat(x_t)) ], where x0_hat is the one-step (Tweedie) estimate
of the clean sample from a noised version. For our flow-matching DiT (x_t = (1-t) x0 + t eps, network predicts v = eps - x0)
the one-step estimate is x0_hat = x_t - t * v. Both latents are decoded with the run's VAE and compared with LPIPS, so the
score measures what the GENERATOR fails to restore (tokenizer error is excluded: decode(x0) is used, not the photo).
Adaptations vs Um et al. (state them in the paper): latent flow matching instead of pixel DDPM; LPIPS(vgg, as in Paper 2); t in {0.5, 0.7, 0.9}
all reported (Um et al. tune t; we do not pick one after seeing results); --k noise draws per image.

Reports, per run and per t:
  * Spearman rho between the minority score of the real test images and h(x), absolute HF energy, and log MSE of the VAE
  * mean minority score in each real h-quintile (Q1 simple ... Q5 complex)
  * overlap: % of the top-20% minority images that are in the top-20% h (20% = unrelated)
  * generated images: minority score of the 10k generated images (re-encoded with the VAE) and their share above the real
    80th percentile (20% = matched; below 20% = the generator under-produces minority images, as Um et al. report)
Output: runs_p3/<ds>_<tag>[_seed<k>]/minority.json and a printed table.

Run from D:\\new_vae_research after `sample` (needs dit.pt, latents.npz, gen_images.npz). GPU: ~10-20 min per run.
  python p3_minority.py --dataset cifar10 --root D:\\new_vae_research\\data      --tags B_strong_mse D_strong_gan_v2
  python p3_minority.py --dataset celeba  --root D:\\new_vae_research\\celeba64  --tags B_strong_mse D_strong_gan
"""
import argparse, json, os
import numpy as np
import torch
from scipy.stats import spearmanr

from p3_latent_gen import load_vae, load_latents, denorm, make_dit, out_dir, test_images
from p3_h_check import spectra
from paper2_train import to_pm1, to_01


@torch.no_grad()
def minority_scores(zn, model, dec, z, lp, dev, t, k, amp, seed, bs=250):
    """zn: normalised latents (N,4,8,8) on CPU. Returns (N,) mean LPIPS between decode(x0) and decode(x0_hat)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for s in range(0, len(zn), bs):
        x0 = zn[s:s + bs].to(dev)
        ref = to_pm1(to_01(dec(denorm(x0, z, dev))).clamp(0, 1))          # LPIPS expects [-1, 1]
        acc = torch.zeros(len(x0), device=dev)
        for _ in range(k):
            eps = torch.randn(x0.shape, device=dev, generator=g)
            xt = (1 - t) * x0 + t * eps
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev == "cuda"):
                v = model(xt, torch.full((len(x0),), t, device=dev)).float()
            x0h = xt - t * v
            img = to_pm1(to_01(dec(denorm(x0h, z, dev))).clamp(0, 1))
            acc += lp(ref, img).flatten()
        out.append((acc / k).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--out_root", default="runs_p3"); ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ts", default="0.5,0.7,0.9"); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex"], help="vgg = same LPIPS as Paper 2")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import lpips
    lp = lpips.LPIPS(net=a.lpips_net, verbose=False).to(dev).eval()
    ts = [float(v) for v in a.ts.split(",")]
    table = []
    real = Er = Hr = hr = None
    for tag in a.tags:
        a.tag = tag
        d = out_dir(a, True)
        try:
            enc, dec, vargs = load_vae(a, dev)
            ck = torch.load(os.path.join(d, "dit.pt"), map_location=dev, weights_only=False)
        except FileNotFoundError as e:
            print(f"skip {tag}: {e}"); continue
        z, norm = load_latents(a, ck["args"]["latent"])
        model = make_dit(argparse.Namespace(**ck["args"])).to(dev).eval()
        model.load_state_dict(ck["ema"])
        if real is None:
            real = test_images(a, vargs)
            Er, Hr, hr = spectra(real)
        zt = norm(z["test_mu"])
        if len(zt) != len(real):
            raise SystemExit(f"{tag}: {len(zt)} test latents but {len(real)} test images; re-run encode")
        # VAE reconstruction error of each test image (tokenizer difficulty), for comparison
        with torch.no_grad():
            mse = torch.cat([((to_01(dec(enc(to_pm1(real[s:s + 250].to(dev)))[0])).cpu() - real[s:s + 250]) ** 2)
                             .flatten(1).mean(1) for s in range(0, len(real), 250)]).numpy()
        # generated images -> latents (posterior mean), same normalisation
        gf = os.path.join(d, "gen_images.npz")
        zg = None
        if os.path.exists(gf):
            fake = torch.from_numpy(np.load(gf)["x"]).float() / 255
            with torch.no_grad():
                mu = torch.cat([enc(to_pm1(fake[s:s + 250].to(dev)))[0].cpu() for s in range(0, len(fake), 250)])
            zg = norm(mu.numpy())
        cuts = np.quantile(hr, [.2, .4, .6, .8]); q = np.digitize(hr, cuts)
        res = {"tag": tag, "dataset": a.dataset, "seed": a.seed, "k": a.k, "per_t": {}}
        for t in ts:
            print(f"{tag}: minority score at t = {t} ...", flush=True)
            ms = minority_scores(zt, model, dec, z, lp, dev, t, a.k, a.amp, seed=0)
            top_ms, top_h = ms >= np.quantile(ms, 0.8), hr >= np.quantile(hr, 0.8)
            r = {"rho_h": float(spearmanr(ms, hr).correlation), "rho_abs_hf": float(spearmanr(ms, Hr).correlation),
                 "rho_vae_logmse": float(spearmanr(ms, np.log(mse)).correlation),
                 "mean_by_h_quintile": [float(ms[q == j].mean()) for j in range(5)],
                 "overlap_top20_pct": float(100 * (top_ms & top_h).sum() / top_ms.sum())}
            if zg is not None:
                mg = minority_scores(zg, model, dec, z, lp, dev, t, a.k, a.amp, seed=1)
                r["gen_share_above_real_p80_pct"] = float(100 * (mg >= np.quantile(ms, 0.8)).mean())
                r["gen_mean"], r["real_mean"] = float(mg.mean()), float(ms.mean())
            res["per_t"][str(t)] = r
            np.save(os.path.join(d, f"minority_t{t}.npy"), ms)
        json.dump(res, open(os.path.join(d, "minority.json"), "w"), indent=1)
        table.append(res)

    print(f"\n{'run':>18s} {'t':>4s} {'rho(ms,h)':>9s} {'rho(ms,absHF)':>13s} {'rho(ms,VAE logMSE)':>18s} {'overlap%':>9s}"
          f" {'gen>p80 %':>9s}   mean minority score by real h-quintile Q1..Q5")
    for res in table:
        for t, r in res["per_t"].items():
            print(f"{res['tag']:>18s} {t:>4s} {r['rho_h']:9.2f} {r['rho_abs_hf']:13.2f} {r['rho_vae_logmse']:18.2f}"
                  f" {r['overlap_top20_pct']:9.1f} {r.get('gen_share_above_real_p80_pct', float('nan')):9.1f}   "
                  + " ".join(f"{v:.3f}" for v in r["mean_by_h_quintile"]))
    print("overlap% = share of the top-20% minority images that are also top-20% h (20 = unrelated, 100 = same images).\n"
          "gen>p80 % = share of generated images whose minority score exceeds the real 80th percentile (20 = matched).")


if __name__ == "__main__":
    main()
