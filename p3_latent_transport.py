"""
p3_latent_transport.py -- candidate method: frozen-decoder latent transport (step 1: diagnostic, no training).

Question: is the generator gap (complex images lost in GENERATION, decoder and encoder fixed) visible as a mismatch in
the second moments of the generated latents? If yes, a map T on the final latents (generator and decoder frozen) can
close part of it; if no (variance ratios ~1 everywhere), the second-moment story is dropped (kill rule below).

`diag` reports, for one setting (dataset, tag, DiT seed):
  A. Latent second moments, in the normalised 256-D space the DiT models (z sampled from the posterior, as in training):
     - variance ratio var_gen / var_real along each eigenvector of the real covariance (sorted from largest to smallest
       real variance), summarised per decile of directions, plus min / max;
     - mean shift along the same directions (in real standard deviations);
     - Frechet distance between Gaussian fits: generated vs real, and real half A vs real half B (noise floor).
  B. Ceilings for the top-quintile share S5 on h and on absolute HF energy (quintile cuts from the real test images):
     - reconstructions from the posterior MEAN (what p3_h_check.py reported, e.g. CelebA D 13.3, CIFAR-10 D 16.0);
     - reconstructions from posterior SAMPLES (what the DiT was trained on; the honest ceiling for a latent map);
     - the generated evaluation latents decoded with the unchanged decoder (the starting point).
  Kill rule (fixed before the run, 12 Nov 2026): if every decile variance ratio is within max(3%, 2 x null sd) of 1
  and the mean shifts stay below max(0.03 sd, 2 x null shift), the second-moment explanation is dropped and step 2 (L0)
  is not run. "Null" = the same statistics for a second random third of the real latents against the first, which
  shows how far from 1 the ratios move by sampling noise alone.

Inputs it reuses (no new generation): runs_p3/<ds>_<tag>/latents.npz and the cached generated latents
runs_p3_dec/<ds>_<tag>[_seed<k>]/gen_latents_{train,eval}.npy written by `p3_decoder_align.py gen`.
Output: runs_p3_lt/<ds>_<tag>[_seed<k>]/diag.json, var_ratio.png. CPU is enough for part A; part B decodes 3 x 10k
images (GPU ~2 min, CPU ~10 min).

Step 2 (`apply`, `table`): L0 = closed-form Gaussian optimal-transport map on the normalised, flattened 256-D latent,
fitted from the 50k cached generated TRAINING latents to real posterior samples, applied to the 10k generated EVALUATION
latents (different noise seeds), decoded with the UNCHANGED decoder. Arms:
  base : no map            l0 : T(z) = mu_r + A (z - mu_g),  A = S_g^-1/2 (S_g^1/2 S_r S_g^1/2)^1/2 S_g^-1/2
  iso  : control, mu_r + s (z - mu_g) with s = sqrt(tr S_r / tr S_g) (same total variance gain, no direction info)
  null : control, the same L0 formula fitted between two halves of the REAL latents and applied to the generated ones
         (should change almost nothing)
Pass test for l0 (written 12 Nov 2026, after the CIFAR-10 diagnostic and BEFORE any l0 run; relative to each setting):
  1. S5 on abs HF >= base + 0.5 x (ceiling - base), ceiling = S5 of decoded posterior samples from diag.json
  2. low-energy share of Q4/Q5 (LowE45) <= base        3. KID <= base        4. KID in Q5 <= base
  5. the iso control gains less than half of l0's S5 gain on abs HF
  6. the null control moves S5 on abs HF by less than 1.0 point
  rFID cannot change (the decoder is not modified); it is printed only as a sanity check.
  Result on CIFAR-10 D (12 Nov 2026): NOT PASSED -- S5 abs HF 10.4 -> 15.3, but iso alone reached 14.3 (check 5) and
  KID was 0.14 above base (check 3; paired difference 0.12 [-1.64, 1.59], not measurable).
Second pass test, for iso as a method of its own (written 12 Nov 2026 after the CIFAR-10 run and BEFORE any CelebA
apply run; on CIFAR-10 iso was a control, so CIFAR-10 does not count as a test of it):
  1. S5 on abs HF >= base + 0.5 x (ceiling - base)   2. LowE45 <= base   3. KID <= base   4. KID in Q5 <= base
  5. the null control moves S5 on abs HF by less than 1.0 point
  The l0 test above is applied unchanged on CelebA.
  After `apply` for all arms, run p3_h_check.py on runs_p3_lt\\<arm> (--no_kid) and p3_kid_diff.py with
  --dec_root runs_p3_lt --arms l0 iso null, then `table`.

Run from D:\\new_vae_research:
  python p3_latent_transport.py diag --dataset celeba  --root D:\\new_vae_research\\celeba64 --tag D_strong_gan --seed 1
  python p3_latent_transport.py diag --dataset cifar10 --root D:\\new_vae_research\\data     --tag D_strong_gan_v2
"""
import argparse, json, os
import numpy as np
import torch

from p3_latent_gen import load_vae, load_latents
from p3_h_check import spectra
from paper2_train import to_01


def sfx(a):
    return f"_seed{a.seed}" if a.seed else ""


def gauss_fd(m1, C1, m2, C2):
    """Frechet distance between two Gaussians (same formula as FID)."""
    ev = np.linalg.eigvals(C1 @ C2)
    tr_sqrt = np.sqrt(np.clip(ev.real, 0, None)).sum()
    return float(((m1 - m2) ** 2).sum() + np.trace(C1) + np.trace(C2) - 2 * tr_sqrt)


@torch.no_grad()
def decode_all(dec, z, dev, bs=250):
    return torch.cat([to_01(dec(z[s:s + bs].to(dev))).clamp(0, 1).cpu() for s in range(0, len(z), bs)])


def q5(cuts, v):
    return float(100 * (np.digitize(v, cuts) == 4).mean())


def diag(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = argparse.Namespace(**vars(a)); base.seed = 0; base.out_root = a.base_root
    z, norm = load_latents(base, "sample")
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    zs = mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)   # posterior samples (DiT input)
    gdir = os.path.join(a.dec_root, f"{a.dataset}_{a.tag}{sfx(a)}")
    G = np.load(os.path.join(gdir, "gen_latents_train.npy")).astype(np.float32)
    Ge = np.load(os.path.join(gdir, "gen_latents_eval.npy")).astype(np.float32)
    R = norm(zs).flatten(1).double().numpy()
    Gn = norm(G).flatten(1).double().numpy()

    # ---- A. second moments along the real eigenbasis.
    # Eigen-directions are estimated on real third A and the real variance along them on real third B, so the
    # ratios are not biased by sorting noisy eigenvalues; real third C gives the null ratios (same data, no mismatch).
    perm = rng.permutation(len(R)); n3 = len(R) // 3
    RA, RB, RC = R[perm[:n3]], R[perm[n3:2 * n3]], R[perm[2 * n3:3 * n3]]
    w, V = np.linalg.eigh(np.cov(RA, rowvar=False))
    V = V[:, np.argsort(w)[::-1]]
    proj_var = lambda X: ((X - X.mean(0)) @ V).var(0, ddof=1)
    vB, vG, vC = proj_var(RB), proj_var(Gn), proj_var(RC)
    ratio, null = vG / np.maximum(vB, 1e-12), vC / np.maximum(vB, 1e-12)
    shift = ((Gn.mean(0) - RB.mean(0)) @ V) / np.sqrt(np.maximum(vB, 1e-12))
    shift0 = ((RC.mean(0) - RB.mean(0)) @ V) / np.sqrt(np.maximum(vB, 1e-12))
    dec_idx = np.array_split(np.arange(V.shape[1]), 10)
    deciles = [{"decile": i + 1, "real_var_mean": float(vB[ix].mean()), "ratio_mean": float(ratio[ix].mean()),
                "ratio_min": float(ratio[ix].min()), "ratio_max": float(ratio[ix].max()),
                "null_ratio_mean": float(null[ix].mean()), "null_ratio_sd": float(null[ix].std()),
                "abs_shift_mean_sd": float(np.abs(shift[ix]).mean()), "null_abs_shift_mean_sd": float(np.abs(shift0[ix]).mean())}
               for i, ix in enumerate(dec_idx)]
    mr, Cr = RB.mean(0), np.cov(RB, rowvar=False)
    fd_gen = gauss_fd(Gn.mean(0), np.cov(Gn, rowvar=False), mr, Cr)
    fd_floor = gauss_fd(RC.mean(0), np.cov(RC, rowvar=False), mr, Cr)
    # kill rule: every decile ratio within 3% of 1 AND within the null spread (2 x its sd), and no mean shift
    kill = all(abs(d["ratio_mean"] - 1) <= max(0.03, 2 * d["null_ratio_sd"]) for d in deciles) and \
        max(d["abs_shift_mean_sd"] for d in deciles) < max(0.03, 2 * max(d["null_abs_shift_mean_sd"] for d in deciles))

    # ---- B. ceilings: S5 of decoded posterior means, posterior samples, generated latents (unchanged decoder)
    from nested_vae_spectral import get_datasets
    _, ev = get_datasets(a.dataset, a.root, 64, n_eval=a.n_eval)
    real = torch.stack([ev[i][0] for i in range(len(ev))])
    Er, Hr, hr = spectra(real)
    ch, ce = np.quantile(hr, [.2, .4, .6, .8]), np.quantile(Hr, [.2, .4, .6, .8])
    _, dec, _ = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
    tmu, tlv = z["test_mu"].astype(np.float32), z["test_lv"].astype(np.float32)
    if len(tmu) != len(real):
        raise SystemExit(f"{len(tmu)} test latents but {len(real)} test images; use the same --n_eval as encode")
    tsamp = tmu + np.exp(0.5 * tlv) * rng.standard_normal(tmu.shape).astype(np.float32)
    ceil = {}
    for name, lat in (("recon_mean", tmu), ("recon_sample", tsamp), ("generated", Ge[: a.n_gen_eval])):
        x = decode_all(dec, torch.from_numpy(lat), dev)
        E, H, h = spectra(x)
        ceil[name] = {"S5_h": q5(ch, h), "S5_absHF": q5(ce, H), "n": len(lat)}

    out = {"setting": f"{a.dataset}_{a.tag}{sfx(a)}", "n_real_latents": len(R), "real_split": "eigenbasis on third A, variance on third B, null on third C", "n_gen_latents": len(Gn),
           "deciles": deciles, "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
           "frechet_gauss_gen_vs_real": fd_gen, "frechet_gauss_real_half_vs_half": fd_floor,
           "ceilings": ceil, "kill_rule_triggered": bool(kill)}
    od = os.path.join(a.out, out["setting"]); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "diag.json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.plot(np.arange(1, len(ratio) + 1), ratio, lw=0.8)
        ax.axhline(1, color="k", ls="--", lw=0.6); ax.set_xlabel("real-covariance eigen-direction (largest variance first)")
        ax.set_ylabel("var gen / var real"); ax.set_title(out["setting"], fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(od, "var_ratio.png"), dpi=150); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print(f"\n{out['setting']}: {len(R)} real posterior samples vs {len(Gn)} generated latents (normalised, 256-D)")
    print(f"{'decile':>6s} {'real var':>9s} {'gen/real var':>13s} {'[min, max]':>16s} {'null ratio':>11s}"
          f" {'|shift| sd':>11s} {'null |shift|':>13s}")
    for d in deciles:
        print(f"{d['decile']:6d} {d['real_var_mean']:9.4f} {d['ratio_mean']:13.3f}   [{d['ratio_min']:5.2f}, {d['ratio_max']:5.2f}]"
              f" {d['null_ratio_mean']:11.3f} {d['abs_shift_mean_sd']:11.3f} {d['null_abs_shift_mean_sd']:13.3f}")
    print("(decile 1 = directions with the most real variance; null = a second real subset against the first, i.e. no mismatch)")
    print(f"Frechet distance of Gaussian fits: generated vs real {fd_gen:.3f}; real vs real (noise floor) {fd_floor:.3f}")
    print(f"\nS5 (% in top real quintile; 20 = matched)   {'on h':>6s} {'on abs HF':>10s}")
    for k, v in ceil.items():
        print(f"  {k:<30s} {v['S5_h']:6.1f} {v['S5_absHF']:10.1f}")
    print("\nKILL RULE TRIGGERED: no second-moment mismatch -> drop L0" if kill else
          "\nkill rule not triggered: a second-moment mismatch exists -> L0 is worth running")
    print(f"saved {od}")


def fit_maps(a, z, norm):
    rng = np.random.default_rng(0)
    mu, lv = z["train_mu"].astype(np.float32), z["train_lv"].astype(np.float32)
    zs = mu + np.exp(0.5 * lv) * rng.standard_normal(mu.shape).astype(np.float32)
    R = norm(zs).flatten(1).double().numpy()
    G = norm(np.load(os.path.join(a.dec_root, f"{a.dataset}_{a.tag}{sfx(a)}", "gen_latents_train.npy")).astype(np.float32)).flatten(1).double().numpy()

    def sqrtm_psd(C):
        w, V = np.linalg.eigh((C + C.T) / 2)
        return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

    def ot(src, dst):
        ms, md = src.mean(0), dst.mean(0)
        Cs, Cd = np.cov(src, rowvar=False), np.cov(dst, rowvar=False)
        Ss = sqrtm_psd(Cs)
        w, V = np.linalg.eigh(Ss)
        Ssi = (V / np.clip(w, 1e-8, None)) @ V.T
        A = Ssi @ sqrtm_psd(Ss @ Cd @ Ss) @ Ssi
        return ms, md, A, Cs, Cd

    mg, mr, A, Cg, Cr = ot(G, R)
    sc = float(np.sqrt(np.trace(Cr) / np.trace(Cg)))
    perm = rng.permutation(len(R)); h = len(R) // 2
    m1, m2, A0, _, _ = ot(R[perm[:h]], R[perm[h:]])
    return {"l0": (mg, mr, A), "iso": (mg, mr, sc * np.eye(len(mg))), "null": (m1, m2, A0), "scale_iso": sc}


@torch.no_grad()
def apply(a):
    from p3_latent_gen import fid_between, quint, strata, test_images, denorm
    from paper2_train import to_pm1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = argparse.Namespace(**vars(a)); base.seed = 0; base.out_root = a.base_root
    z, norm = load_latents(base, "sample")
    enc, dec, vargs = load_vae(argparse.Namespace(runs=a.runs, dataset=a.dataset, tag=a.tag, which=a.which), dev)
    ZE = np.load(os.path.join(a.dec_root, f"{a.dataset}_{a.tag}{sfx(a)}", "gen_latents_eval.npy")).astype(np.float32)
    zn = norm(ZE)
    if a.arm != "base":
        maps = fit_maps(a, z, norm)
        m_src, m_dst, A = maps[a.arm]
        f = zn.flatten(1).double().numpy()
        zn = torch.from_numpy(((f - m_src) @ A.T + m_dst).astype(np.float32)).view_as(zn)
        if a.arm == "iso":
            print(f"iso scale s = {maps['scale_iso']:.4f}")
    lat = denorm(zn.to(dev), z, dev).cpu()
    fake = decode_all(dec, lat, dev)
    d = os.path.join(a.out, a.arm, f"{a.dataset}_{a.tag}{sfx(a)}"); os.makedirs(d, exist_ok=True)
    real = test_images(base, vargs)
    rec = torch.cat([to_01(dec(enc(to_pm1(real[s:s + 250].to(dev)))[0])).cpu() for s in range(0, len(real), 250)])
    from torchvision.utils import save_image
    save_image(fake[:64], os.path.join(d, "samples.png"), nrow=8, padding=1)
    np.savez_compressed(os.path.join(d, "gen_images.npz"), x=(fake * 255).round().clamp(0, 255).to(torch.uint8).numpy())
    res = {"tag": a.tag, "dataset": a.dataset, "arm": a.arm, "gfid": fid_between(real, fake, dev),
           "rfid": fid_between(real, rec, dev), **strata(real, fake, dev)}
    json.dump(res, open(os.path.join(d, "gen_eval.json"), "w"), indent=1)
    print(f"arm {a.arm}: gFID {res['gfid']:.2f}, rFID {res['rfid']:.2f}, Q5 share (h) {100 * res['Q5']['share_gen']:.1f}%")
    q = argparse.Namespace(**vars(base)); q.out_root = os.path.join(a.out, a.arm); q.seed = a.seed
    quint(q, fake=fake, real=real)


def table(a):
    st = f"{a.dataset}_{a.tag}{sfx(a)}"
    rows, nan = {}, float("nan")
    for arm in ("base", "l0", "iso", "null"):
        d = os.path.join(a.out, arm, st)
        if not os.path.exists(os.path.join(d, "gen_eval.json")):
            continue
        G = json.load(open(os.path.join(d, "gen_eval.json")))
        Q = json.load(open(os.path.join(d, "quint_eval.json"))) if os.path.exists(os.path.join(d, "quint_eval.json")) else {}
        H = json.load(open(os.path.join(d, "h_check.json"))) if os.path.exists(os.path.join(d, "h_check.json")) else {}
        rows[arm] = dict(gfid=G["gfid"], rfid=G["rfid"], q5h=100 * G["Q5"]["share_gen"], q5a=H.get("q5_share_abs_hf", nan),
                         lowe=H.get("low_energy_in_Q4Q5_pct", nan), kid=Q.get("kid_all", nan), kidq5=Q.get("Q5", {}).get("kid", nan))
    D = json.load(open(os.path.join(a.out, st, "diag.json")))
    ceil = D["ceilings"]["recon_sample"]["S5_absHF"]
    print(f"{st}   (ceiling: decoded posterior samples, S5 abs HF = {ceil:.1f})")
    print(f"{'arm':>5s} {'gFID':>6s} {'rFID':>6s} {'Q5 h%':>6s} {'Q5 absHF%':>9s} {'lowE Q45%':>9s} {'KID':>6s} {'KID Q5':>7s}")
    for k, r in rows.items():
        print(f"{k:>5s} {r['gfid']:6.2f} {r['rfid']:6.2f} {r['q5h']:6.1f} {r['q5a']:9.1f} {r['lowe']:9.1f} {r['kid']:6.2f} {r['kidq5']:7.2f}")
    kd = os.path.join(a.out, f"kid_diff_{st}.json")
    if os.path.exists(kd):
        K = json.load(open(kd))["arms"]
        for arm, v in K.items():
            print(f"  KID diff vs base, {arm}: {v['all'][0]:.2f} [{v['all'][1]:.2f}, {v['all'][2]:.2f}]   "
                  f"Q5: {v['q5'][0]:.2f} [{v['q5'][1]:.2f}, {v['q5'][2]:.2f}]")
    if not all(k in rows for k in ("base", "l0", "iso", "null")):
        print("run `apply` for base, l0, iso and null (and p3_h_check.py on each) before the pass test"); return
    b, l, i, n = rows["base"], rows["l0"], rows["iso"], rows["null"]
    need = b["q5a"] + 0.5 * (ceil - b["q5a"])
    gain = l["q5a"] - b["q5a"]
    checks = {f"S5 abs HF >= {need:.1f} (half of the gap to the sample ceiling)": l["q5a"] >= need,
              f"LowE45 <= base ({b['lowe']:.1f})": l["lowe"] <= b["lowe"],
              "KID <= base": l["kid"] <= b["kid"], "KID in Q5 <= base": l["kidq5"] <= b["kidq5"],
              f"iso control gains < half of l0's gain ({i['q5a'] - b['q5a']:.1f} vs {gain:.1f})": (i["q5a"] - b["q5a"]) < 0.5 * gain,
              f"null control moves S5 abs HF by < 1.0 ({n['q5a'] - b['q5a']:+.1f})": abs(n["q5a"] - b["q5a"]) < 1.0}
    print("\npass test for l0 (written 12 Nov 2026, before any l0 run):")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'fail'}  {k}")
    print("  => " + ("ALL PASS" if all(checks.values()) else "NOT PASSED"))
    checks = {f"S5 abs HF >= {need:.1f} (half of the gap to the sample ceiling)": i["q5a"] >= need,
              f"LowE45 <= base ({b['lowe']:.1f})": i["lowe"] <= b["lowe"],
              "KID <= base": i["kid"] <= b["kid"], "KID in Q5 <= base": i["kidq5"] <= b["kidq5"],
              f"null control moves S5 abs HF by < 1.0 ({n['q5a'] - b['q5a']:+.1f})": abs(n["q5a"] - b["q5a"]) < 1.0}
    conf = a.dataset != "cifar10"
    print(f"\npass test for iso as a method (written 12 Nov 2026 before the CelebA runs)"
          + ("" if conf else " -- NOT a test here: iso was chosen after seeing this CIFAR-10 run") + ":")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'fail'}  {k}")
    print("  => " + ("ALL PASS" if all(checks.values()) else "NOT PASSED") + ("" if conf else " (post hoc, not counted)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag", "apply", "table"])
    ap.add_argument("--arm", default="base", choices=["base", "l0", "iso", "null"])
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--base_root", default="runs_p3"); ap.add_argument("--dec_root", default="runs_p3_dec")
    ap.add_argument("--out", default="runs_p3_lt"); ap.add_argument("--n_eval", type=int, default=10000)
    ap.add_argument("--n_gen_eval", type=int, default=10000)
    ap.add_argument("--q_n", type=int, default=300); ap.add_argument("--q_rep", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"diag": diag, "apply": apply, "table": table}[a.cmd](a)
