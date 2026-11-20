"""
p3_decoder_align.py -- method pilot: can a decoder fine-tuned on GENERATED latents restore the fine detail that the
generator loses on complex images? (DiT frozen; only the VAE decoder changes.)

Arms (same steps, batch, learning rate; all keep a reconstruction anchor so the decoder still reconstructs):
  a : anchor only -- reconstruction of the real images whose latents are nearest to the generated latents
      (control for extra decoder training on the region of latent space the generator visits)
  b : a + match the mean log radial power spectrum of decoded generated images to that of real images (all images)
  c : a + the same spectrum match inside each complexity quintile (h quintile of the decoded image, cuts from real
      training images), quintiles weighted equally
  The two loss terms are each divided by their value at step 0, so neither dominates by scale (no tuned weight).
--anchor gan (iteration 2, added after iteration 1; report it as such): the anchor is the decoder's OWN Paper 2
  training loss (Gaussian NLL + LPIPS + adaptive-weight GAN with its saved critic, critic updated too) instead of
  plain MSE. Reason: in iteration 1 the MSE-only anchor pulled the adversarial decoder towards an MSE decoder (arm a:
  Q5 share 9.5% -> 3.8%, KID 7.46 -> 9.13), so arm a was not a neutral control. Arms are saved as a_gan, b_gan, c_gan
  and judged by the same pass test, with c_gan against base and against a_gan.

Confirmatory test for arm b_gan (written 2026-11-05, AFTER b_gan looked good on CelebA D DiT seed 0 and BEFORE the
  runs below; seed 0 b_gan is post hoc and is reported as such). Run on new settings only: CelebA D with DiT seed 1
  (new generated latents, new decoder seed) and CIFAR-10 D_strong_gan_v2. Thresholds are relative to that setting's base,
  so they carry over between datasets:
    1. Q5 abs-HF share closes at least half of the base generator gap: >= base + 0.5 * (recon - base)
    2. low-energy share of Q4/Q5 not above base          3. KID <= base          4. KID in Q5 <= base
    5. rFID <= 1.10 x base                                6. Q5 abs-HF share above arm a_gan
  Seeds: --seed k > 0 reads the DiT of runs_p3/<ds>_<tag>_seed<k>/, draws new generated latents, trains with
  torch seed k, and writes to <arm>/<ds>_<tag>_seed<k>/ (seed 0 paths unchanged).

Pass test (fixed before any run, from the Paper 3 doc): arm c reaches Q5 abs-HF share >= 10.3%, keeps the low-energy
share of Q4/Q5 at or below 18.9%, does not worsen KID or KID in Q5 against arm `base`, raises rFID by at most 10%,
and beats arm a on Q5 abs-HF share. (Q5 abs HF and low-energy come from p3_h_check.py run on each arm folder.)

Commands (run from D:\\new_vae_research; needs the baseline p3_latent_gen.py run of the tag):
  gen   : 50,000 training latents and 10,000 evaluation latents from the frozen baseline DiT (different seeds)  (~2 h)
  train : --arm a | b | c                                                                                       (~20 min each)
  eval  : --arm base | a | b | c   decode the 10,000 evaluation latents, gFID, rFID, quint (KID, grid)            (~5 min each)
  table : pass test over all arms
then per arm (CPU is fine):
  python p3_h_check.py --dataset celeba --root ... --tags D_strong_gan --out_root runs_p3_dec\\<arm> --no_kid
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import p3_latent_gen as P
from p3_latent_gen import load_vae, load_latents, denorm, datasets, test_images, fid_between, quint, strata
from paper2_train import to_pm1, to_01, hf_fraction

PASS = {"q5_abs_hf": 10.3, "low_e_q45": 18.9, "rfid_rel": 1.10}


def ns(a, root):
    b = argparse.Namespace(**vars(a)); b.out_root = root
    return b


def sfx(a):
    return f"_seed{a.seed}" if getattr(a, "seed", 0) else ""


def work_dir(a):
    d = os.path.join(a.dec_root, f"{a.dataset}_{a.tag}{sfx(a)}")
    os.makedirs(d, exist_ok=True)
    return d


def arm_name(a, arm):
    return arm + ("_gan" if getattr(a, "anchor", "mse") == "gan" and arm != "base" else "")


def arm_dir(a, arm):
    d = os.path.join(a.dec_root, arm_name(a, arm), f"{a.dataset}_{a.tag}{sfx(a)}")
    os.makedirs(d, exist_ok=True)
    return d


class RadialSpec:
    """Mean power in integer-radius frequency rings 1..R of the grey image, as a (B, R) tensor; differentiable."""
    def __init__(self, n, dev, R=None):
        f = torch.fft.fftfreq(n, device=dev) * n
        r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).round().long().flatten()
        R = R or n // 2
        M = torch.zeros(n * n, R, device=dev)
        for k in range(1, R + 1):
            m = r == k
            M[m, k - 1] = 1.0 / m.sum()
        self.M = M

    def __call__(self, x):
        g = x.mean(1)
        G = torch.fft.fft2(g - g.mean((1, 2), keepdim=True)).abs() ** 2
        return torch.log(G.flatten(1) @ self.M + 1e-8)


def quintiles(h, cuts):
    return torch.bucketize(h, cuts)


# =========================================================================================== gen
@torch.no_grad()
def gen(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = ns(a, a.base_root)
    ck = torch.load(os.path.join(a.base_root, f"{a.dataset}_{a.tag}{sfx(a)}", "dit.pt"), map_location=dev, weights_only=False)
    z, _ = load_latents(b, ck["args"]["latent"])
    model = P.make_dit(argparse.Namespace(**ck["args"])).to(dev).eval(); model.load_state_dict(ck["ema"])
    for name, n, seed in (("train", a.n_train_gen, 424242 + 1000003 * a.seed), ("eval", a.n_eval_gen, 777 + 1000003 * a.seed)):
        out, t0 = [], time.time()
        for s in range(0, n, a.gen_bs):
            m = min(a.gen_bs, n - s)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
                zn = P.euler(model, m, dev, a.sample_steps, seed + s)
            out.append(denorm(zn.float(), z, dev).cpu())
        np.save(os.path.join(work_dir(a), f"gen_latents_{name}.npy"), torch.cat(out).numpy().astype(np.float32))
        print(f"{name}: {n} generated latents in {time.time() - t0:.0f}s", flush=True)


# =========================================================================================== train
def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    b = ns(a, a.base_root); b.seed = 0          # latents and VAE are shared by all DiT seeds
    enc, dec, vargs = load_vae(b, dev)
    dec.train().requires_grad_(True)
    tr, _ = datasets(b, vargs)
    X = torch.cat([(x * 255).round().to(torch.uint8) for x, *_ in DataLoader(tr, 500, shuffle=False, num_workers=a.workers)])
    z, norm = load_latents(b, "sample")
    MU = torch.from_numpy(z["train_mu"].astype(np.float32))
    if len(MU) != len(X):
        raise SystemExit(f"{len(X)} training images but {len(MU)} latents; re-run p3_latent_gen.py encode")
    ZG = torch.from_numpy(np.load(os.path.join(work_dir(a), "gen_latents_train.npy")))
    # nearest real latent (normalised space) of every generated latent
    tn, gn = norm(MU.numpy()).flatten(1).to(dev), norm(ZG.numpy()).flatten(1).to(dev)
    NN = torch.cat([torch.cdist(gn[s:s + 1000], tn).argmin(1) for s in range(0, len(gn), 1000)]).cpu()
    del tn, gn
    spec = RadialSpec(X.shape[-1], dev)
    with torch.no_grad():                      # real targets: mean log spectrum, overall and per h quintile
        Sr, Hr = [], []
        for s in range(0, len(X), 1000):
            x = X[s:s + 1000].to(dev).float() / 255
            Sr.append(spec(x)); Hr.append(hf_fraction(x))
        Sr, Hr = torch.cat(Sr), torch.cat(Hr)
        cuts = torch.quantile(Hr, torch.tensor([.2, .4, .6, .8], device=dev))
        qr = quintiles(Hr, cuts)
        T_all, T_q = Sr.mean(0), torch.stack([Sr[qr == j].mean(0) for j in range(5)])
    opt = torch.optim.Adam(dec.parameters(), lr=a.lr)
    gan = a.anchor == "gan"
    if gan:
        if getattr(vargs, "no_gan", False):
            raise SystemExit("--anchor gan needs a decoder trained with a GAN (tags D_* or G_*)")
        from paper2_train import build
        from paper2_decoder_gan import AutoencoderGANLoss
        ck_vae = torch.load(os.path.join(a.runs, f"{a.dataset}_{a.tag}", f"{a.which}.pt"), map_location=dev, weights_only=False)
        Dc = build(vargs)[2].to(dev); Dc.load_state_dict(ck_vae["D"]); Dc.train()
        lossf = AutoencoderGANLoss(critic=vargs.critic, disc_start=0, w_lpips=vargs.w_lpips, w_gan=vargs.w_gan,
                                   beta_kl=0.0, use_lpips=True, n_pixels=3 * 64 * 64).to(dev)
        with torch.no_grad():
            lossf.log_sigma.copy_(ck_vae["log_sigma"].to(dev))
        lossf.log_sigma.requires_grad_(False)
        opt_d = torch.optim.Adam(Dc.parameters(), lr=a.lr, betas=(0.5, 0.9))
    steps = 50 if a.smoke else a.steps
    L0 = S0 = None
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        i = torch.randint(0, len(ZG), (a.bs,))
        xr = X[NN[i]].to(dev).float() / 255
        if gan:                                   # the decoder's own training loss, fp32 (adaptive GAN weight)
            xh = dec(MU[NN[i]].to(dev))
            l_rec, _ = lossf.generator_loss(to_pm1(xr), xh, xr.new_zeros(()), Dc, dec.last_layer, step)
        else:
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
                rec = (dec(MU[NN[i]].to(dev)).float() + 1) / 2
            l_rec = F.mse_loss(rec, xr)
        # The two terms are back-propagated one after the other (gradients add up, so the update is the same as for
        # l_rec/L0 + l_spec/S0); this frees the anchor graph before the second decoder pass and halves peak GPU memory.
        opt.zero_grad(set_to_none=True)
        l_spec = torch.zeros((), device=dev)
        if a.arm in ("b", "c"):
            if L0 is None:
                L0 = max(abs(l_rec.item()), 1e-6)            # abs: the GAN anchor (NLL) can be negative
            (l_rec / L0).backward()
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
                xg = ((dec(ZG[i].to(dev)).float() + 1) / 2).clamp(-0.5, 1.5)
            sg = spec(xg)
            if a.arm == "b":
                l_spec = (sg.mean(0) - T_all).abs().mean()
            else:
                qg = quintiles(hf_fraction(xg.detach().clamp(0, 1)), cuts)
                terms = [(sg[qg == j].mean(0) - T_q[j]).abs().mean() for j in range(5) if (qg == j).sum() >= 4]
                l_spec = torch.stack(terms).mean()
            if S0 is None:
                S0 = max(l_spec.item(), 1e-6)
            (l_spec / S0).backward()
            del xg, sg
        else:
            l_rec.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        if gan:
            d_loss, _ = lossf.critic_loss(to_pm1(xr), xh.detach(), Dc)
            opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
        if step % (10 if a.smoke else 100) == 0 or step == steps:
            rec_ = {"step": step, "rec": round(l_rec.item(), 6), "spec": round(l_spec.item(), 4), "s": round(time.time() - t0)}
            log.append(rec_); print(json.dumps(rec_), flush=True)
    d = arm_dir(a, a.arm)
    torch.save({"dec": dec.state_dict(), "arm": arm_name(a, a.arm), "anchor": a.anchor, "steps": steps, "lr": a.lr, "bs": a.bs, "log": log}, os.path.join(d, "decoder.pt"))
    print("saved", os.path.join(d, "decoder.pt"))


# =========================================================================================== eval
@torch.no_grad()
def evaluate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = ns(a, a.base_root); b.seed = 0
    enc, dec, vargs = load_vae(b, dev)
    d = arm_dir(a, a.arm)
    if a.arm != "base":
        dec.load_state_dict(torch.load(os.path.join(d, "decoder.pt"), map_location=dev, weights_only=False)["dec"])
    dec.eval()
    ZE = torch.from_numpy(np.load(os.path.join(work_dir(a), "gen_latents_eval.npy")))
    fake = torch.cat([to_01(dec(ZE[s:s + 250].to(dev))).cpu() for s in range(0, len(ZE), 250)])
    real = test_images(b, vargs)
    rec = torch.cat([to_01(dec(enc(to_pm1(real[s:s + 250].to(dev)))[0])).cpu() for s in range(0, len(real), 250)])
    from torchvision.utils import save_image
    save_image(fake[:64], os.path.join(d, "samples.png"), nrow=8, padding=1)
    np.savez_compressed(os.path.join(d, "gen_images.npz"), x=(fake * 255).round().clamp(0, 255).to(torch.uint8).numpy())
    res = {"tag": a.tag, "dataset": a.dataset, "arm": arm_name(a, a.arm), "gfid": fid_between(real, fake, dev),
           "rfid": fid_between(real, rec, dev), **strata(real, fake, dev)}
    json.dump(res, open(os.path.join(d, "gen_eval.json"), "w"), indent=1)
    print(f"arm {arm_name(a, a.arm)}: gFID {res['gfid']:.2f}, rFID {res['rfid']:.2f}, Q5 share (h) {100 * res['Q5']['share_gen']:.1f}%")
    q = ns(a, os.path.join(a.dec_root, arm_name(a, a.arm))); q.seed = a.seed
    quint(q, fake=fake, real=real)


# =========================================================================================== table
def table(a):
    rows = {}
    for arm in ("base", "a", "b", "c", "a_gan", "b_gan", "c_gan"):
        d = os.path.join(a.dec_root, arm, f"{a.dataset}_{a.tag}{sfx(a)}")
        if not os.path.exists(os.path.join(d, "gen_eval.json")):
            continue
        G = json.load(open(os.path.join(d, "gen_eval.json")))
        Q = json.load(open(os.path.join(d, "quint_eval.json"))) if os.path.exists(os.path.join(d, "quint_eval.json")) else {}
        H = json.load(open(os.path.join(d, "h_check.json"))) if os.path.exists(os.path.join(d, "h_check.json")) else {}
        nan = float("nan")
        rows[arm] = dict(gfid=G["gfid"], rfid=G["rfid"], q5h=100 * G["Q5"]["share_gen"], q5a=H.get("q5_share_abs_hf", nan),
                         lowe=H.get("low_energy_in_Q4Q5_pct", nan), rec_a=H.get("q5_recon_abs_hf", nan), kid=Q.get("kid_all", nan), kidq5=Q.get("Q5", {}).get("kid", nan))
    print(f"{a.dataset} {a.tag}" + (f" (DiT seed {a.seed})" if a.seed else ""))
    print(f"{'arm':>5s} {'gFID':>6s} {'rFID':>6s} {'Q5 h%':>6s} {'Q5 absHF%':>9s} {'lowE Q45%':>9s} {'KID':>6s} {'KID Q5':>7s}")
    for k, r in rows.items():
        print(f"{k:>5s} {r['gfid']:6.2f} {r['rfid']:6.2f} {r['q5h']:6.1f} {r['q5a']:9.1f} {r['lowe']:9.1f} {r['kid']:6.2f} {r['kidq5']:7.2f}")
    for cn, an in (("c", "a"), ("c_gan", "a_gan")):
        if cn not in rows or "base" not in rows:
            continue
        c, bse = rows[cn], rows["base"]
        checks = {f"Q5 abs HF >= {PASS['q5_abs_hf']}": c["q5a"] >= PASS["q5_abs_hf"],
                  f"low-energy Q4/Q5 <= {PASS['low_e_q45']}": c["lowe"] <= PASS["low_e_q45"],
                  "KID <= base": c["kid"] <= bse["kid"], "KID in Q5 <= base": c["kidq5"] <= bse["kidq5"],
                  "rFID <= 1.10 x base": c["rfid"] <= PASS["rfid_rel"] * bse["rfid"],
                  f"Q5 abs HF above arm {an}": an in rows and c["q5a"] > rows[an]["q5a"]}
        print(f"\npass test for arm {cn} (fixed before the runs):")
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'fail'}  {k}")
        print("  => " + ("ALL PASS" if all(checks.values()) else "NOT PASSED"))
    if "b_gan" in rows and "base" in rows:
        c, bse = rows["b_gan"], rows["base"]
        need = bse["q5a"] + 0.5 * (bse["rec_a"] - bse["q5a"])
        checks = {f"Q5 abs HF >= {need:.1f} (half of base generator gap closed)": c["q5a"] >= need,
                  f"low-energy Q4/Q5 <= base ({bse['lowe']:.1f})": c["lowe"] <= bse["lowe"],
                  "KID <= base": c["kid"] <= bse["kid"], "KID in Q5 <= base": c["kidq5"] <= bse["kidq5"],
                  "rFID <= 1.10 x base": c["rfid"] <= PASS["rfid_rel"] * bse["rfid"],
                  "Q5 abs HF above arm a_gan": "a_gan" in rows and c["q5a"] > rows["a_gan"]["q5a"]}
        conf = not (a.dataset == "celeba" and a.tag == "D_strong_gan" and a.seed == 0)
        print(f"\n{'CONFIRMATORY' if conf else 'POST-HOC (this is the run b_gan was chosen on)'} test for arm b_gan"
              " (written 2026-11-05):")
        for k, v in checks.items():
            print(f"  {'PASS' if v else 'fail'}  {k}")
        print("  => " + ("ALL PASS" if all(checks.values()) else "NOT PASSED"))
    print("(Q5 abs HF and low-energy columns need p3_h_check.py on runs_p3_dec\\<arm>)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "train", "eval", "table"])
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--arm", choices=["base", "a", "b", "c"])
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--base_root", default="runs_p3"); ap.add_argument("--dec_root", default="runs_p3_dec")
    ap.add_argument("--n_train_gen", type=int, default=50000); ap.add_argument("--n_eval_gen", type=int, default=10000)
    ap.add_argument("--gen_bs", type=int, default=500); ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--n_eval", type=int, default=10000); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor", default="mse", choices=["mse", "gan"], help="gan = iteration 2 (see docstring)")
    ap.add_argument("--q_n", type=int, default=300); ap.add_argument("--q_rep", type=int, default=20)
    a = ap.parse_args()
    if a.cmd in ("train", "eval") and not a.arm:
        raise SystemExit("--arm is required")
    if a.cmd == "train" and a.arm == "base":
        raise SystemExit("arm base is the untouched decoder; nothing to train")
    {"gen": gen, "train": train, "eval": evaluate, "table": table}[a.cmd](a)
