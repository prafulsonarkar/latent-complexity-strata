"""
p3_fix.py -- Paper 3: can the generator's loss of complex images be reduced, and does it matter WHICH score is rebalanced?

Three fixes, each changing one thing against the baseline DiT of p3_latent_gen.py (same width/depth/steps/batch/sampler):
  oversample : unconditional DiT; training latents of training-h quintiles Q4 / Q5 drawn 1.5x / 2x as often
  cond_h     : DiT conditioned on the h-quintile of each training image (label dropout 10% -> classifier-free guidance);
               sampling draws the 5 labels equally (20% each), guidance w = 1 (plain conditional) and w = 2
  cond_ms    : the SAME conditional DiT, but the label is the quintile of the minority score of Um, Lee & Ye (ICLR 2024),
               computed with the baseline DiT at t = 0.7 (k = 3). Only the label source differs from cond_h.
               (Um et al. use classifier guidance; we use CFG so that cond_h and cond_ms share one mechanism.)
Pre-registered reading (fixed before any run):
  * a fix "reduces the generator loss" if the generated Q5 share moves towards the reconstruction Q5 share by at least
    half of the baseline generator gap on BOTH h and absolute HF, with gFID no more than 10% worse than the baseline;
  * cond_h is evaluated on h, the score it was trained on, so absolute HF and KID are the checks that it is not
    just gaming h (also watch the low-energy % from p3_h_check.py).

Commands (run from D:\\new_vae_research; needs the baseline run of p3_latent_gen.py for the tag: latents.npz, dit.pt):
  labels : h of every training image and minority score of every training latent           (~15 min, GPU)
  train  : --fix oversample | cond_h | cond_ms                                              (~3 h each, GPU)
  sample : --fix ... [--guidance 1,2]; writes gen_images.npz, gen_eval.json, quint_eval.json and strata_grid.png
  table  : one table over baseline and all fixes found for a dataset
Outputs: runs_p3_fix/<fix>[_w<g>]/<ds>_<tag>/ ; labels in runs_p3_fix/labels_<ds>_<tag>.npz
Then, for the decomposition and abs-HF checks, per fix folder:
  python p3_h_check.py --dataset celeba --root ... --tags D_strong_gan --out_root runs_p3_fix/cond_h_w1 --no_kid
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import p3_latent_gen as P
from p3_latent_gen import DiT, t_embed, load_vae, load_latents, denorm, datasets, test_images, fid_between, quint, strata
from paper2_train import to_01, hf_fraction

N_CLS, NULL = 5, 5
OVERSAMPLE_W = [1.0, 1.0, 1.0, 1.5, 2.0]


class CondDiT(DiT):
    """DiT with a class embedding added to the time embedding (adaLN conditioning); class NULL = unconditional."""
    def __init__(self, width=384, depth=12, heads=6):
        super().__init__(width=width, depth=depth, heads=heads)
        self.cls = nn.Embedding(N_CLS + 1, width)
        nn.init.normal_(self.cls.weight, std=0.02)

    def forward(self, x, t, y):
        B = x.shape[0]
        h = self.inp(x.flatten(2).transpose(1, 2)) + self.pos
        c = self.temb(t_embed(t)) + self.cls(y)
        for b in self.blocks:
            h = b(h, c)
        s, sh = self.ada_f(c)[:, None].chunk(2, -1)
        h = self.out(self.nf(h) * (1 + s) + sh)
        return h.transpose(1, 2).reshape(B, self.ch, self.n, self.n)


def base_ckpt(a, dev):
    return torch.load(os.path.join(a.base_root, f"{a.dataset}_{a.tag}", "dit.pt"), map_location=dev, weights_only=False)


def fix_dir(a, name):
    d = os.path.join(a.fix_root, name, f"{a.dataset}_{a.tag}" + (f"_seed{a.seed}" if a.seed else ""))
    os.makedirs(d, exist_ok=True)
    return d


def labels_path(a):
    os.makedirs(a.fix_root, exist_ok=True)
    return os.path.join(a.fix_root, f"labels_{a.dataset}_{a.tag}.npz")


def with_base(a):
    """Namespace that points p3_latent_gen helpers at the baseline folder (latents.npz lives there)."""
    b = argparse.Namespace(**vars(a)); b.out_root = a.base_root
    return b


# =========================================================================================== labels
@torch.no_grad()
def labels(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = with_base(a)
    enc, dec, vargs = load_vae(b, dev)
    tr, _ = datasets(b, vargs)
    h = torch.cat([hf_fraction(x) for x, *_ in DataLoader(tr, 500, shuffle=False, num_workers=a.workers)]).numpy()
    z, norm = load_latents(b, "sample")
    if len(h) != len(z["train_mu"]):
        raise SystemExit(f"{len(h)} training images but {len(z['train_mu'])} training latents; re-run encode")
    ck = base_ckpt(a, dev)
    model = P.make_dit(argparse.Namespace(**ck["args"])).to(dev).eval(); model.load_state_dict(ck["ema"])
    z, norm = load_latents(b, ck["args"]["latent"])
    import lpips
    from p3_minority import minority_scores
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    print(f"minority score of {len(h)} training latents (t = 0.7, k = 3) ...", flush=True)
    ms = minority_scores(norm(z["train_mu"]), model, dec, z, lp, dev, 0.7, 3, a.amp, seed=0)
    q = lambda v: np.digitize(v, np.quantile(v, [.2, .4, .6, .8]))
    from scipy.stats import spearmanr
    np.savez(labels_path(a), h=h, ms=ms, q_h=q(h), q_ms=q(ms))
    print(f"saved {labels_path(a)}; Spearman rho(h, minority) on the training set = {spearmanr(h, ms).correlation:.2f}; "
          f"quintile agreement {100 * (q(h) == q(ms)).mean():.1f}% (20% = chance)")


# =========================================================================================== train
def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    b = with_base(a)
    ckb = base_ckpt(a, dev); ba = ckb["args"]
    z, _ = load_latents(b, ba["latent"])
    MU = torch.from_numpy(z["train_mu"].astype(np.float32)).to(dev)
    LV = torch.from_numpy(z["train_lv"].astype(np.float32)).to(dev)
    m_, s_ = (torch.from_numpy(z[k]).to(dev)[None, :, None, None] for k in ("ch_mean", "ch_std"))
    L = np.load(labels_path(a))
    cond = a.fix in ("cond_h", "cond_ms")
    Y = torch.from_numpy(L["q_h" if a.fix in ("cond_h", "oversample") else "q_ms"]).long().to(dev)
    w = torch.tensor(OVERSAMPLE_W, device=dev)[Y] if a.fix == "oversample" else torch.ones(len(Y), device=dev)
    mk = (lambda: CondDiT(ba["width"], ba["depth"], ba["heads"])) if cond else (lambda: DiT(width=ba["width"], depth=ba["depth"], heads=ba["heads"]))
    model, ema = mk().to(dev), mk().to(dev)
    ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=ba["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev == "cuda")
    path, step, log = os.path.join(fix_dir(a, a.fix), "dit.pt"), 0, []
    if os.path.exists(path):
        ck = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, log = ck["step"], ck["log"]; print(f"resumed at step {step}", flush=True)
    steps, bs = (300 if a.smoke else a.steps), ba["bs"]
    print(f"{a.fix}: {'conditional' if cond else 'unconditional'} DiT, {steps} steps of batch {bs}, same settings as the baseline", flush=True)
    ema_p, model_p = list(ema.parameters()), [p.detach() for p in model.parameters()]
    t0, acc = time.time(), 0.0
    model.train()
    while step < steps:
        i = torch.multinomial(w, bs, replacement=True)
        x0 = MU[i] + (torch.randn_like(MU[i]) * (0.5 * LV[i]).exp() if ba["latent"] == "sample" else 0)
        x0 = (x0 - m_) / s_
        eps, t = torch.randn_like(x0), torch.rand(bs, device=dev)
        xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * eps
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
            if cond:
                y = torch.where(torch.rand(bs, device=dev) < 0.1, torch.full_like(Y[i], NULL), Y[i])
                out = model(xt, t, y)
            else:
                out = model(xt, t)
            loss = ((out.float() - (eps - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            torch._foreach_lerp_(ema_p, model_p, 1 - min(ba["ema"], (1 + step) / (10 + step)))
        acc += loss.item(); step += 1
        le = 50 if a.smoke else 500
        if step % le == 0 or step == steps:
            rec = {"step": step, "loss": round(acc / le, 5), "it_s": round(le / (time.time() - t0), 1)}
            log.append(rec); print(json.dumps(rec), flush=True); acc, t0 = 0.0, time.time()
        if step % (100 if a.smoke else 5000) == 0 or step == steps:
            P.atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                           "log": log, "fix": a.fix, "base_args": ba}, path)
    print("saved", path)


# =========================================================================================== sample + evaluate
@torch.no_grad()
def sample(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = with_base(a)
    _, dec, vargs = load_vae(b, dev)
    ck = torch.load(os.path.join(fix_dir(a, a.fix), "dit.pt"), map_location=dev, weights_only=False)
    ba = ck["base_args"]
    z, _ = load_latents(b, ba["latent"])
    cond = a.fix in ("cond_h", "cond_ms")
    model = (CondDiT(ba["width"], ba["depth"], ba["heads"]) if cond else DiT(width=ba["width"], depth=ba["depth"], heads=ba["heads"])).to(dev).eval()
    model.load_state_dict(ck["ema"])
    n = 500 if a.smoke else a.n_gen
    real = test_images(b, vargs)
    gs = [float(g) for g in a.guidance.split(",")] if cond else [1.0]
    for g in gs:
        name = f"{a.fix}_w{g:g}" if cond else a.fix
        od = fix_dir(a, name)
        gen = torch.Generator(device=dev).manual_seed(1000003 * a.seed + 7)
        ylab = torch.arange(n, device=dev) % N_CLS
        ylab = ylab[torch.randperm(n, device=dev, generator=gen)]
        X, t0 = [], time.time()
        for s in range(0, n, a.gen_bs):
            m = min(a.gen_bs, n - s)
            x = torch.randn(m, 4, 8, 8, device=dev, generator=gen)
            y = ylab[s:s + m]
            ts = torch.linspace(1, 0, a.sample_steps + 1, device=dev)
            for k in range(a.sample_steps):
                t = ts[k].expand(m)
                with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
                    if not cond:
                        v = model(x, t)
                    elif g == 1.0:
                        v = model(x, t, y)
                    else:
                        vv = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, NULL)]))
                        vc, vu = vv.float().chunk(2)
                        v = vu + g * (vc - vu)
                x = x + (ts[k + 1] - ts[k]) * v.float()
            X.append(to_01(dec(denorm(x, z, dev))).cpu())
        fake = torch.cat(X)
        print(f"{name}: generated {len(fake)} images in {time.time() - t0:.0f}s", flush=True)
        from torchvision.utils import save_image
        save_image(fake[:64], os.path.join(od, "samples.png"), nrow=8, padding=1)
        np.savez_compressed(os.path.join(od, "gen_images.npz"), x=(fake * 255).round().clamp(0, 255).to(torch.uint8).numpy())
        res = {"tag": a.tag, "dataset": a.dataset, "fix": name, "seed": a.seed, "guidance": g,
               "gfid": fid_between(real, fake, dev), **strata(real, fake, dev)}
        if a.fix == "cond_h":      # does the generated image land in the h-quintile it was asked for?
            hq = np.digitize(hf_fraction(fake).numpy(), np.quantile(hf_fraction(real).numpy(), [.2, .4, .6, .8]))
            res["label_hit_pct"] = float(100 * (hq == ylab.cpu().numpy()).mean())
        json.dump(res, open(os.path.join(od, "gen_eval.json"), "w"), indent=1)
        print(f"{name}: gFID {res['gfid']:.2f}; Q5 share (h) {100 * res['Q5']['share_gen']:.1f}%"
              + (f"; generated image in its requested h-quintile {res['label_hit_pct']:.1f}% (20% = chance)" if a.fix == "cond_h" else ""))
        q = argparse.Namespace(**vars(b)); q.out_root = os.path.join(a.fix_root, name)
        quint(q, fake=fake, real=real)


# =========================================================================================== table
def table(a):
    import glob
    rows = []
    cand = [("baseline", os.path.join(a.base_root, f"{a.dataset}_{a.tag}"))] + \
           [(os.path.basename(os.path.dirname(d)), d) for d in sorted(glob.glob(os.path.join(a.fix_root, "*", f"{a.dataset}_{a.tag}")))]
    for name, d in cand:
        g = os.path.join(d, "gen_eval.json")
        if not os.path.exists(g):
            continue
        G = json.load(open(g))
        H = json.load(open(os.path.join(d, "h_check.json"))) if os.path.exists(os.path.join(d, "h_check.json")) else {}
        Q = json.load(open(os.path.join(d, "quint_eval.json"))) if os.path.exists(os.path.join(d, "quint_eval.json")) else {}
        nan = float("nan")
        rows.append((name, G["gfid"], 100 * G["Q5"]["share_gen"], H.get("q5_share_abs_hf", nan), H.get("gen_gap_q5_h", nan),
                     H.get("gen_gap_q5_abs_hf", nan), Q.get("kid_all", nan), Q.get("Q5", {}).get("kid", nan),
                     H.get("low_energy_in_Q4Q5_pct", nan), G.get("label_hit_pct", nan)))
    print(f"{a.dataset} {a.tag}")
    print(f"{'run':>14s} {'gFID':>6s} {'Q5 h%':>6s} {'Q5 absHF%':>9s} {'gen gap h':>9s} {'gen gap absHF':>13s} {'KID':>6s} {'KID Q5':>7s} {'lowE Q45%':>9s} {'label hit%':>10s}")
    for r in rows:
        print(f"{r[0]:>14s} {r[1]:6.1f} {r[2]:6.1f} {r[3]:9.1f} {r[4]:9.1f} {r[5]:13.1f} {r[6]:6.1f} {r[7]:7.1f} {r[8]:9.1f} {r[9]:10.1f}")
    print("gen gap = reconstruction Q5 share - generated Q5 share (points; smaller = the fix helps). gen gap / abs-HF columns need\n"
          "p3_h_check.py run on that folder (--out_root runs_p3_fix/<run>). label hit = generated image falls in its requested h-quintile.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "train", "sample", "table"])
    ap.add_argument("--dataset", required=True, choices=["cifar10", "celeba"]); ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--fix", choices=["oversample", "cond_h", "cond_ms"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--guidance", default="1,2"); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--runs", default="runs_p2"); ap.add_argument("--which", default="last")
    ap.add_argument("--base_root", default="runs_p3"); ap.add_argument("--fix_root", default="runs_p3_fix")
    ap.add_argument("--n_eval", type=int, default=10000); ap.add_argument("--n_gen", type=int, default=10000)
    ap.add_argument("--gen_bs", type=int, default=500); ap.add_argument("--sample_steps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--q_n", type=int, default=300); ap.add_argument("--q_rep", type=int, default=20)
    a = ap.parse_args()
    if a.cmd in ("train", "sample") and not a.fix:
        raise SystemExit("--fix is required for train and sample")
    {"labels": labels, "train": train, "sample": sample, "table": table}[a.cmd](a)
