# Latent Diffusion Under-Generates Complex Images

Code for the paper **"Latent Diffusion Under-Generates Complex Images: A Stratified Analysis and a Decoder-Side
Correction"** by Praful Sonarkar, Pooja Jain, Richa Makhijani and Gaurav Sharma (submitted to *Image and Vision
Computing*, 2026).

The code covers four things:
- the complexity-stratified evaluation protocol (quintiles of spectral complexity, absolute high-frequency energy, the
  low-energy test, and the tokenizer/generator gap);
- the flow-matching DiT trained on the latents of four frozen autoencoders;
- the adversarially anchored spectral decoder alignment (`b_gan`);
- the latent-side baselines (`iso`, `L0`).

The pass tests for each fix are written, with their dates, in the docstrings of the scripts. Each one was committed
before the runs it judges.

## Files

| File | What it does |
|---|---|
| `paper2_train.py` | Autoencoder definitions and training (A, B, G, D; from the companion paper on decoder capacity vs. training loss). The other scripts import the models from here |
| `p3_latent_gen.py` | `encode` latents, `train` the DiT, `sample`, per-quintile FID/KID (`quint`), `recon` (tokenizer control), `ifid`, `summary` |
| `p3_h_check.py` | Robustness of the strata: absolute HF energy, low-energy share, KID(Q5) − KID(Q1) with paired subsets |
| `p3_decoder_align.py` | Decoder alignment: `gen` latents, `train` arms (a/b/c, `--anchor gan` for iteration 2), `eval`, `table` (pass test) |
| `p3_kid_diff.py` | Paired-subset KID difference between a decoder arm and the unchanged decoder, with a 95 % range |
| `p3_latent_transport.py` | Latent-side baselines: `diag` (variance ratio per eigen-decile), `apply` (`base`, `iso`, `l0`, `null`), `table` |
| `p3_minority.py` | Minority score of Um et al. adapted to the latent flow model, and its relation to complexity (Table 3) |
| `p3_fix.py` | Generator-side arms: classifier-free guidance on the complexity quintile or on the minority-score quintile, and oversampling of complex training images (Table 4) |
| `make_p3_figs.py` | Figures of the paper from the result files |
| `spectral_rd_vae_test.py` | helper functions used by nested_vae_spectral.py.|
## Requirements

Python 3.12, PyTorch with CUDA (all results were obtained on one 8 GB Quadro RTX 4000), and:

```
pip install torch torchvision numpy scipy matplotlib lpips torchmetrics torch-fidelity
```

## Data

CIFAR-10 and CelebA at 64×64, with the same folders and splits as the companion autoencoder paper (see
`paper2_train.py`). Evaluation always uses the 10,000 real test images. The datasets are not redistributed here.

## Pipeline

Run everything from the repository folder. Replace `<root>` with your data folder. Use
`--dataset celeba --tag D_strong_gan` for CelebA.

```
# 1. autoencoder (repeat for each of A, B, G, D with its own --tag and options; see the docstring)
python paper2_train.py train --dataset cifar10 --root <root> --tag D_strong_gan_v2

# 2. latents, DiT, samples and strata
python p3_latent_gen.py encode --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_latent_gen.py train  --dataset cifar10 --root <root> --tag D_strong_gan_v2 --steps 50000 --amp
python p3_latent_gen.py sample --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_latent_gen.py recon  --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_latent_gen.py ifid   --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_h_check.py --root <root> --tags D_strong_gan_v2
#    second DiT seed: add --seed 1 to train and sample

# 3. decoder alignment (iteration 2 = adversarial anchor)
python p3_decoder_align.py gen   --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_decoder_align.py train --dataset cifar10 --root <root> --tag D_strong_gan_v2 --arm b --anchor gan
python p3_decoder_align.py eval  --dataset cifar10 --root <root> --tag D_strong_gan_v2 --arm b --anchor gan
python p3_kid_diff.py --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_decoder_align.py table --dataset cifar10 --root <root> --tag D_strong_gan_v2

# 4. minority score and generator-side arms: see the docstrings of p3_minority.py and p3_fix.py for the commands

# 5. latent-side baselines (decoder unchanged)
python p3_latent_transport.py diag  --dataset cifar10 --root <root> --tag D_strong_gan_v2
python p3_latent_transport.py apply --dataset cifar10 --root <root> --tag D_strong_gan_v2 --arm iso
python p3_latent_transport.py table --dataset cifar10 --root <root> --tag D_strong_gan_v2
```

Every script has a `--smoke` option that runs the whole chain on a few images in minutes. The numbers from a smoke run
mean nothing; it only checks the code. Long training runs resume when the same command is run again.

Outputs go to `runs_p2/`, `runs_p3/`, `runs_p3_dec/` and `runs_p3_lt/`, one folder per dataset and tag (and seed).
Model checkpoints are not included in this repository.

<!-- TODO (Praful): check every command above against the exact flags you used, especially --arm/--anchor names
     and the A/B/G/D options of paper2_train.py, before making the repository public. -->

## Citation

<!-- TODO: replace with the journal reference after acceptance -->
```
P. Sonarkar, P. Jain, R. Makhijani, G. Sharma, Latent diffusion under-generates complex images: a stratified
analysis and a decoder-side correction, submitted to Image and Vision Computing (2026).
```

## License

<!-- TODO (Praful): choose a license with your supervisors (for example MIT) and add a LICENSE file. -->

## Contact

Praful Sonarkar, Indian Institute of Information Technology Nagpur: dtvi22cse001@iiitn.ac.in
