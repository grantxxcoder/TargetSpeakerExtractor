# How to change the data

All numbers live in **`experiments/configs/generator.yaml`**. Nothing is hardcoded
in the notebook or the trainer. Edit that file, rerun one command.

A full 20,000-trial rebuild takes **~58 s**. Rebuilds are cheap — no audio is stored.

---

## The command

```bash
source ../tse_venv/bin/activate          # from the repo root
python scripts/build_manifest.py --split train
```

One split per run. Valid splits: `train`, `val`, `eval_public`, `eval_private`,
`smoke_train`, `smoke_val`.

Rebuild everything:

```bash
for s in train val eval_public eval_private smoke_train smoke_val; do
  python scripts/build_manifest.py --split $s
done
```

Output goes to `data/manifests/<split>.csv`, plus a `.meta.yaml` recording the seed,
the config MD5 and the git commit. Same config + same seed = byte-identical CSV.

---

## Recipe 1 — make one parameter easier or harder

Edit the range under `defaults:`. `[low, high]` is sampled uniformly per trial; a
single value is fixed.

```yaml
defaults:
  sir_db: [-5.0, 15.0]     # before
  sir_db: [0.0, 12.0]      # after: interferer never louder than the target
```

Rebuild the splits you care about. **Which parameter to touch first, and what it
costs in realism: `difficulty-dial.md` §3.**

## Recipe 2 — change one split only

Add the key under that split. Split values override `defaults:`.

```yaml
splits:
  train: {n_trials: 20000, noise_split: tr, target_absent_fraction: 0.35,
          sir_db: [0.0, 12.0]}
```

## Recipe 3 — change how many trials

```yaml
splits:
  train: {n_trials: 5000, ...}
```

Rows come out in a fixed order, so the first 5,000 of a 20,000 build are the same
5,000 trials. Shrinking for a quick test does not reshuffle anything.

---

## After any change, check three things

```bash
jupyter lab src/exploratory/data_setup.ipynb
```

1. **§4** — the distributions look like what you asked for.
2. **§7** — the leak scoreboard. AUCs must not have gone up.
3. The `n_failed` count in `<split>.meta.yaml`. A jump means your new range is
   fighting a constraint and trials are being rejected — that bends the
   distribution and is how shortcuts appear (§4, §7).

---

## Rules

- **Never hand-edit a manifest CSV.** It is generated. Change the config and rebuild.
- **Never change `seed:` to get a different draw.** Change the parameter you actually
  mean. The seed is logged as the reproducibility record.
- **Log it.** Any rebuild feeding a result goes in `experiments/results/` with the
  config, the git commit and the date.
- **Eval splits are speaker-disjoint from train.** Do not change `paths.splits`.

---

## Once B12 lands (not yet implemented)

One regime is drawn per trial and recorded in a new `regime` column. Parameters not
named under a regime fall back to `defaults:`, so `hard` is just the wide ranges.

```yaml
regimes:
  base: {weight: 0.6, sir_db: [0.0, 12.0], snr_db: [8.0, 20.0]}
  hard: {weight: 0.4}                        # inherits defaults
```

Six parameters are regime-scoped: `sir_db`, `snr_db`, `overlap_ratio`, `t60_s`,
`source_distance_m`, `target_activity_ratio`. Everything else is global.

### Recipe 4 — the data is too hard

**Filter the manifest, do not rebuild.**

```python
df = pd.read_csv("data/manifests/train.csv")
df = df[df.regime == "base"]        # keeps ~60% of trials
```

This is the whole point of the column. Only changing the *band values* needs a
rebuild.

### Two cautions

- **`regime` is not a reporting stratum.** `base` is a sub-range of `hard`, so a
  `hard` trial is usually not extreme — it lands inside the base band 50–71 % of the
  time depending on the parameter. Report on value bands (SIR band, overlap band,
  T60 above/below the latency budget) instead. `decisions.md` 2026-08-13.
- **Re-run the §7 leak audit within each regime**, not just pooled. A shortcut inside
  `base` can be diluted to invisibility in a combined AUC.

`regime` is renderer metadata. The model receives the mixture audio and the
enrollment, nothing else.

---

See also: `data-construction-parameters.md` (what each parameter means),
`difficulty-dial.md` (which to change first).
