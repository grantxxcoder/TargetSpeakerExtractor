# Data licences and attribution obligations

**Written 2026-08-10.** Decision rationale is in `docs/decisions/decisions.md`
(2026-08-10, "Data licensing"). This file is the operational checklist:
what each licence actually requires you to *do*.

Not legal advice. If a real redistribution or commercial question arises,
ask the university's research office.

## The three corpora

| Corpus | Licence | Attribution required | Commercial use | Share-alike |
|---|---|---|---|---|
| LibriSpeech | CC BY 4.0 | yes | yes | no |
| AMI Meeting Corpus | CC BY 4.0 | yes | yes | no |
| WHAM! noise | CC BY-NC 4.0 | yes | **no** | no |

WHAM! is the binding constraint. Every **constructed** mixture contains
WHAM! noise, so the LibriSpeech+WHAM! train/val/eval sets are NC.

**The AMI secondary eval set is not NC.** It contains no WHAM! noise — it is
real meeting audio only — so those trials remain CC BY 4.0 and may be
published under permissive terms. Do not apply the NC blanket to the AMI leg
by reflex; it is a separate release decision.

Neither licence is ShareAlike, but that is not why our *code* is
unaffected. Code is unaffected because it is not an adaptation of the audio.
Even CC BY-SA would not reach it.

## What you must actually do

### 1. Attribution block — put this in the thesis and the README

> This work uses the LibriSpeech corpus (Panayotov et al., 2015), licensed
> CC BY 4.0; the WHAM! noise corpus (Wichern et al., 2019), licensed
> CC BY-NC 4.0; and the AMI Meeting Corpus (Carletta et al., 2005),
> licensed CC BY 4.0. The constructed mixtures are derivative of
> LibriSpeech and WHAM! and are distributed, where distributed at all,
> under CC BY-NC 4.0. Trials derived from AMI contain no WHAM! material and
> remain under CC BY 4.0. All source audio was resampled, downmixed to mono
> and remixed for this work.

Attribution under CC BY/BY-NC means: name the work, name the creator, link
the licence, and indicate whether you changed it. We changed it (resampled,
downmixed, remixed), so say so.

### 2. Cite the papers properly — these are separate from the licence

- Panayotov, Chen, Povey & Khudanpur (2015). *LibriSpeech: an ASR corpus
  based on public domain audio books.* ICASSP.
- Wichern et al. (2019). *WHAM!: Extending Speech Separation to Noisy
  Environments.* Interspeech.
- Maciejewski, Wichern & Le Roux (2020). *WHAMR!: Noisy and Reverberant
  Single-Channel Speech Separation.* ICASSP. **Cite this even though we use
  no WHAMR! audio** — we generate RIRs to their published room-parameter
  distributions, which is a borrowed method and must be credited as one.
- Carletta et al. (2005). *The AMI Meeting Corpus: A Pre-announcement.*
  MLMI 2005; published in Springer LNCS 3869 (2006). For a fuller corpus
  description, prefer Carletta (2007), *Unleashing the killer corpus*,
  Language Resources and Evaluation 41(2):181–190.
- Cosentino, Pariente, Cornell, Deleforge & Vincent (2020). *LibriMix: An
  Open-Source Dataset for Generalizable Speech Separation.* arXiv:2005.11262.
  Never formally published — cite the arXiv entry. **Conditional:** only if
  you borrow its normalisation logic or manifest schema.

BibTeX for WHAM! and WHAMR! is on <http://wham.whisper.ai/>.

### 3. When you release the trial split

`docs/data/metric-definitions.md:198-200` requires publishing a public trial
split. Two ways to do it:

**Preferred — release code, not audio.** Publish the generator, the
manifests (source file IDs, offsets, gains, SNRs, RIR seeds) and the split
config. Users regenerate locally from corpora they obtain themselves. This
avoids redistribution entirely, is smaller, and is *better reproducibility*
than shipping WAVs.

**Fallback — release audio.** Then it must be CC BY-NC 4.0, with the
attribution block above included in the archive. Do not put it anywhere
that asserts a permissive licence by default.

### 4. What you may not do

- Sell, licence commercially, or embed the generated **constructed** audio
  in a product.
- Release constructed mixtures under CC BY, MIT, Apache or public domain.
- Treat "it's for research" as a blanket exemption. NC restricts the *use*,
  not the *user*; academic use is fine, but a spin-out later is not.

Not prohibited, but our house rule: **do not mirror the LibriSpeech or
WHAM! archives.** Both licences permit redistribution with attribution
(CC BY and CC BY-NC respectively), so this is a preference, not a legal
limit — link to the original sources so users always get the canonical
checksums.

### 5. Model weights

Probably unrestricted, but genuinely unsettled. A trained model is not
usually treated as a derivative work of its training data, and no
NC-licensed audio is recoverable from the weights. If weights are ever
released, state that training data included an NC corpus and leave the
downstream user to make their own call.

## Corpora we deliberately did not use, for licence reasons

- **wsj0 / wsj0-2mix** (and therefore the real WHAM!/WHAMR! mixtures) —
  requires a paid LDC licence we do not hold. This is why we use only the
  WHAM! *noise* archive, which is standalone, and generate our own RIRs.
  See `docs/data/data-setup.md` step 2.

## Checklist

- [ ] Attribution block in `README.md`
- [ ] Attribution block in the thesis, data chapter
- [ ] All five citations in the bibliography
- [ ] `LICENCE-DATA.md` or equivalent shipped with any released audio
- [ ] Release decision made: code+manifests (preferred) or audio (NC)
- [ ] WHAM! NC status noted wherever weights are published

## Sources

- LibriSpeech licence — <https://www.openslr.org/12>
- WHAM! licence — <http://wham.whisper.ai/>
- AMI licence — <https://groups.inf.ed.ac.uk/ami/corpus/license.shtml>
- CC BY-NC 4.0 deed — <https://creativecommons.org/licenses/by-nc/4.0/>
- CC BY 4.0 deed — <https://creativecommons.org/licenses/by/4.0/>
