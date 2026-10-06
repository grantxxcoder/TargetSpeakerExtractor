#!/usr/bin/env bash
# AB-SDR pair (decisions-m2.md 2026-10-03): stage epochs 3 and 5 of both runs,
# then render -> offline ASR -> judge (epoch-5 pair first), then the report's
# turbo ASR. One-off driver. Not committed.
#
# BEFORE RUNNING: unzip ALL-sir0-bsrnn_artif_ft-e6.zip and
# ALL-sir0-bsrnn_interf_artif_ft-e6.zip into their own folders under kaggle_out/
# (as with kaggle_out/arm_results and kaggle_out/control), and export
# GEMINI_API_KEY in this shell.
set -u
cd "$(dirname "$0")"
PY=../tse_venv/bin/python
LOG="eval-artif-$(date +%F).log"

find_run() {   # $1 = config stem -> the unzipped run folder that trained it
    local meta
    meta=$(grep -l "^config: experiments/configs/$1.yaml$" \
           kaggle_out/*/results/meta.yaml kaggle_out/*/*/results/meta.yaml 2>/dev/null | head -1)
    [ -n "$meta" ] && dirname "$(dirname "$meta")"
}

stage() {      # $1 config stem, $2 tag, $3 expected w_interf
    local run meta src dst e
    run=$(find_run "$1") || true
    [ -n "$run" ] || { echo "no unzipped run for $1 under kaggle_out/ -- unzip ALL-sir0-$1-e6.zip there first"; return 1; }
    meta="$run/results/meta.yaml"
    echo "== $1 <- $run"
    grep -q "^epochs_run: 6$" "$meta" || { echo "  did not run 6 epochs"; return 1; }
    grep "^git_commit:" "$meta" | sed 's/^/  /'
    grep "^git_commit:" "$meta" | grep -q dirty && { echo "  bundle stamped -dirty, stopping"; return 1; }
    for e in 3 5; do
        src="$run/models/model_sir0_e00$e.pt"
        dst="models/model_sir0_$2-e$e.pt"
        [ -f "$src" ] || { echo "  missing $src"; return 1; }
        # The checkpoint must carry the repo's config. Not config_md5: the
        # notebook re-dumps the file, so its hash never matches. It sets exactly
        # these three keys (checked on the 2026-10-02 pair); anything else differing
        # means a different run.
        $PY - "$src" "$e" "$3" "experiments/configs/$1.yaml" <<'EOF' || return 1
import sys, torch, yaml
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert c["epoch"] == int(sys.argv[2]), f"epoch {c['epoch']}, expected {sys.argv[2]}"
loss = c["config"]["loss"]
assert loss.get("w_artif") == 4.0 and loss.get("w_interf") == float(sys.argv[3]), loss
local, ck = flat(yaml.safe_load(open(sys.argv[4]))), flat(c["config"])
notebook = {"data.num_workers", "content_probe.data_root", "content_probe.manifest_dir"}
diff = {k: (local.get(k), ck.get(k)) for k in set(local) | set(ck)
        if k not in notebook and local.get(k) != ck.get(k)}
assert not diff, f"checkpoint config differs from {sys.argv[4]}: {diff}"
EOF
        [ -e "$dst" ] || cp "$src" "$dst"
        echo "  $dst  (md5 $(md5sum "$dst" | cut -c1-8))"
    done
}

main() {
    stage bsrnn_artif_ft        artif-ft        1.0 || exit 1
    stage bsrnn_interf_artif_ft interf-artif-ft 2.0 || exit 1
    [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY is not set -- export it, then rerun"; exit 1; }

    for r in "artif-ft-e5 bsrnn_artif_ft" "interf-artif-ft-e5 bsrnn_interf_artif_ft" \
             "artif-ft-e3 bsrnn_artif_ft" "interf-artif-ft-e3 bsrnn_interf_artif_ft"; do
        set -- $r
        $PY scripts/run_eval_suite.py --tag "$1" --checkpoint "models/model_sir0_$1.pt" \
            --config "experiments/configs/$2.yaml" --steps estimates,asr,judge \
            || { echo "FAILED on $1, stopping"; exit 1; }
    done

    echo "== judge done for all four. Turbo ASR (the report's offline ASR) next; Ctrl-C is safe here."
    for t in artif-ft-e5 interf-artif-ft-e5 artif-ft-e3 interf-artif-ft-e3; do
        EST=$(ls -d experiments/results/*-est-$t | tail -1)
        $PY scripts/evaluate.py --config experiments/configs/eval_offline_asr_turbo.yaml \
            --systems estimate --est "$EST" --out "experiments/results/$(date +%F)-eval-asr-turbo-$t" \
            || { echo "FAILED turbo on $t, stopping"; exit 1; }
    done
}

main 2>&1 | tee "$LOG"
