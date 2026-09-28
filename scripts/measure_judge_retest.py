import sys, csv, json, collections, statistics, random
from pathlib import Path
ROOT = Path("/home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor")
sys.path.insert(0, str(ROOT))
from src.live_model_metric.lcf_wer import count_errors
random.seed(20260922)

rows = list(csv.DictReader(open(ROOT/"experiments/results/judge_responses.csv")))
clips = collections.defaultdict(list)
for r in rows: clips[r["key"].rsplit("|r",1)[0]].append(r)
rep = {k:v for k,v in clips.items() if len(v) >= 3}

def meta_for(t):
    p = ROOT/"data/rendered"/t.rsplit("-",2)[0]/t/"meta.json"
    return json.loads(p.read_text()) if p.exists() else None

def wer(ref, hyp):
    c = count_errors(ref, hyp)
    if c.reference_word_count == 0: return None
    return 100.0*(c.substitutions+c.deletions+c.insertions)/c.reference_word_count

allclips, samedate = [], []
datespread = collections.Counter()
for k,v in sorted(rep.items()):
    m = meta_for(v[0]["trial_id"])
    if not m or not m.get("target_text","").strip(): continue
    tgt = m["target_text"]
    ws = [wer(tgt, r["text"]) for r in v]
    allclips.append(ws)
    datespread[len({r["run_date"] for r in v})] += 1
    bydate = collections.defaultdict(list)
    for r in v: bydate[r["run_date"]].append(wer(tgt, r["text"]))
    best = max(bydate.values(), key=len)
    if len(best) >= 3: samedate.append(best)

print(f"clips: {len(allclips)}; distinct run_dates per clip: {dict(datespread)}")
print(f"clips with >=3 calls on ONE date: {len(samedate)}\n")

def report(name, groups, n=103):
    var = statistics.mean([statistics.pvariance(g) for g in groups])
    sem = (var/n)**0.5
    B=20000; sems=[]
    for _ in range(B):
        s=[groups[random.randrange(len(groups))] for _ in range(len(groups))]
        sems.append((statistics.mean([statistics.pvariance(g) for g in s])/n)**0.5)
    sems.sort()
    lo,hi = sems[int(.025*B)], sems[int(.975*B)]
    print(f"{name} (n_clips={len(groups)})")
    print(f"  pooled within-clip SD (per trial)      : {var**0.5:6.2f} pts")
    print(f"  SEM of an n=103 aggregate (judge only) : {sem:6.3f} pts   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  95% band on ONE system aggregate       : +/-{1.96*sem:5.2f} pts")
    print(f"  95% band on a PAIRED system difference : +/-{1.96*(2*var/n)**0.5:5.2f} pts")
    print(f"  project-state.md currently claims SEM  :  0.500 pts -> understated {sem/0.5:.1f}x\n")

report("ALL repeat calls", allclips)
report("SAME-DATE calls only (drift excluded)", samedate)
