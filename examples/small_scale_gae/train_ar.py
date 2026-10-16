"""AR SFT: explanation text -> activation vector. Arms: final | multi | attn (see core.py)."""
import argparse, hashlib, json, math, os, time
import numpy as np, torch
from core import DEV, AR, load_split, rawvar_baseline, mse_nrm, save_ar, cosine_lr, CUT_EVERY, DATA
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["final", "multi", "attn"], required=True)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--new-lr-mult", type=float, default=10.0, help="LR multiplier for the fresh readout layer (attn arm)")
p.add_argument("--epochs", type=float, default=4)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--eval-every", type=int, default=200)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--cap-eval", type=int, default=0, help="smoke tests: cap val/eval sizes")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
log = open(f"{a.out}/log.jsonl", "a")
def emit(**k):
    print(json.dumps(k), flush=True); log.write(json.dumps(k) + "\n"); log.flush()

rows, texts, V = load_split("ar")
meta = pq.read_table(f"{DATA}/base.parquet", columns=["doc_id"]).to_pandas()
isval = np.array([int(hashlib.sha256(f"val|{meta.doc_id[r]}".encode()).hexdigest()[:8], 16) % 20 == 0 for r in rows])
tr = np.nonzero(~isval)[0]; va = np.nonzero(isval)[0][:2000]
_, ev_texts, evV = load_split("eval")
if a.cap_eval:
    va = va[:a.cap_eval]; ev_texts, evV = ev_texts[:a.cap_eval], evV[:a.cap_eval]
base = rawvar_baseline(V)  # whole AR split, same baseline as rl.py
emit(event="data", n_train=len(tr), n_val=len(va), n_eval=len(ev_texts), rawvar=base)

ar = AR("attn" if a.arm == "attn" else "final").to(DEV)
mode = "every" if a.arm == "multi" else "full"
newn = ar.new_param_names()
groups = [{"params": [p_ for n, p_ in ar.named_parameters() if n not in newn], "mult": 1.0},
          {"params": [p_ for n, p_ in ar.named_parameters() if n in newn], "mult": a.new_lr_mult}]
opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
steps = int(a.epochs * len(tr) / a.bs)

@torch.no_grad()
def evaluate(txts, gold, cuts=False, bs=128):
    ar.eval()
    full, by_cut = [], {}
    for i in range(0, len(txts), bs):
        with torch.autocast(DEV, dtype=torch.bfloat16):
            preds, cl = ar(txts[i:i + bs], "every" if cuts else "full")
        g = gold[i:i + bs].to(DEV)
        for b, (pr, cs) in enumerate(zip(preds, cl)):
            m = mse_nrm(pr.float(), g[b:b + 1].expand(len(cs), -1))
            full.append(m[-1].item())
            if cuts:
                for c, x in zip(cs[:-1], m[:-1].tolist()):
                    by_cut.setdefault(c, []).append(x)
    ar.train()
    res = {"fve": 1 - np.mean(full) / base, "mse": float(np.mean(full))}
    if cuts:
        res["fve_by_cut"] = {int(c): [1 - np.mean(v) / base, len(v)] for c, v in sorted(by_cut.items())}
    return res

t0 = time.time(); step = 0; perm = np.random.permutation(tr); ptr = 0
ar.train()
while step < steps:
    if ptr + a.bs > len(perm):
        perm = np.random.permutation(tr); ptr = 0
    idx = perm[ptr:ptr + a.bs]; ptr += a.bs
    lr = cosine_lr(step, steps, a.lr, a.warmup)
    for g in opt.param_groups:
        g["lr"] = lr * g["mult"]
    with torch.autocast(DEV, dtype=torch.bfloat16):
        loss, _, _ = ar.loss([texts[i] for i in idx], V[idx].to(DEV), mode)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
    opt.step(); step += 1
    if step % 25 == 0:
        emit(step=step, loss=loss.item(), lr=lr, gn=gn.item(), t=round(time.time() - t0))
    if step % a.eval_every == 0 or step == steps:
        emit(step=step, val=evaluate([texts[i] for i in va], V[va]))
save_ar(ar, a.out, {"arm": a.arm, "args": vars(a)})
emit(event="final_eval", eval=evaluate(ev_texts, evV, cuts=True), t=round(time.time() - t0))
