"""AV SFT: next-token loss on explainer explanations with the activation injected at <gh_stars>.
Signal check from the repo: eval loss with the real vector vs a shuffled vector (gap > 0 = uses it)."""
import argparse, json, math, os, time
import numpy as np, torch
from core import DEV, AV, load_split, save_av, cosine_lr

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--mb", type=int, default=32, help="microbatch (grad accumulation; same update as --bs)")
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--inj-scale", type=float, default=0, help="0 = round number a bit above mean vector norm (repo rule)")
p.add_argument("--eval-every", type=int, default=300)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-eval", type=int, default=1000)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
log = open(f"{a.out}/log.jsonl", "a")
def emit(**k):
    print(json.dumps(k), flush=True); log.write(json.dumps(k) + "\n"); log.flush()

_, texts, V = load_split("av")
_, ev_texts, evV = load_split("eval")
ev_texts, evV = ev_texts[:a.n_eval], evV[:a.n_eval]
if not a.inj_scale:
    mn = V.norm(dim=-1).mean().item()
    mag = 10 ** math.floor(math.log10(mn))
    a.inj_scale = math.ceil(mn * 1.1 / mag) * mag
av = AV(inj_scale=a.inj_scale).to(DEV)
emit(event="data", n_train=len(texts), n_eval=len(ev_texts), inj_scale=a.inj_scale,
     mean_norm=V.norm(dim=-1).mean().item(), prompt_len=len(av.prompt))
opt = torch.optim.AdamW(av.model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
steps = int(a.epochs * len(texts) / a.bs)
shuf = torch.randperm(len(evV), generator=torch.Generator().manual_seed(1))

@torch.no_grad()
def evaluate():
    av.eval(); r = {}
    for name, vv in (("real", evV), ("shuf", evV[shuf])):
        tot = cnt = 0.0
        for i in range(0, len(ev_texts), 128):
            with torch.autocast(DEV, dtype=torch.bfloat16):
                l, n = av.sft_loss(ev_texts[i:i + 128], vv[i:i + 128], reduce=False)
            tot += l.sum().item(); cnt += n.sum().item()
        r[name] = tot / cnt
    r["gap"] = r["shuf"] - r["real"]
    av.train(); return r

t0 = time.time(); step = 0; perm = np.random.permutation(len(texts)); ptr = 0
emit(step=0, eval=evaluate())
av.train()
while step < steps:
    if ptr + a.bs > len(perm):
        perm = np.random.permutation(len(texts)); ptr = 0
    idx = perm[ptr:ptr + a.bs]; ptr += a.bs
    lr = cosine_lr(step, steps, a.lr, a.warmup)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True); loss = 0.0
    for s in range(0, len(idx), a.mb):
        mi = idx[s:s + a.mb]
        with torch.autocast(DEV, dtype=torch.bfloat16):
            l = av.sft_loss([texts[i] for i in mi], V[mi]) * len(mi) / len(idx)
        l.backward(); loss += l.detach()
    gn = torch.nn.utils.clip_grad_norm_(av.model.parameters(), 1.0)
    opt.step(); step += 1
    if step % 25 == 0:
        emit(step=step, loss=loss.item(), lr=lr, gn=gn.item(), t=round(time.time() - t0))
    if step % a.eval_every == 0 or step == steps:
        emit(step=step, eval=evaluate())
save_av(av, a.out, {"args": vars(a)})
emit(event="done", t=round(time.time() - t0))
