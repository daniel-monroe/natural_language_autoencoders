"""RL: GRPO on the AV with reward -MSE_nrm(AR(explanation), gold); AR trained online (MSE) on the
same rollouts. Mirrors configs/rl.sh semantics: n=8 samples/prompt, group-normalised advantages,
k2 KL-to-SFT loss (coef 0.01), failed extraction / truncated generation -> reward -2, one optimizer
step per rollout batch for both models, constant LR.
--baseline grpo: one advantage per rollout (the base). --baseline gae: per-token GAE(lambda) on the
leak-free prefix value from vbase2.py (attn arm only; docs/gae.md)."""
import argparse, copy, json, math, os, time
import numpy as np, torch, torch.nn.functional as F
from core import DEV, VGen, load_av, load_ar, save_av, save_ar, load_split, rawvar_baseline, mse_nrm

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["final", "multi", "attn"], required=True)
p.add_argument("--av", required=True)
p.add_argument("--ar", required=True)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--prompts", type=int, default=64)
p.add_argument("--n", type=int, default=8)
p.add_argument("--actor-lr", type=float, default=2e-5)
p.add_argument("--critic-lr", type=float, default=2e-5)
p.add_argument("--kl", type=float, default=0.01)
p.add_argument("--max-new", type=int, default=200)
p.add_argument("--mb", type=int, default=64)
p.add_argument("--eval-every", type=int, default=50)
p.add_argument("--ckpt-every", type=int, default=250, help="overwrite <out>/latest (disk is tight)")
p.add_argument("--new-lr-mult", type=float, default=10.0, help="critic LR multiplier for the attn readout layer (as in SFT)")
p.add_argument("--n-eval", type=int, default=8192, help="held-out vectors per periodic eval (1000 was too noisy)")
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--baseline", choices=["grpo", "gae"], default="grpo", help="gae: attn arm only")
p.add_argument("--gae-lambda", type=float, default=0.97, help="gae: lambda; 1 == pure control variate (A_grpo - f_t)")
p.add_argument("--vb-decay", type=float, default=0.9, help="per-step decay of the online value fit's statistics")
p.add_argument("--gen", choices=["vllm", "hf"], default="vllm")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
log = open(f"{a.out}/log.jsonl", "a")
def emit(**k):
    print(json.dumps(k), flush=True); log.write(json.dumps(k) + "\n"); log.flush()

pol = load_av(a.av); ref = load_av(a.av).eval().to(torch.bfloat16)  # frozen: bf16 weights, -15% fwd time
for q in ref.parameters():
    q.requires_grad_(False)
ar = load_ar(a.ar)
assert ar.arm == a.arm, (ar.arm, a.arm)
mode = "every" if a.arm == "multi" else "full"
_, _, Vr = load_split("rl", need_expl=False)
_, _, Var = load_split("ar")
_, _, evV = load_split("eval", need_expl=False)  # round trip needs vectors only (15.3k)
base = rawvar_baseline(Var)
emit(event="data", n_rl=len(Vr), n_eval=len(evV), rawvar=base, arm=a.arm)
opt_a = torch.optim.AdamW(pol.model.parameters(), lr=a.actor_lr, betas=(0.9, 0.95), weight_decay=0.0)
newn = ar.new_param_names()
cgroups = [{"params": [q for n, q in ar.named_parameters() if n not in newn], "lr": a.critic_lr},
           {"params": [q for n, q in ar.named_parameters() if n in newn], "lr": a.critic_lr * a.new_lr_mult}]
opt_c = torch.optim.AdamW([g for g in cgroups if g["params"]], betas=(0.9, 0.95), weight_decay=0.0)
P = len(pol.prompt)
pf = None; cv = None
if a.baseline == "gae":
    assert a.arm == "attn", "prefix values need the attn prefix scan"
    import vbase2
    pf = vbase2.PrefixFeatures(pol.tok)
    cv = vbase2.ControlVariate(decay=a.vb_decay)
eng = VGen(pol, a.av, seed=a.seed) if a.gen == "vllm" else pol


def score(expls, gold, critic=None):
    """-> mse per sample (nan for failed)."""
    critic = critic or ar
    m = torch.full((len(expls),), float("nan"))
    ok = [i for i, e in enumerate(expls) if e]
    if ok:
        critic.eval()
        pred = critic.predict_full([expls[i] for i in ok])
        m[ok] = mse_nrm(pred, gold[ok].to(DEV)).cpu()
    return m


@torch.no_grad()
def roundtrip(n, temps=(1.0,)):
    pol.eval(); r = {}
    V = evV[:n]
    for T in temps:
        gen = eng.generate(V, a.max_new, temperature=T)
        ex = pol.decode_expl(gen)
        m = score(ex, V)
        fail = torch.isnan(m)
        m2 = torch.where(fail, torch.full_like(m, 2.0), m)  # failed = orthogonal-equivalent
        r[f"T{T}"] = {"fve": 1 - m[~fail].mean().item() / base, "fve_fail_as_orth": 1 - m2.mean().item() / base,
                      "fail": fail.float().mean().item(),
                      "len": float(np.mean([len(t) for t, _ in gen]))}
        if T == temps[0]:
            r["sample"] = ex[:3]
    return r


def logps(model, ids, att, vecs):
    out = model.logits_from(ids, att, vecs, P - 1)
    return -F.cross_entropy(out.flatten(0, 1).float(), ids[:, P:].flatten(), reduction="none").view(ids.shape[0], -1)


t0 = time.time()
emit(step=0, eval=roundtrip(a.n_eval))
for step in range(1, a.steps + 1):
    idx = np.random.choice(len(Vr), a.prompts, replace=False)
    vec = Vr[idx].repeat_interleave(a.n, 0)
    pol.eval(); tt = [time.time()]
    gen = eng.generate(vec, a.max_new, temperature=1.0)
    ex = pol.decode_expl(gen)
    tt.append(time.time())
    if pf is None:
        m = score(ex, vec); S = None
    else:  # one fused critic pass: rewards + leak-free prefix scores at every state
        m, S, pfst = pf.scan(ar, gen, ex, vec)
    tt.append(time.time())
    fail = torch.isnan(m)
    rew = torch.where(fail, torch.full_like(m, -2.0), -m)
    g = rew.view(a.prompts, a.n)
    sd = (g.std(1, keepdim=True) + 1e-6).expand(-1, a.n).flatten()
    vbst = {}
    a_grpo = ((g - g.mean(1, keepdim=True)).flatten() / sd)  # [N] group-normalised advantage
    if cv is None:
        adv = a_grpo[:, None]  # [N, 1], same for every token
    else:  # [N, Lmax]: GAE(lambda) on the per-prefix value fit on past steps
        adv, vbst = cv.step(S, rew, [len(t) for t, _ in gen], a.n, a_grpo, gae_lambda=a.gae_lambda); vbst.update(pfst)
    tt.append(time.time())

    # ---- actor (GRPO, single on-policy step => ratio 1; sample-mean token aggregation)
    pol.train(); opt_a.zero_grad(set_to_none=True)
    N = len(gen); kls = []
    order = np.argsort([len(t) for t, _ in gen])
    for s in range(0, N, a.mb):
        mbi = order[s:s + a.mb]
        R = max(len(gen[i][0]) for i in mbi)
        ids = torch.full((len(mbi), P + R), pol.eos, dtype=torch.long)
        msk = torch.zeros(len(mbi), R)
        for j, i in enumerate(mbi):
            t = gen[i][0]
            ids[j, :P] = torch.tensor(pol.prompt); ids[j, P:P + len(t)] = torch.tensor(t); msk[j, :len(t)] = 1
        ids, msk = ids.to(DEV), msk.to(DEV)
        att = torch.cat([torch.ones(len(mbi), P, device=DEV), msk], 1).long()
        v = vec[mbi].to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            lp = logps(pol, ids, att, v)
            with torch.no_grad():
                lr_ = logps(ref, ids, att, v)
        d = lp - lr_
        av_ = adv[mbi][:, :R].to(DEV)
        tok = -av_ * torch.exp(lp - lp.detach()) + a.kl * 0.5 * d ** 2
        per = (tok * msk).sum(1) / msk.sum(1).clamp_min(1)
        (per.sum() / N).backward()
        kls.append(((0.5 * d.detach() ** 2 * msk).sum(1) / msk.sum(1).clamp_min(1)).sum().item())
    gna = torch.nn.utils.clip_grad_norm_(pol.model.parameters(), 1.0)
    opt_a.step()
    if a.gen == "vllm":
        eng.sync()
    tt.append(time.time())

    # ---- critic (online MSE on this rollout's valid explanations)
    ok = [i for i in range(N) if ex[i]]
    ar.train(); opt_c.zero_grad(set_to_none=True)
    for s in range(0, len(ok), 128):
        sl = ok[s:s + 128]
        with torch.autocast(DEV, dtype=torch.bfloat16):
            l, _, _ = ar.loss([ex[i] for i in sl], vec[sl].to(DEV), mode)
        (l * len(sl) / max(1, len(ok))).backward()
    gnc = torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
    opt_c.step()
    torch.cuda.synchronize(); tt.append(time.time())

    emit(step=step, reward=rew.mean().item(), fve_rollout=1 - m[~fail].mean().item() / base,
         fail=fail.float().mean().item(), len=float(np.mean([len(t) for t, _ in gen])),
         kl=sum(kls) / N, gna=gna.item(), gnc=gnc.item(), t=round(time.time() - t0),
         t_gen=tt[1] - tt[0], t_score=tt[2] - tt[1], t_vb=tt[3] - tt[2], t_actor=tt[4] - tt[3], t_critic=tt[5] - tt[4], **vbst)
    if step % a.eval_every == 0 or step == a.steps:
        emit(step=step, eval=roundtrip(a.n_eval))
    if step % a.ckpt_every == 0 and step < a.steps:
        save_av(pol, f"{a.out}/latest/av", {"args": vars(a), "step": step})
        save_ar(ar, f"{a.out}/latest/ar", {"arm": a.arm, "args": vars(a), "step": step})
save_av(pol, f"{a.out}/av", {"args": vars(a)})
save_ar(ar, f"{a.out}/ar", {"arm": a.arm, "args": vars(a)})
emit(step=a.steps, final=roundtrip(len(evV), temps=(1.0, 0.0)), t=round(time.time() - t0))
