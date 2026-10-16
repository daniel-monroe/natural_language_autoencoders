"""Leak-free per-token prefix values + GAE advantages for rl.py --baseline gae (docs/gae.md).

State t of rollout i = response tokens toks[:t]. Everything that multiplies grad log pi(a_t | s_t) must be a
function of toks[:t] (+ other rollouts, + past steps). an earlier version violated this four ways (docs/gae.md, "Leak tests"). Here:
  - NO retokenisation of the final explanation and no text search on the final response. All alignment is in
    BYTES of the AV's own tokens: byte offset of state t = sum of token byte lengths < t (exact; specials = 0).
  - body tokens = the AV's own token ids after the first "<explanation>" (leading pure-whitespace tokens
    skipped). k(t) = number of body tokens visible at state t; after the full "</explanation>" has been
    emitted, k = body up to the close tag minus trailing whitespace tokens. Partial tags are just past tokens.
  - prefix score s_t = -mse of the critic readout after k(t) body tokens: ONE causal critic pass over
    CRITIC_PREFIX + body ids, attn prefix scan at every position (readout k sees positions < nP + k only).
  - every rollout gets features, failed ones included (no gating on the future).
Per-prefix value (scale-free, in the optimizer's units): f_t = linear (per 20-token bin) prediction of
A_grpo = (R - mean_g) / sd_g from features normalised by sd_{-i} (std of the OTHER rollouts' rewards):
(s_t - rloo), (s_t - others' mean s at t), (others' mean - rloo), missing flag, t/200, fraction of others
finished. Fit online on past steps only. ControlVariate.step turns it into GAE(lambda) advantages
(lambda = 1: A_grpo - f_t; f == 0: exactly GRPO).
Rewards: the fused pass also yields the full-explanation readout for rollouts whose body ids start with the
canonical ids of the extracted explanation (the usual case); the rest are scored with AR.predict_full."""
import numpy as np, torch
from core import DEV, mse_nrm, MAX_EXPL_TOKENS

NB = 10
OPEN, CLOSE = b"<explanation>", b"</explanation>"


class PrefixFeatures:
    def __init__(self, tok):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        bd = {c: b for b, c in bytes_to_unicode().items()}
        spec = set(tok.all_special_ids) | set(tok.added_tokens_decoder.keys())
        vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
        self.tbytes = [b"" if i in spec else bytes(bd[c] for c in s) for i, s in enumerate(vocab)]
        self.ws = np.array([len(b) > 0 and b.strip() == b"" for b in self.tbytes])
        self.tok = tok

    def align(self, toks):
        """-> (body ids for the scan, k[t] for states t = 0..L). Function of the token sequence only, and
        k[t] depends on toks[:t] only (leak test, docs/gae.md §6)."""
        L = len(toks)
        k = np.zeros(L + 1, np.int64)
        tb = [self.tbytes[x] for x in toks]
        off = np.zeros(L + 1, np.int64)
        off[1:] = np.cumsum([len(b) for b in tb])
        bs = b"".join(tb)
        o = bs.find(OPEN)
        if o < 0:
            return [], k
        oe = o + len(OPEN)
        b0 = int(np.searchsorted(off[1:], oe, side="right"))   # first token whose end byte > oe
        while b0 < L and self.ws[toks[b0]]:
            b0 += 1
        c = bs.find(CLOSE, oe)
        if c >= 0:
            tc = int(np.searchsorted(off, c + len(CLOSE), side="left"))  # first state that has seen the full tag
            jc = int(np.searchsorted(off[1:], c, side="right"))  # token containing byte c
            je = jc
            while je > b0 and self.ws[toks[je - 1]]:
                je -= 1
            end_scan = max(b0, tc - 1)
        else:
            tc = L + 1; end_scan = L
        end_scan = min(end_scan, b0 + MAX_EXPL_TOKENS)
        t = np.arange(L + 1)
        k = np.clip(t - b0, 0, end_scan - b0)
        if c >= 0:
            k[tc:] = min(max(0, je - b0), end_scan - b0)
        return list(toks[b0:end_scan]), k

    @torch.no_grad()
    def scan(self, ar, gen, ex, gold, bs=128):
        """-> rewards mse [N] (nan where ex is None), S [N, Tm+1] prefix score -mse at every state (nan if k=0)."""
        N = len(gen); Lav = [len(t) for t, _ in gen]; Tm = max(Lav)
        al = [self.align(list(t)) for t, _ in gen]
        E = {}
        okx = [i for i in range(N) if ex[i]]
        if okx:
            enc = self.tok([ex[i] for i in okx], add_special_tokens=False).input_ids
            E = {i: e[:MAX_EXPL_TOKENS] for i, e in zip(okx, enc)}
        canon = {i for i in okx if len(E[i]) and al[i][0][:len(E[i])] == E[i]}
        mse = torch.full((N,), float("nan")); S = np.full((N, Tm + 1), np.nan, np.float32)
        nP = len(ar.P); ar.eval(); dev = ar.head.weight.device
        rows = [i for i in range(N) if len(al[i][0])]
        for s in range(0, len(rows), bs):
            rb = rows[s:s + bs]
            seqs = [ar.P + al[i][0] for i in rb]
            T = max(map(len, seqs))
            ids = torch.zeros(len(rb), T, dtype=torch.long)
            for j, q in enumerate(seqs):
                ids[j, :len(q)] = torch.tensor(q)
            Lb = torch.tensor([len(al[i][0]) for i in rb])
            Lm = int(Lb.max())
            ar_ = torch.arange(Lm)
            last = torch.where(ar_[None] < Lb[:, None], nP + ar_[None], 0).to(dev)  # readout after k=1..Lm body tokens
            with torch.autocast(DEV, dtype=torch.bfloat16):
                h = ar.backbone(ids.to(dev), torch.arange(T, device=dev)[None].expand(len(rb), -1), None)
                r = ar.head(ar.pool(h, last))
            g = gold[rb].to(dev)
            mm = mse_nrm(r.float(), g[:, None, :].expand(-1, Lm, -1)).cpu().numpy()  # mm[j, k-1]
            for j, i in enumerate(rb):
                k = al[i][1]
                S[i, :len(k)] = np.where(k > 0, -mm[j, np.clip(k - 1, 0, Lm - 1)], np.nan)
                if i in canon:
                    mse[i] = float(mm[j, len(E[i]) - 1])
        rest = [i for i in okx if i not in canon]
        if rest:
            pred = ar.predict_full([ex[i] for i in rest])
            mse[rest] = mse_nrm(pred, gold[rest].to(DEV)).cpu()
        for i in range(N):  # after the end: final state value
            S[i, Lav[i] + 1:] = S[i, Lav[i]]
        return mse, S, {"vb_noncanon_rescored": len(rest), "vb_scored_fused": len(okx) - len(rest)}


def design(F, tb):
    k = F.shape[1]
    D = np.zeros((len(F), k * (NB + 1) + 1), np.float64)
    D[:, :k] = F
    for b in range(NB):
        m = tb == b
        D[m, k * (b + 1):k * (b + 2)] = F[m]
    D[:, -1] = 1.0
    return D


class ControlVariate:
    def __init__(self, decay=0.9, ridge=1e-4, zclip=10.0):
        self.decay, self.ridge, self.zclip = decay, ridge, zclip
        self.A = None; self.bv = None; self.beta = None

    def features(self, S, rew, Lav, n):
        """rows for every rollout i and position t < L_i -> (F, ri, rt). Inputs per row: own S[i, t] (state t),
        other rollouts of the group (rewards, S at t, finished flags). Never R_i, never S[i, >t]."""
        N = len(Lav); G = N // n
        R = rew.numpy().astype(np.float64).reshape(G, n)
        Fs, ris, rts = [], [], []
        for g in range(G):
            for a in range(n):
                i = g * n + a; L = Lav[i]
                if L == 0:
                    continue
                oth = [g * n + b for b in range(n) if b != a]
                Ro = R[g, [b for b in range(n) if b != a]]
                rloo = Ro.mean(); sdo = max(Ro.std(ddof=1), 1e-3)
                O = S[oth, :L]
                cnt = (~np.isnan(O)).sum(0)
                om = np.where(cnt > 0, np.nansum(O, 0) / np.maximum(cnt, 1), rloo)
                s = S[i, :L].astype(np.float64)
                miss = np.isnan(s)
                s = np.where(miss, om, s)
                done = (np.arange(L)[None] >= np.array([Lav[j] for j in oth])[:, None]).mean(0)
                z = np.stack([(s - rloo) / sdo, (s - om) / sdo, (om - rloo) / sdo], 1).clip(-self.zclip, self.zclip)
                F = np.concatenate([z, miss[:, None], (np.arange(L) / 200.0)[:, None], done[:, None]], 1)
                Fs.append(F); ris.append(np.full(L, i)); rts.append(np.arange(L))
        return np.concatenate(Fs), np.concatenate(ris), np.concatenate(rts)

    def step(self, S, rew, Lav, n, a_grpo, gae_lambda=0.97):
        """a_grpo [N] = rl.py's GRPO advantage. -> (adv [N, Tm] tensor, stats).
        GAE(lambda) with value V_t = f_t (state t, a_t-independent; the fit from PAST steps), terminal
        V_L = a_grpo, no intermediate rewards, gamma = 1: adv_t = sum_k lambda^k (V_{t+k+1} - V_{t+k}).
        lambda = 1 is the pure control variate a_grpo - f_t; lambda < 1 is biased by design (V_{t+k+1}
        stands in for the realised return). Before the first fit (step 1) adv == a_grpo on every token."""
        N = len(Lav); Tm = max(Lav)
        F, ri, rt = self.features(S, rew, Lav, n)
        D = design(F, np.minimum(rt // 20, NB - 1))
        y = a_grpo.numpy().astype(np.float64)[ri]
        w = 1.0 / np.asarray(Lav, np.float64)[ri] ** 2
        adv = np.repeat(a_grpo.numpy()[:, None], Tm, 1).astype(np.float32)
        st = {"vb_rows": int(len(F)), "vb_fitted": self.beta is not None}
        if self.beta is not None:
            f = D @ self.beta
            V = np.zeros((N, Tm + 1)); V[ri, rt] = f
            ag = a_grpo.numpy()
            for i in range(N):
                L = Lav[i]
                if L == 0:
                    continue
                V[i, L] = ag[i]; acc = 0.0
                for t in range(L - 1, -1, -1):
                    acc = (V[i, t + 1] - V[i, t]) + gae_lambda * acc
                    adv[i, t] = acc
            st["gae_lambda"] = gae_lambda
            a_ = adv[ri, rt].astype(np.float64)
            st["vb_norm_adv2_ratio_vs_grpo"] = float((w * a_ ** 2).sum() / (w * y ** 2).sum())
        Aw = (D * w[:, None]).T
        self.A = Aw @ D if self.A is None else self.decay * self.A + Aw @ D
        self.bv = Aw @ y if self.bv is None else self.decay * self.bv + Aw @ y
        lam = self.ridge * np.trace(self.A) / len(self.A)
        self.beta = np.linalg.solve(self.A + lam * np.eye(len(self.A)), self.bv)
        return torch.tensor(adv), st
