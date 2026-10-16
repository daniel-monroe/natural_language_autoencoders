"""Small-scale NLA core: data, AR (three readout arms), AV with injection, metrics.

Faithful to the repo where it matters:
  - AR = blocks 0..K of the base LM (K+1 layers), final norm -> Identity, Linear(d, d) head
    initialised to identity (TRAINING_NOTES "identity-init value_head").
  - Loss = MSE between pred and gold, both L2-normalised to sqrt(d) (mse_scale="sqrt_d_model"),
    i.e. 2(1-cos). FVE = 1 - MSE / rawvar, rawvar = var of normalised golds about their raw mean.
  - Critic template "Summary of the following text: <text>{explanation}</text> <summary>",
    AV template = repo's _DEFAULT_ACTOR_TEMPLATE, vector injected at one token's embedding,
    rescaled to injection_scale.
Differences forced by the tiny model: SmolLM2's tokenizer has no single-token CJK char, so the
injection slot is the unused special token <gh_stars> (id 7). The critic prompt is tokenised in
three pieces (prefix / explanation / suffix) so every arm sees identical token ids.

AR readout arms:
  final  : read value head at the last suffix token (the repo's design).
  multi  : same model, but during training one copy of the suffix is attached after every
           10 explanation tokens (and at the end); each copy attends to the explanation prefix
           up to its cut plus itself, with positions continuing from the cut. One forward pass
           with a block mask == running each suffix on a KV-cached prefix.
  attn   : no suffix. One extra multi-head attention layer with learned queries attends over the
           last kept block's outputs at all positions (causally restricted to a cut when
           evaluating prefixes). Query init 0 => starts as mean pooling.
"""
import json, math, os, re
from glob import glob

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.stage3_build import _DEFAULT_ACTOR_TEMPLATE
from nla.schema import extract_explanation, wrap_explanation

DEV = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for smoke tests
BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
LAYER = 20
DATA = os.environ.get("NLA_SMALL_DATA", "nla_small/data")
INJ_TOKEN = "<gh_stars>"
CRITIC_PREFIX = "Summary of the following text: <text>"
CRITIC_SUFFIX = "</text> <summary>"
CUT_EVERY = 10
MAX_EXPL_TOKENS = 256


# ----------------------------------------------------------------------------- data
def load_split(split, need_expl=True):
    meta = pq.read_table(f"{DATA}/base.parquet").to_pandas()
    V = np.load(f"{DATA}/base_vecs.npy", mmap_mode="r")
    expl = {}
    for f in glob(f"{DATA}/expl/shard*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r["expl"] is not None:
                expl[r["row"]] = r["expl"]
    rows = np.nonzero((meta.split == split).values)[0]
    if need_expl:
        # Drop explainer artefacts: the explainer sometimes reads the "<" of its <end_text>
        # delimiter as part of the snippet (~10% of rows after the prompt fix).
        def stray_lt(r):
            e = expl[r]
            return ("angle bracket" in e.lower() or '"<' in e or "'<" in e or '<"' in e) and "<" not in meta.text[r][-80:]
        rows = np.array([r for r in rows if r in expl and not stray_lt(r)])
    texts = [expl.get(int(r)) for r in rows]
    return rows, texts, torch.from_numpy(np.ascontiguousarray(V[rows]))


def normalize(v, scale):
    return v / (v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) / scale).to(v.dtype)


def mse_nrm(pred, gold):
    d = gold.shape[-1]
    s = math.sqrt(d)
    return ((normalize(pred.float(), s) - normalize(gold.float(), s)) ** 2).mean(-1)


def rawvar_baseline(V):
    vn = normalize(V.float(), math.sqrt(V.shape[-1]))
    return ((vn - vn.mean(0, keepdim=True)) ** 2).mean().item()


# ----------------------------------------------------------------------------- AR
class AttnReadout(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.q = nn.Parameter(torch.zeros(heads, self.dh))
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, h, last):
        """Prefix-scan readout. h [B,T,d]; last [B,C] long = index of the last position each readout may
        see (readout j pools positions 0..last[b,j]). The queries are constants, so every prefix's
        softmax-pooled value is a ratio of running sums: one scan gives ALL cuts in O(T) per row.
        fp32 in log space (logcumsumexp; values split into +/- parts), so no prefix can underflow."""
        B, T, _ = h.shape
        x = self.norm(h)
        K = self.k(x).view(B, T, self.h, self.dh).float()
        Vv = self.v(x).view(B, T, self.h, self.dh).float()
        s = torch.einsum("hd,bthd->bth", self.q.float(), K) / math.sqrt(self.dh)            # [B,T,H]
        lden = torch.logcumsumexp(s, 1)
        ls = s[..., None]
        pos = torch.logcumsumexp(ls + torch.log(Vv.clamp_min(1e-30)), 1)                         # [B,T,H,dh]
        neg = torch.logcumsumexp(ls + torch.log((-Vv).clamp_min(1e-30)), 1)  # same tiny floor on both sides cancels; no -inf/NaN grads
        ix = last[:, :, None, None].expand(-1, -1, self.h, self.dh)
        ld = torch.gather(lden, 1, last[:, :, None].expand(-1, -1, self.h))[..., None]
        out = torch.exp(torch.gather(pos, 1, ix) - ld) - torch.exp(torch.gather(neg, 1, ix) - ld)
        return self.o(out.to(h.dtype).reshape(B, last.shape[1], -1))

    def forward_masked(self, h, keymask):  # reference implementation; h [B,T,d]; keymask [B,C,T] bool -> [B,C,d]
        B, T, _ = h.shape
        x = self.norm(h)
        K = self.k(x).view(B, T, self.h, self.dh)
        Vv = self.v(x).view(B, T, self.h, self.dh)
        logits = torch.einsum("hd,bthd->bht", self.q, K) / math.sqrt(self.dh)
        logits = logits[:, None].expand(B, keymask.shape[1], self.h, T)
        logits = logits.masked_fill(~keymask[:, :, None, :], float("-inf"))
        att = logits.float().softmax(-1).to(Vv.dtype)
        out = torch.einsum("bcht,bthd->bchd", att, Vv).reshape(B, keymask.shape[1], -1)
        return self.o(out)


class AR(nn.Module):
    def __init__(self, readout="final", base=BASE, layer=LAYER):
        super().__init__()
        m = AutoModelForCausalLM.from_pretrained(base, num_hidden_layers=layer + 1,
                                                 torch_dtype=torch.float32, attn_implementation="sdpa")
        self.lm = m.model
        self.lm.norm = nn.Identity()
        d = m.config.hidden_size
        self.head = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(d))
        self.readout = readout
        if readout == "attn":
            self.pool = AttnReadout(d, m.config.num_attention_heads)
        self.tok = AutoTokenizer.from_pretrained(base)
        self.P = self.tok(CRITIC_PREFIX, add_special_tokens=True).input_ids
        self.S = self.tok(CRITIC_SUFFIX, add_special_tokens=False).input_ids

    def new_param_names(self):
        return {n for n, _ in self.named_parameters() if n.startswith("pool.")}

    def backbone(self, ids, pos, mask):
        h = self.lm.embed_tokens(ids)
        pe = self.lm.rotary_emb(h, pos)
        for layer in self.lm.layers:
            o = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
            h = o[0] if isinstance(o, tuple) else o
        return h

    def build(self, texts, cuts_mode, device):
        """cuts_mode: 'full' (single readout at the end) or 'every' (every CUT_EVERY tokens + end).
        Returns tensors plus per-sample list of (readout index or cut length) and cut lengths."""
        P, S = self.P, self.S
        seqs, poss, specs = [], [], []
        for t in texts:
            E = self.tok(t, add_special_tokens=False).input_ids[:MAX_EXPL_TOKENS]
            L = len(E)
            cuts = [L] if cuts_mode == "full" else sorted(set(list(range(CUT_EVERY, L, CUT_EVERY)) + [L]))
            ids = P + E
            pos = list(range(len(ids)))
            spec = []  # (start of copy, cut) ; for attn: (None, cut)
            if self.readout == "attn":
                spec = [(None, c) for c in cuts]
            else:
                for c in cuts:
                    spec.append((len(ids), c))
                    ids = ids + S
                    pos = pos + list(range(len(P) + c, len(P) + c + len(S)))
            seqs.append(ids); poss.append(pos); specs.append(spec)
        T = max(map(len, seqs))
        B = len(seqs)
        ids = torch.zeros(B, T, dtype=torch.long)
        pos = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, T, dtype=torch.bool)
        M = [len(P) + len(self.tok(t, add_special_tokens=False).input_ids[:MAX_EXPL_TOKENS]) for t in texts]
        tri = torch.ones(T, T, dtype=torch.bool).tril()
        for b in range(B):
            n = len(seqs[b])
            ids[b, :n] = torch.tensor(seqs[b]); pos[b, :n] = torch.tensor(poss[b])
            m = M[b]
            mask[b, :m, :m] = tri[:m, :m]
            for st, c in specs[b]:
                if st is None:
                    continue
                s = len(S)
                mask[b, st:st + s, :len(P) + c] = True
                mask[b, st:st + s, st:st + s] = tri[:s, :s]
            idx = torch.arange(n, T)
            mask[b, idx, idx] = True  # padding rows attend to themselves (no NaN)
        return ids.to(device), pos.to(device), mask[:, None].to(device), specs, M

    def forward(self, texts, cuts_mode="full"):
        """Returns list (per sample) of [n_cuts, d] predictions and list of cut lengths."""
        dev = self.head.weight.device
        ids, pos, mask, specs, M = self.build(texts, cuts_mode, dev)
        if cuts_mode == "full":
            # one causal right-padded sequence per row: real tokens never see padding, so drop the
            # dense mask and let SDPA take the causal flash kernel (identical readouts).
            mask = None
        h = self.backbone(ids, pos, mask)
        if self.readout == "attn":
            C = max(len(s) for s in specs)
            last = torch.zeros(len(texts), C, dtype=torch.long)
            for b, sp in enumerate(specs):
                last[b, :len(sp)] = torch.tensor([len(self.P) + c - 1 for _, c in sp])
            r = self.head(self.pool(h, last.to(dev)))  # [B,C,d]
            preds = [r[b, :len(sp)] for b, sp in enumerate(specs)]
        else:
            preds = []
            for b, sp in enumerate(specs):
                ix = torch.tensor([st + len(self.S) - 1 for st, _ in sp], device=dev)
                preds.append(self.head(h[b, ix]))
        cuts = [[c for _, c in sp] for sp in specs]
        return preds, cuts

    def loss(self, texts, gold, cuts_mode):
        preds, cuts = self(texts, cuts_mode)
        # full-length readout always gets weight 1 (same gradient as the 'final' arm); in 'every'
        # mode the mean over shorter cuts is added on top (review fix: averaging all cuts gave the
        # full-length readout ~1/12 of the gradient, confounding the arm comparison).
        per = []
        for b, p in enumerate(preds):
            m = mse_nrm(p, gold[b:b + 1].expand_as(p))
            per.append(m[-1] + (m[:-1].mean() if len(m) > 1 else 0.0))
        return torch.stack(per).mean(), preds, cuts

    @torch.no_grad()
    def predict_full(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            with torch.autocast(DEV, dtype=torch.bfloat16):
                p, _ = self(texts[i:i + bs], "full")
            out.append(torch.stack([x[-1] for x in p]).float())
        return torch.cat(out)


def save_ar(ar, path, extra=None):
    os.makedirs(path, exist_ok=True)
    torch.save({"state": ar.state_dict(), "readout": ar.readout, **(extra or {})}, f"{path}/ar.pt")


def load_ar(path, device=DEV):
    ck = torch.load(f"{path}/ar.pt", map_location="cpu")
    ar = AR(ck["readout"])
    ar.load_state_dict(ck["state"])
    ar.arm = ck.get("arm")
    return ar.to(device)


# ----------------------------------------------------------------------------- AV
class AV(nn.Module):
    def __init__(self, init=BASE, inj_scale=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(init, torch_dtype=torch.float32,
                                                          attn_implementation="sdpa")
        self.tok = AutoTokenizer.from_pretrained(BASE)
        content = _DEFAULT_ACTOR_TEMPLATE.format(injection_char=INJ_TOKEN)
        self.prompt = self.tok.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=True)
        if isinstance(self.prompt, dict):
            self.prompt = self.prompt["input_ids"]
        inj_id = self.tok.convert_tokens_to_ids(INJ_TOKEN)
        hits = [i for i, t in enumerate(self.prompt) if t == inj_id]
        assert len(hits) == 1, hits
        self.inj_pos = hits[0]
        nb = self.tok.convert_ids_to_tokens([self.prompt[self.inj_pos - 1], self.prompt[self.inj_pos + 1]])
        assert nb == [">", "</"], nb
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.inj_scale = inj_scale
        self.share_prefix = True

    def embed(self, ids, vecs):
        e = self.model.get_input_embeddings()(ids)
        e = e.clone()
        e[:, self.inj_pos] = normalize(vecs.float(), self.inj_scale).to(e.dtype)
        return e

    def sft_batch(self, texts, vecs, device):
        P = self.prompt
        rs = [self.tok(wrap_explanation(t), add_special_tokens=False).input_ids + [self.eos] for t in texts]
        T = len(P) + max(map(len, rs))
        ids = torch.full((len(rs), T), self.eos, dtype=torch.long)
        lab = torch.full((len(rs), T), -100, dtype=torch.long)
        att = torch.zeros(len(rs), T, dtype=torch.long)
        for b, r in enumerate(rs):
            n = len(P) + len(r)
            ids[b, :n] = torch.tensor(P + r); att[b, :n] = 1
            lab[b, len(P):n] = torch.tensor(r)
        return ids.to(device), att.to(device), lab.to(device)

    def logits_from(self, ids, att, vecs, start):
        """Logits predicting tokens start+1.. only (skips the lm_head over the 134-token prompt:
        full-sequence fp32 logits over the 49k vocab OOM'd a 48GB A6000 at bs 64)."""
        # att unused: sequences are right-padded and causal, so real tokens never see padding (padded
        # positions are loss-masked); passing no mask lets SDPA take the flash kernel.
        # The prompt up to the injection slot (117 of 134 tokens) is identical for every row: encode it
        # once (B=1, with grad) and broadcast its KV cache; exact same math, ~half the tokens.
        e = self.embed(ids, vecs)
        if not self.share_prefix:
            return self.model.lm_head(self.model.model(inputs_embeds=e).last_hidden_state[:, start:-1])
        s, B = self.inj_pos, ids.shape[0]
        assert start >= s
        cache = self.model.model(inputs_embeds=e[:1, :s], use_cache=True).past_key_values
        for L in cache.layers:
            L.keys, L.values = L.keys.expand(B, -1, -1, -1), L.values.expand(B, -1, -1, -1)
        h = self.model.model(inputs_embeds=e[:, s:], past_key_values=cache, use_cache=True).last_hidden_state
        return self.model.lm_head(h[:, start - s:-1])

    def sft_loss(self, texts, vecs, reduce=True):
        dev = self.model.device
        ids, att, lab = self.sft_batch(texts, vecs, dev)
        s = len(self.prompt) - 1
        logits = self.logits_from(ids, att, vecs.to(dev), s)
        l = F.cross_entropy(logits.flatten(0, 1).float(), lab[:, s + 1:].flatten(), reduction="none")
        l = l.view(lab.shape[0], -1)
        m = (lab[:, s + 1:] != -100).float()
        if reduce:  # per-sample token mean, then mean over samples (Miles sum_of_sample_mean)
            return ((l * m).sum(1) / m.sum(1).clamp_min(1)).mean()
        return (l * m).sum(1), m.sum(1)

    @torch.no_grad()
    def generate(self, vecs, max_new=200, temperature=1.0, bs=512):
        """Returns list of (token_ids, completed) per vector. completed = emitted <|im_end|>."""
        dev = self.model.device
        out = []
        for i in range(0, len(vecs), bs):
            v = vecs[i:i + bs].to(dev)
            ids = torch.tensor([self.prompt] * len(v), device=dev)
            emb = self.embed(ids, v)
            kw = dict(do_sample=temperature > 0, max_new_tokens=max_new, eos_token_id=self.eos,
                      pad_token_id=self.eos, attention_mask=torch.ones_like(ids))
            if temperature > 0:
                kw.update(temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                g = self.model.generate(inputs_embeds=emb, **kw)
            for row in g.tolist():
                if self.eos in row:
                    out.append((row[:row.index(self.eos) + 1], True))
                else:
                    out.append((row, False))
        return out

    def decode_expl(self, gen):
        """Repo semantics: only completed generations with a closed <explanation> tag count."""
        res = []
        for toks, done in gen:
            txt = self.tok.decode(toks, skip_special_tokens=True)
            e = extract_explanation(txt) if done else None
            res.append(e if e else None)
        return res


class VGen:
    """vLLM rollout engine for an AV (RL hot path; HF generate is a Python loop per token).
    The injected vector enters through prompt_embeds = AV.embed(prompt) (exact same embeddings as
    the HF path); weights are pushed in-process after every optimizer step (sync()). Like the repo
    (SGLang sampler + FSDP trainer), sampling runs in bf16 while log-probs come from the trainer."""

    def __init__(self, av, path, mem=0.2, seed=0):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process => direct weight sync
        from vllm import LLM
        self.av = av
        self.llm = LLM(model=path, tokenizer=BASE, skip_tokenizer_init=True, enable_prompt_embeds=True,
                       dtype="bfloat16", gpu_memory_utilization=mem, max_model_len=1024,  # >= max_num_seqs: vLLM 0.11 sizes a pinned sampled-token buffer by max_model_len
                       enable_prefix_caching=False, seed=seed, max_num_seqs=1024)
        self.sync()

    def sync(self):
        sd = {k: v.detach() for k, v in self.av.model.state_dict().items()}
        self.llm.collective_rpc(lambda w: w.model_runner.model.load_weights(sd.items()) and None)

    @torch.no_grad()
    def generate(self, vecs, max_new=200, temperature=1.0):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, max_tokens=max_new,
                            stop_token_ids=[self.av.eos], detokenize=False)
        dev = self.av.model.device
        ids = torch.tensor([self.av.prompt] * len(vecs), device=dev)
        emb = self.av.embed(ids, vecs.to(dev)).to(torch.bfloat16).cpu()
        outs = self.llm.generate([{"prompt_embeds": e} for e in emb], sp, use_tqdm=False)
        res = []
        for o in outs:
            t = list(o.outputs[0].token_ids)
            done = o.outputs[0].finish_reason == "stop"
            if done and (not t or t[-1] != self.av.eos):
                t.append(self.av.eos)
            res.append((t, done))
        return res


def save_av(av, path, extra=None):
    os.makedirs(path, exist_ok=True)
    av.model.save_pretrained(path, safe_serialization=True)
    json.dump({"inj_scale": av.inj_scale, **(extra or {})}, open(f"{path}/nla_small.json", "w"))


def load_av(path, device=DEV):
    cfg = json.load(open(f"{path}/nla_small.json"))
    return AV(path, cfg["inj_scale"]).to(device)


def cosine_lr(step, total, lr, warmup, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
