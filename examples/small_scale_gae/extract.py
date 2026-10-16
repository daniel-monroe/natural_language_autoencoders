"""Stage 0 (small): FineWeb docs -> layer-K residual vectors of SmolLM2-135M-Instruct.

Mirrors nla/datagen/stage0_extract.py: per-doc keyed position sampling (_MIN_POSITION=50,
reused verbatim), raw (unnormalised) vectors, hook output of block K (= hidden_states[K+1]),
right-truncation at max_length, add_special_tokens=True. Document-level split.
Outputs data/base.parquet (meta + text) and data/base_vecs.npy (float32 [N, d]).
"""
import argparse, hashlib, os
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.datagen.stage0_extract import _sample_positions

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
p.add_argument("--layer", type=int, default=20)
p.add_argument("--n-docs", type=int, default=30000)
p.add_argument("--positions", type=int, default=5)
p.add_argument("--max-length", type=int, default=1024)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default=os.environ.get("NLA_SMALL_DATA", "nla_small/data"))
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "right"; tok.truncation_side = "right"
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda().eval()
special = set(tok.all_special_ids)

def split_of(doc_id):  # document-level split, keyed hash: 30/30/30/10
    u = int(hashlib.sha256(f"split|{doc_id}".encode()).hexdigest()[:8], 16) / 2**32
    return "av" if u < .3 else "ar" if u < .6 else "rl" if u < .9 else "eval"

ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
rows = {k: [] for k in ["doc_id", "split", "n_raw_tokens", "text"]}
vecs = []
buf = []
def flush(buf):
    texts = [t for _, t in buf]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=a.max_length, add_special_tokens=True)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids.cuda(), attention_mask=enc.attention_mask.cuda(), output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[a.layer + 1]  # output of block K
    for b, (doc_id, _) in enumerate(buf):
        ids = enc.input_ids[b][enc.attention_mask[b].bool()].tolist()
        for pos in sorted(_sample_positions(ids, a.positions, special, doc_id, a.seed)):
            rows["doc_id"].append(doc_id); rows["split"].append(split_of(doc_id))
            rows["n_raw_tokens"].append(pos + 1)
            rows["text"].append(tok.decode(ids[: pos + 1], skip_special_tokens=True))
            vecs.append(hs[b, pos].float().cpu().numpy())
n = 0
for ex in ds:
    buf.append((ex["id"], ex["text"])); n += 1
    if len(buf) == a.batch:
        flush(buf); buf = []
    if n >= a.n_docs: break
    if n % 5000 == 0: print(n, "docs", len(vecs), "vecs", flush=True)
if buf: flush(buf)
V = np.stack(vecs).astype(np.float32)
np.save(f"{a.out}/base_vecs.npy", V)
pq.write_table(pa.table(rows), f"{a.out}/base.parquet")
norms = np.linalg.norm(V, axis=1)
print("rows", len(V), "d", V.shape[1], "norm mean/median/p95", norms.mean(), np.median(norms), np.percentile(norms, 95))
import collections; print(collections.Counter(rows["split"]))
