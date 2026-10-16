"""Stage 2 (small): explanations from a local open-weights LLM via vLLM (no API key available).

Uses the repo's exact instruction prompt (_DEFAULT_INSTRUCTION), strict <analysis> extraction and
list-cleanup (_extract_and_clean), and the >=2-feature filter. Explains the av/ar splits plus a
fixed eval subset. One process per GPU; shard = row_index % world.
"""
import argparse, os, json
import numpy as np, pyarrow.parquet as pq
from nla.datagen.stage2_api_explain import _DEFAULT_INSTRUCTION as _REPO_INSTRUCTION, _DEFAULT_RESPONSE_PATTERN, _extract_and_clean, _MIN_FEATURES

p = argparse.ArgumentParser()
p.add_argument("--model", default="stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ")
p.add_argument("--rank", type=int, required=True)
p.add_argument("--world", type=int, required=True)
p.add_argument("--n-eval", type=int, default=3000)
p.add_argument("--limit", type=int, default=0, help="smoke test: only this many rows")
p.add_argument("--data", default=os.environ.get("NLA_SMALL_DATA", "nla_small/data"))
a = p.parse_args()
# One clause added to the repo prompt: Qwen3 otherwise reads the "<" of <end_text> as part of the
# snippet and invents a dangling "<" in 2/3 of explanations (seen in the 64-row smoke test).
_DEFAULT_INSTRUCTION = _REPO_INSTRUCTION.replace(
    "Text to analyze:", "Text to analyze (<begin_text> and <end_text> are delimiters, not part of the text):")
assert _DEFAULT_INSTRUCTION != _REPO_INSTRUCTION

t = pq.read_table(f"{a.data}/base.parquet").to_pandas()
t["row"] = np.arange(len(t))
ev = t[t.split == "eval"]
ev = ev.sample(n=min(a.n_eval, len(ev)), random_state=0)
want = t[t.split.isin(["av", "ar"])]
todo = sorted(set(want.row) | set(ev.row))
if a.limit: todo = todo[: a.limit]
todo = [r for r in todo if r % a.world == a.rank]
out_path = f"{a.data}/expl/shard{a.rank}.jsonl"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["row"] for l in open(out_path)}
todo = [r for r in todo if r not in done]
print(f"rank {a.rank}: {len(todo)} rows to do ({len(done)} done)", flush=True)

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=0.90, enable_prefix_caching=True)
sp = SamplingParams(temperature=1.0, max_tokens=300)  # repo stage2 config: max_tokens 300, T=1.0
CH = 4096
with open(out_path, "a") as f:
    for s in range(0, len(todo), CH):
        rows = todo[s : s + CH]
        msgs = [[{"role": "user", "content": _DEFAULT_INSTRUCTION.format(text=t.text.iloc[r])}] for r in rows]
        outs = llm.chat(msgs, sp, use_tqdm=False)
        kept = 0
        for r, o in zip(rows, outs):
            raw = o.outputs[0].text
            c = _extract_and_clean(raw, _DEFAULT_RESPONSE_PATTERN)
            ok = c is not None and c.count("\n\n") + 1 >= _MIN_FEATURES
            kept += ok
            f.write(json.dumps({"row": int(r), "expl": c if ok else None, "raw": raw if not ok else None}) + "\n")
        f.flush()
        print(f"rank {a.rank}: {s + len(rows)}/{len(todo)} kept {kept}/{len(rows)}", flush=True)
print(f"rank {a.rank}: DONE", flush=True)
os._exit(0)  # vLLM 0.11 engine shutdown hangs for >5 min after the last request; skip it
