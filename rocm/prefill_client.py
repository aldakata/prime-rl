"""Reader-style prefill bench: /v1/completions with prompt_logprobs=0, max_tokens=1, random unique token prompts."""
import json, random, sys, threading, time, urllib.request
url, model, n, conc, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
rng = random.Random(0)
prompts = [[rng.randrange(1000, 150000) for _ in range(rng.randint(lo, hi))] for _ in range(n)]
lock = threading.Lock(); idx = [0]; done_tok = [0]; errs = [0]; lat = []
def worker():
    while True:
        with lock:
            if idx[0] >= n: return
            p = prompts[idx[0]]; idx[0] += 1
        body = json.dumps(dict(model=model, prompt=p, max_tokens=1, temperature=0.0, prompt_logprobs=0)).encode()
        t = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=3600))
            assert len(r["choices"][0]["prompt_logprobs"]) == len(p)
            with lock: done_tok[0] += len(p); lat.append(time.time() - t)
        except Exception as e:
            with lock: errs[0] += 1
            print("err", repr(e)[:200], flush=True)
# warmup
n0 = n; t0 = time.time()
ts = [threading.Thread(target=worker) for _ in range(conc)]; [x.start() for x in ts]; [x.join() for x in ts]
dt = time.time() - t0; lat.sort()
print(json.dumps(dict(requests=n0, errors=errs[0], concurrency=conc, prompt_tokens=done_tok[0], seconds=round(dt, 1),
      prefill_tok_s=round(done_tok[0] / dt), req_s=round((n0 - errs[0]) / dt, 2),
      p50_latency_s=round(lat[len(lat) // 2], 1) if lat else None, p95_latency_s=round(lat[int(len(lat) * .95)], 1) if lat else None)), flush=True)
