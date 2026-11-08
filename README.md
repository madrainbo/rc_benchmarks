# rc_benchmarks

High-level benchmark results for **rc_engine**, a Rust inference engine (CUDA and Metal kernels) for Mixture-of-Experts language models, compared with llama.cpp. The engine's source is in a private repository, available on request.

**Setup.** NVIDIA RTX 5080 16 GB (Ryzen 9 9900X, WSL2) and Apple M4 Max 64 GB. Greedy sampling, one warmup run discarded, five timed runs, median reported. Same GGUF file and context size for both engines. CUDA toolkit 13.1 (13.1.115). llama.cpp `7d56da7` on the PC, except Nemotron, which needs a newer snapshot built 2026-10-10 (0.5.0-dev, commit unknown); on the Mac, 0.5.0-dev (commit unknown). The Qwen3.6-35B-A3B results on the RTX 5080 were re-measured on 2026-11-08 with the same llama.cpp build; the other rows are from October.

## Single-stream decode (tok/s)

128-token prompt, 256 generated tokens.

| Machine | Model | rc_engine | llama.cpp | Ratio |
| --- | --- | --- | --- | --- |
| RTX 5080 | LFM2.5-8B-A1B Q4_K_M | 545.0 | 444.0 | 1.23× |
| RTX 5080 | Qwen3.6-35B-A3B IQ3_S | 216.3 | 176.6 | 1.22× |
| RTX 5080 | Qwen3.6-14B-A3B Q6_K | 204.4 | 176.4 | 1.16× |
| RTX 5080 | GLM-4.7-Flash Q3_K_XL | 202.7 | 182.5 | 1.11× |
| RTX 5080 | Nemotron-3.5-30B IQ3_XXS | 151.0 | 112.6 | 1.34× |
| M4 Max | Qwen3.6-35B-A3B IQ3_S | 112.6 | 97.0–101.5 | 1.11–1.16× |
| M4 Max | Qwen3.6-35B-A3B Q4_K_M | 114.4 | 90.5 | 1.26× |

For Nemotron, whose 19.7 GB file does not fit the card, llama.cpp's figure is its best `--n-cpu-moe` setting.

## Concurrent streams (RTX 5080, Qwen3.6-35B-A3B IQ3_S)

Aggregate tok/s, 256 tokens per stream, each stream continuing a different passage of text.

| Streams | rc_engine | llama.cpp | Ratio |
| --- | --- | --- | --- |
| 1 | 200.3 | 160.9 | 1.24× |
| 4 | 510.8 | 358.1 | 1.43× |
| 12 | 479.4 | 291.4 | 1.65× |

At 12 streams rc_engine uses nearly all of the card's 16 GB, so the result depends on what else holds video memory: 627 tok/s in a run where the whole model fit on the GPU, 479 tok/s (the median above) with one layer's experts left in system RAM. The previous version of this table reported 580.4 vs 279.1.

An earlier version of this table (685 vs 308 at 12 streams) used near-identical prompts, which let the streams share the same experts and raised both engines' figures.

## Other results

- **Correctness.** Perplexity within about 1% of llama.cpp on all five models tested. Free-running greedy output matches llama.cpp for 59–74% of the first 64 tokens; llama.cpp compared with itself (flash attention on and off) matches for 65–83%.
- **Failover.** Killing a node mid-stream: the stream finished with no missing or duplicated tokens in 3 of 3 runs on the CUDA node and in 2 of 3 valid runs on the Metal node. The remaining Metal run lost part of the answer.
- **Prompt cache** (Qwen3.6-35B, a 4.5k-token document). Requests that share the document and differ in their last question: 0.083 s each after the first, against llama.cpp's 0.178 s. A conversation evicted from its slot resumes from host RAM in 0.118 s, against llama.cpp's 0.381 s and about 1.0 s for a full re-prefill. A new session with the same 4.5k-token system prompt and a different ~400-token first question: 0.256 s, against llama.cpp's 0.181 s (both about 1.2 s without the cache). An earlier version of this page listed shared-prefix reuse as missing.
- **Stability.** A 30-minute soak at 4 concurrent streams completed 3,748 requests with no errors. Requests at 8k, 16k and 32k context all passed.
- **Prefill at 32k context** (RTX 5080, Qwen3.6-35B-A3B IQ3_S, one sequence, three timed runs). Time to first token for 1.8k / 7.7k / 29k-token prompts: rc_engine 0.43 / 1.68 / 6.67 s, llama.cpp 0.44 / 1.70 / 6.58 s, within about 1.5% of each other. The previous version of this page (1.8k / 6.6k / 29k tokens) had rc_engine 3–5% ahead. An earlier version of this page reported rc_engine 1.2–1.4× slower here; that run gave rc_engine four 32k-token sequences of cache against llama.cpp's one.

- **Time to first token for a single short request:** 77 vs 113 ms median (256-token answers to distinct ~130-token prompts). An earlier version of this page listed this as about 15 ms slower than llama.cpp.
- **Gateway.** Requests sent through rc_engine's cluster gateway instead of straight to the node: 172 vs 170 tok/s and 48 vs 47 ms to the first token for a single stream; no measurable overhead.

## Images (RTX 5080, Qwen3.6-35B-A3B IQ3_S with its own vision encoder, F16)

One sequence, five timed runs, median. Each run uses a new copy of the image, so neither engine's cache holds it. Both engines produce the same number of image tokens.

| | rc_engine | llama.cpp |
| --- | --- | --- |
| First request, new 640×488 JPEG (323 prompt tokens) | 0.20 s | 0.34 s |
| First request, new 1280×640 PNG (823 prompt tokens) | 1.03 s | 0.53 s |
| Follow-up question on the same image (JPEG / PNG) | 0.14 / 0.23 s | 0.22 / 0.22 s |
| Decode in an image conversation | 210.9 tok/s | 172.8 tok/s |

## Where rc_engine is slower (in progress)

- **First request with a larger image:** about 2× slower (1.03 vs 0.53 s for a 1280×640 PNG). rc_engine's image encoder does not use tensor cores yet.
- **A new session that repeats a long system prompt:** 0.256 vs 0.181 s (see Prompt cache).
- **Prefill of very long prompts:** 6.67 vs 6.58 s at 29k tokens (about 1.4%).
