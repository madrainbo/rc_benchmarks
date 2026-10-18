# rc_benchmarks

High-level benchmark results for **rc_engine**, a Rust inference engine (CUDA and Metal kernels) for Mixture-of-Experts language models, compared with llama.cpp. The engine's source is in a private repository, available on request.

**Setup.** NVIDIA RTX 5080 16 GB (Ryzen 9 9900X, WSL2) and Apple M4 Max 64 GB. Greedy sampling, one warmup run discarded, five timed runs, median reported. Same GGUF file and context size for both engines. CUDA toolkit 13.1 (13.1.115). llama.cpp `7d56da7` on the PC, except Nemotron, which needs a newer snapshot built 2026-10-10 (0.5.0-dev, commit unknown); on the Mac, 0.5.0-dev (commit unknown).

## Single-stream decode (tok/s)

128-token prompt, 256 generated tokens.

| Machine | Model | rc_engine | llama.cpp | Ratio |
| --- | --- | --- | --- | --- |
| RTX 5080 | LFM2.5-8B-A1B Q4_K_M | 545.0 | 444.0 | 1.23× |
| RTX 5080 | Qwen3.6-35B-A3B IQ3_S | 211.6 | 175.1 | 1.21× |
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
| 1 | 187.1 | 158.6 | 1.18× |
| 4 | 454.7 | 350.5 | 1.30× |
| 12 | 580.4 | 279.1 | 2.08× |

An earlier version of this table (685 vs 308 at 12 streams) used near-identical prompts, which let the streams share the same experts and raised both engines' figures.

## Other results

- **Correctness.** Perplexity within about 1% of llama.cpp on all five models tested. Free-running greedy output matches llama.cpp for 59–74% of the first 64 tokens; llama.cpp compared with itself (flash attention on and off) matches for 65–83%.
- **Failover.** Killing a node mid-stream: the stream finished with no missing or duplicated tokens in 3 of 3 runs on the CUDA node and in 2 of 3 valid runs on the Metal node. The remaining Metal run lost part of the answer.
- **Prompt cache** (Qwen3.6-35B, a 4.5k-token document). Requests that share the document and differ in their last question: 0.086 s each after the first, against llama.cpp's 0.183 s. A conversation evicted from its slot resumes from host RAM in 0.12 s, against llama.cpp's 0.38 s and about 1.0 s for a full re-prefill. An earlier version of this page listed shared-prefix reuse as missing.
- **Stability.** A 30-minute soak at 4 concurrent streams completed 3,748 requests with no errors. Requests at 8k, 16k and 32k context all passed.
- **Prefill at 32k context** (RTX 5080, Qwen3.6-35B-A3B IQ3_S, one sequence, three timed runs). Time to first token for 1.8k / 6.6k / 29k-token prompts: rc_engine 0.44 / 1.43 / 6.74 s, llama.cpp 0.47 / 1.51 / 6.94 s. An earlier version of this page reported rc_engine 1.2–1.4× slower here; that run gave rc_engine four 32k-token sequences of cache against llama.cpp's one.

## Where rc_engine is slower (in progress)

- Time to first token for a single short request is about 15 ms slower than llama.cpp (138 vs 121 ms median, 256-token answers to distinct ~130-token prompts).
