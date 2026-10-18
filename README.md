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

Aggregate tok/s, 256 tokens per stream.

| Streams | rc_engine | llama.cpp |
| --- | --- | --- |
| 1 | 193.6 | 159.7 |
| 4 | 484.8 | 366.3 |
| 12 | 685.3 | 307.8 |

## Other results

- **Correctness.** Perplexity within about 1% of llama.cpp on all five models tested. Free-running greedy output matches llama.cpp for 59–74% of the first 64 tokens; llama.cpp compared with itself (flash attention on and off) matches for 65–83%.
- **Failover.** Killing a node mid-stream: the stream finished with no missing or duplicated tokens in 3 of 3 runs on the CUDA node and in 2 of 3 valid runs on the Metal node. The remaining Metal run lost part of the answer.
- **Cache restore.** A conversation evicted from its slot resumes from host RAM in 0.063 s, against 0.98 s for a full re-prefill (4.5k-token prompt, Qwen3.6-35B).
- **Stability.** A 30-minute soak at 4 concurrent streams completed 3,748 requests with no errors. Requests at 8k, 16k and 32k context all passed.

## Where rc_engine is slower or lacks a feature (in progress)

Both items are being worked on; this section will be updated with new measurements.

- Prefill of the 35B model at 32k context is 1.2–1.4× slower than llama.cpp.
- Requests that share a prompt prefix but differ at the end get no cache reuse; llama.cpp reuses them.
