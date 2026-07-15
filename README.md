# microgpt.rs

A tiny, readable GPT training script in pure Rust, ported from the original dependency-free Python version.

[![Stars](https://img.shields.io/github/stars/barrontang/microgpt.rs?style=for-the-badge)](https://github.com/barrontang/microgpt.rs/stargazers)
[![Forks](https://img.shields.io/github/forks/barrontang/microgpt.rs?style=for-the-badge)](https://github.com/barrontang/microgpt.rs/network/members)
[![Issues](https://img.shields.io/github/issues/barrontang/microgpt.rs?style=for-the-badge)](https://github.com/barrontang/microgpt.rs/issues)
[![License](https://img.shields.io/github/license/barrontang/microgpt.rs?style=for-the-badge)](https://github.com/barrontang/microgpt.rs/blob/master/LICENSE)

## What This Repo Is

This repo contains two versions of the same core idea:

- `microgpt.py`: the most atomic, educational Python implementation
- `microgpt.rs`: a Rust port with explicit control over memory and performance

If you want to understand how GPT training works at the algorithm level, this project is intentionally small and transparent.

## Why People Star This Project

- Clear learning value: no framework magic, just the core algorithm
- Side-by-side language comparison: Python and Rust implementations
- Great for interviews, study groups, and systems-ML discussions
- Friendly for first contributions and experimentation

If this project helps you learn, consider starring it.

## Quick Start

### Run Python version

```powershell
python microgpt.py
```

### Build and run Rust version

```powershell
rustc microgpt.rs -O -o microgpt.exe
.\microgpt.exe
```

Notes:

- `input.txt` is expected in the repo root
- If `python` is not installed, use your local Python command (for example `py` on Windows)

## Rust vs Python: Practical Advantages

Both versions are valuable. They optimize for different goals.

| Topic | Python (`microgpt.py`) | Rust (`microgpt.rs`) | Why Rust Can Win |
|---|---|---|---|
| Learning speed | Fast to read and tweak | More verbose | Python is best for first-pass learning |
| Runtime speed | Slower for heavy loops | Compiled native code | Rust can run significantly faster on CPU-bound workloads |
| Memory behavior | Dynamic objects and GC overhead | Predictable ownership + stack/heap control | Rust often reduces memory overhead and improves cache behavior |
| Safety | Runtime type errors possible | Compile-time safety checks | Rust catches more bugs before runtime |
| Deployment | Requires Python runtime | Single binary possible | Rust is easier to ship in constrained environments |
| Refactoring confidence | Flexible but easier to break silently | Strict compiler guidance | Rust compiler improves large-change reliability |
| Concurrency scaling | GIL limits CPU parallelism | Fearless concurrency model | Rust is better for future multithreaded training experiments |

## Suggested Use Pattern

1. Start with `microgpt.py` to understand the math and graph flow.
2. Move to `microgpt.rs` to explore systems-level performance and safety.
3. Benchmark both versions with identical settings and share results in an issue.

## Contributing

Contributions are welcome, especially from first-time contributors.

Good contribution ideas:

- Add benchmark scripts and report reproducible results
- Add mini tests for autograd operations (`add`, `mul`, `log`, `exp`, `relu`)
- Improve tokenizer performance without changing behavior
- Add command-line flags for hyperparameters
- Keep Python and Rust outputs comparable for the same random seed

### Commit style (recommended)

Use clear prefixes:

- `feat:` new feature
- `fix:` bug fix
- `perf:` performance improvement
- `docs:` documentation update
- `refactor:` internal cleanup without behavior change

Example:

```text
perf: speed up softmax inner loop in microgpt.rs
```

## Roadmap

- [ ] Add a simple benchmark harness (`python` vs `rust`)
- [ ] Add optional model/config arguments via CLI
- [ ] Add tiny correctness tests for core math ops
- [ ] Add CI to run format/lint/basic checks
- [ ] Add performance notes and profiling tips

## Attribution

Original minimalist concept and style inspiration by Andrej Karpathy's educational work.

---

If you build something cool from this code, open an issue and share it. That is the fastest way to grow this project and attract more contributors.
