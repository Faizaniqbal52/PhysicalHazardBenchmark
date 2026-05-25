# PhysicalHazardBenchmark

**Benchmarking Large Language Models on Multi-Sensor Physical Hazard Assessment**

A reproducible benchmark for evaluating how large language models assess
physical environmental safety data across multi-sensor scenarios grounded in
internationally recognised safety standards (NIOSH, OSHA, WHO, ASHRAE, ISO).

---

## What This Is

PhysicalHazardBenchmark evaluates whether LLMs produce appropriate precautionary
signals when presented with environmental sensor data. It tests three distinct
reasoning requirements:

| Category | Description | Scenarios |
|----------|-------------|-----------|
| **A — Multi-sensor Joint Assessment** | Multiple sensors simultaneously elevated below individual limits (OSHA additive index Em > 1.0) | 20 |
| **B — Proportionality** | Single sensor exceeds threshold at magnitudes from 1% to 400% | 20 |
| **C — Pattern Disambiguation** | Correct hazard-type identification from sensor patterns (e.g. heat vs fire) | 20 |

**Total: 60 scenarios × 2 prompt formats × 5 models × 3 runs = 1,800 API calls**

---

## Models Evaluated

| Model | Provider | API |
|-------|----------|-----|
| ChatGPT-4o | OpenAI | `gpt-4o` |
| Gemini 2.5 Flash | Google | `gemini-2.5-flash` |
| DeepSeek | DeepSeek | `deepseek-chat` |
| Kimi | Moonshot AI | `moonshot-v1-auto` |
| Llama 3.1 8B Instant | Groq | `llama-3.1-8b-instant` |

---

## Repository Contents

```
PhysicalHazardBenchmark/
├── README.md
├── LICENSE
├── data/
│   ├── scenarios.json          # All 60 scenario definitions with sensor values
│   ├── ground_truth.json       # Ground truth labels and correct actions
│   └── benchmark_results.csv  # Raw model responses (1,800 rows)
├── prompts/
│   ├── c6_structured.py        # Structured tabular prompt formatter
│   └── c7_prose.py             # Plain prose prompt formatter
├── scoring/
│   ├── scorer.py               # Q1, Q2, Q3 scoring rubric
│   └── scorer_test.py          # 46-case scorer validation suite
├── experiments/
│   ├── benchmark_experiment.py # Full automation script (1,800 API calls)
│   └── pilot.py                # 8-scenario pilot runner
├── analysis/
│   ├── score_results.py        # Scores raw CSV → scored JSON
│   ├── tables.py               # Generates all result tables
│   ├── figures.py              # Generates all 6 paper figures
│   └── statistics.py           # Wilcoxon, McNemar, Cohen's d
└── paper/
    └── llm_hazard_benchmark.tex  # LaTeX source
```

---

## How to Reproduce Results

### 1. Install dependencies

```bash
pip install openai google-generativeai requests scipy numpy matplotlib
```

### 2. Set API keys

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export DEEPSEEK_API_KEY=sk-...
export KIMI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...
```

### 3. Run the benchmark

```bash
python experiments/benchmark_experiment.py
# Output: data/benchmark_results.csv (1,800 rows)
# Runtime: ~4 hours depending on rate limits
# Checkpoint saved every 5 calls — safe to resume after interruption
```

### 4. Score results

```bash
python analysis/score_results.py
# Input:  data/benchmark_results.csv
# Output: data/scored_results.json
```

### 5. Generate tables and figures

```bash
python analysis/tables.py      # Prints all result tables
python analysis/figures.py     # Saves figures to figures/
python analysis/statistics.py  # Prints Wilcoxon and Cohen's d results
```

### Using pre-computed results

To skip re-running the API calls, use the included
`data/benchmark_results.csv` and start from step 4.

---

## Key Results

| Finding | Result |
|---------|--------|
| Category A Q1 (multi-sensor) | 0.000 – 0.292 across all models |
| Category B Q1 (single-sensor) | 0.975 – 1.000 across all models |
| Format effect (ChatGPT-4o) | Prose significantly better than structured (p=0.001) |
| Format effect (other models) | No significant difference |

---

## Scorer Note

A question-echo artefact was identified and corrected post-hoc. Gemini 2.5 Flash
consistently reproduced the question text before answering, which triggered false
Q1 credits in the original scorer. The corrected scorer strips question-echo
prefixes before scoring. All results in the paper and this repository use the
corrected scorer. Full details in `scoring/scorer.py` and Appendix A of the paper.

---

## Citation

```bibtex
@misc{iqbal2026physicalhazard,
  author    = {Faizan Iqbal},
  title     = {Benchmarking Large Language Models on Multi-Sensor Physical
               Hazard Assessment},
  year      = {2026},
  url       = {https://github.com/Faizaniqbal52/PhysicalHazardBenchmark},
  note      = {Version 1.0}
}
```

---

## License

MIT License — see `LICENSE` for details.

---

## Contact

**Faizan Iqbal** · Lovely Professional University, India
- ORCID: [0009-0002-8998-9347](https://orcid.org/0009-0002-8998-9347)
- HuggingFace: [huggingface.co/faizaniqbal](https://huggingface.co/faizaniqbal)
- X: [@faizaniqbal__52](https://x.com/faizaniqbal__52)
