"""
PRISM Evaluation Pipeline — F4 three-question split
One API call per question per scenario-condition pair.
Scores Q1/Q2/Q3 independently, stored as separate columns.

Usage:
    export GEMINI_API_KEY="your_key_here"
    python eval_pipeline.py
"""

import os, json, csv, time, sys, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scoring import score_q1, score_q2, score_q3

API_KEY     = os.environ.get("GEMINI_API_KEY", "")
MODEL       = "gemini-2.0-flash"
API_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
DELAY_SEC   = 1.2
MAX_RETRIES = 3

_BASE        = Path(__file__).resolve().parent.parent
PROMPTS_PATH = str(_BASE / "data"    / "prompts.json")
RESULTS_PATH = str(_BASE / "results" / "raw_results.csv")

# Q1 is the primary causal measure; Q2/Q3 are secondary
FIELDNAMES = [
    "scenario_id", "condition", "scenario_type",
    "hazard_detected", "correct_action", "urgency_level",
    "threshold_name", "threshold_value", "threshold_unit",
    # Raw responses per question
    "response_q1", "response_q2", "response_q3",
    # Scores per question (each 0–1)
    "score_q1",   # threshold arithmetic — PRIMARY
    "score_q2",   # hazard classification
    "score_q3",   # action recommendation
    # Errors
    "error_q1", "error_q2", "error_q3",
]

# ── Question extraction ───────────────────────────────────────────────────────
# The prompt contains "Q1. ...\n\nQ2. ...\n\nQ3. ..."
# We strip the context block and ask each question independently,
# keeping the context block so the model has the sensor data.

def split_prompt_questions(full_prompt: str):
    """
    Split the combined prompt into three separate prompts, one per question.
    Each carries the full sensor context + only its own question.
    """
    # Separate context (everything before Q1) from the question block
    q_marker = "Q1."
    if q_marker not in full_prompt:
        # Fallback: old single-question format — wrap as Q1 only
        return [full_prompt, None, None]

    ctx_end   = full_prompt.index(q_marker)
    context   = full_prompt[:ctx_end].rstrip()

    # Extract individual question texts
    import re
    q_texts = re.findall(r'Q\d\.\s.*?(?=\n\nQ\d\.|$)', full_prompt[ctx_end:], re.DOTALL)
    q_texts = [q.strip() for q in q_texts]

    prompts = []
    for i, q in enumerate(q_texts[:3]):
        prompts.append(f"{context}\n\n{q}")
    # Pad to exactly 3
    while len(prompts) < 3:
        prompts.append(None)
    return prompts


# ── Gemini call ───────────────────────────────────────────────────────────────
def call_gemini(prompt_text: str):
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 600},
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(API_URL, json=payload, timeout=30)
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip(), None
            elif r.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"\n  Rate limit. Waiting {wait}s ...", flush=True)
                time.sleep(wait)
            else:
                return "", f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                return "", str(e)
    return "", "max retries"


# ── Resume support ────────────────────────────────────────────────────────────
def load_completed(path):
    completed = set()
    if not Path(path).exists():
        return completed
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((row["scenario_id"], row["condition"]))
    return completed


# ── Main ──────────────────────────────────────────────────────────────────────
def run_evaluation():
    if not API_KEY:
        print("ERROR: set GEMINI_API_KEY")
        sys.exit(1)

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        prompts = json.load(f)

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    completed   = load_completed(RESULTS_PATH)
    write_hdr   = not Path(RESULTS_PATH).exists()
    outfile     = open(RESULTS_PATH, "a", newline="", encoding="utf-8")
    writer      = csv.DictWriter(outfile, fieldnames=FIELDNAMES)
    if write_hdr:
        writer.writeheader()

    total  = len(prompts)
    errors = 0

    for i, p in enumerate(prompts):
        key = (p["scenario_id"], p["condition"])
        if key in completed:
            continue

        pct = (i + 1) / total * 100
        print(f"[{i+1:3d}/{total}] {p['scenario_id']} | {p['condition']:<24}| {pct:.0f}%",
              end=" ", flush=True)

        q_prompts = split_prompt_questions(p["prompt"])
        gt = {
            "hazard_detected": p["hazard_detected"],
            "correct_action":  p["correct_action"],
        }

        r1, e1 = call_gemini(q_prompts[0]) if q_prompts[0] else ("", "no Q1 prompt")
        time.sleep(DELAY_SEC)
        r2, e2 = call_gemini(q_prompts[1]) if q_prompts[1] else ("", "no Q2 prompt")
        time.sleep(DELAY_SEC)
        r3, e3 = call_gemini(q_prompts[2]) if q_prompts[2] else ("", "no Q3 prompt")
        time.sleep(DELAY_SEC)

        s1 = score_q1(r1, gt) if not e1 else 0.0
        s2 = score_q2(r2, gt) if not e2 else 0.0
        s3 = score_q3(r3, gt) if not e3 else 0.0

        if e1 or e2 or e3:
            errors += 1

        print(f"Q1={s1:.2f} Q2={s2:.2f} Q3={s3:.2f}", flush=True)

        writer.writerow({
            "scenario_id":    p["scenario_id"],
            "condition":      p["condition"],
            "scenario_type":  p["scenario_type"],
            "hazard_detected":p["hazard_detected"],
            "correct_action": p["correct_action"],
            "urgency_level":  p["urgency_level"],
            "threshold_name": p["threshold_name"],
            "threshold_value":p["threshold_value"],
            "threshold_unit": p["threshold_unit"],
            "response_q1":    r1.replace("\n", " "),
            "response_q2":    r2.replace("\n", " "),
            "response_q3":    r3.replace("\n", " "),
            "score_q1":       s1,
            "score_q2":       s2,
            "score_q3":       s3,
            "error_q1":       e1 or "",
            "error_q2":       e2 or "",
            "error_q3":       e3 or "",
        })
        outfile.flush()

    outfile.close()
    print(f"\nDone. errors={errors} → {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
