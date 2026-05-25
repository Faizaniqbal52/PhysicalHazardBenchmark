"""
PRISM Pilot: 10 scenarios × 8 conditions = 80 API calls
Prints raw model outputs and scores to terminal and saves to CSV.

Usage:
    export GEMINI_API_KEY="your_key"
    python pilot.py
"""

import os, sys, json, csv, time, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scoring import score_response

API_KEY  = os.environ.get("GEMINI_API_KEY", "")
MODEL    = "gemini-2.0-flash"
API_URL  = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
DELAY    = 1.2

CONDITIONS_ORDER = [
    "C1_baseline", "C2_raw", "C3_plaintext",
    "C4_prism", "C5_prose_matched",
    "A1_no_urgency", "A2_no_thresholds", "A3_no_correlation"
]

PILOT_SCENARIOS = [
    "S003",   # multi_hazard  — extreme values, all sensors danger
    "S005",   # toxic_gas     — CO >> NIOSH IDLH
    "S007",   # safe_normal   — everything safe, should not alarm
    "S009",   # fire_risk     — high temp + smoke
    "S016",   # heat_stress   — temp 40°C, high humidity
    "S001",   # structural    — vibration danger
    "S002",   # air_quality   — PM2.5 elevated
    "S010",   # fire_risk     — second fire instance
    "S017",   # heat_stress   — second heat instance
    "S025",   # safe_normal   — second safe instance (false-positive check)
]

FIELDNAMES = [
    "scenario_id", "scenario_type", "condition",
    "hazard_detected_gt", "urgency_level",
    "hazard_detection", "action_recommendation",
    "threshold_citation", "specificity", "total_score",
    "response_text", "api_error",
]


def call_gemini(prompt_text):
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, json=payload, timeout=30)
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip(), None
            elif r.status_code == 429:
                time.sleep(60 * (attempt + 1))
            else:
                return "", f"HTTP {r.status_code}"
        except Exception as e:
            if attempt == 2:
                return "", str(e)
            time.sleep(5)
    return "", "max retries"


def run_pilot():
    if not API_KEY:
        print("ERROR: set GEMINI_API_KEY")
        sys.exit(1)

    with open("/home/claude/prism_eval/data/prompts.json") as f:
        all_prompts = json.load(f)

    # Index prompts by (scenario_id, condition)
    prompt_index = {(p["scenario_id"], p["condition"]): p for p in all_prompts}

    # Verify pilot scenario IDs exist
    with open("/home/claude/prism_eval/data/scenarios.json") as f:
        all_scenarios = json.load(f)
    sid_set = {s["scenario_id"] for s in all_scenarios}
    valid_pilots = [sid for sid in PILOT_SCENARIOS if sid in sid_set]
    if len(valid_pilots) < len(PILOT_SCENARIOS):
        missing = set(PILOT_SCENARIOS) - sid_set
        print(f"WARNING: scenario IDs not found: {missing}")
        print(f"Using first 10 available scenarios instead")
        valid_pilots = [s["scenario_id"] for s in all_scenarios[:10]]

    scenario_map = {s["scenario_id"]: s for s in all_scenarios}
    total_calls  = len(valid_pilots) * len(CONDITIONS_ORDER)
    print(f"\nPILOT: {len(valid_pilots)} scenarios × {len(CONDITIONS_ORDER)} conditions = {total_calls} API calls")
    print(f"Model: {MODEL}  |  Delay: {DELAY}s between calls")
    print("="*70)

    results = []
    call_n  = 0

    for sid in valid_pilots:
        s = scenario_map[sid]
        print(f"\n{'─'*70}")
        print(f"SCENARIO {sid} | {s['scenario_type']} | hazard={s['hazard_detected']} | urgency={s['urgency_level']}")
        print(f"{'─'*70}")

        for cond in CONDITIONS_ORDER:
            key = (sid, cond)
            if key not in prompt_index:
                print(f"  [{cond:<22}]  SKIP — prompt not found")
                continue

            p = prompt_index[key]
            call_n += 1
            print(f"\n  [{cond:<22}]  call {call_n}/{total_calls}")

            response, error = call_gemini(p["prompt"])

            if error:
                print(f"  ERROR: {error}")
                scores = {"hazard_detection":0,"action_recommendation":0,
                          "threshold_citation":0,"specificity":0,"total_score":0}
            else:
                gt = {"hazard_detected": p["hazard_detected"],
                      "correct_action":  p["correct_action"]}
                scores = score_response(response, gt)

                # Print response + scores
                print(f"  Score: {scores['total_score']:.2f}/4  "
                      f"[hazard={scores['hazard_detection']:.1f} "
                      f"action={scores['action_recommendation']:.2f} "
                      f"thresh={scores['threshold_citation']:.1f} "
                      f"spec={scores['specificity']:.1f}]")
                print(f"  Response: {response[:300].replace(chr(10),' ')}"
                      + ("..." if len(response) > 300 else ""))

            results.append({
                "scenario_id":       sid,
                "scenario_type":     s["scenario_type"],
                "condition":         cond,
                "hazard_detected_gt":p["hazard_detected"],
                "urgency_level":     p["urgency_level"],
                **scores,
                "response_text":     response.replace("\n", " ") if not error else "",
                "api_error":         error or "",
            })
            time.sleep(DELAY)

    # Save to CSV
    out_path = "/home/claude/prism_eval/results/pilot_results.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    # Summary table
    print(f"\n\n{'='*70}")
    print("PILOT SUMMARY — Mean scores by condition")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'N':>3} {'Total':>6} {'Hazard':>7} {'Action':>7} {'Thresh':>7} {'Spec':>6}")
    print(f"{'─'*25} {'─'*3} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*6}")

    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        if not r["api_error"]:
            by_cond[r["condition"]].append(r)

    for cond in CONDITIONS_ORDER:
        rows = by_cond[cond]
        if not rows:
            continue
        n   = len(rows)
        tot = sum(r["total_score"] for r in rows) / n
        haz = sum(r["hazard_detection"] for r in rows) / n
        act = sum(r["action_recommendation"] for r in rows) / n
        thr = sum(r["threshold_citation"] for r in rows) / n
        spc = sum(r["specificity"] for r in rows) / n
        print(f"  {cond:<23} {n:>3} {tot:>6.3f} {haz:>7.3f} {act:>7.3f} {thr:>7.3f} {spc:>6.3f}")

    print(f"\nRaw results → {out_path}")
    return results
