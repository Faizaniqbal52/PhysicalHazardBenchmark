"""
C6: Inference-Forcing Structured  (raw values + thresholds + standards, zero pre-classification)
C7: Inference-Forcing Prose       (same information as C6, expressed as natural prose)

What both contain:
  - Raw sensor value + unit
  - Safety threshold value + unit
  - Standard name (NIOSH IDLH, WHO 2021, OSHA, etc.)
  - Temporal context (single snapshot)

What both explicitly omit:
  - URGENCY / urgency-level labels
  - EVENT_TAGS / event classifications
  - SAFE / WARNING / DANGER / CAUTION status labels
  - normalized multiplier values
  - Any hazard classification of any kind
  - PRISM tags, middleware references, framing

The only difference between C6 and C7 is FORMAT:
  C6 uses a structured tabular layout (no PRISM branding)
  C7 expresses identical information as flowing prose

If C6 > C7, structured format aids inference independently of content.
If C6 ≈ C7, format doesn't matter — only information does.
This is the cleanest test of PRISM's structural claim.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompt_formatter import SAFETY_THRESHOLDS, TASK_QUESTION

SENSOR_KEYS = [
    "co_ppm", "co2_ppm", "pm25_ugm3", "voc_ppb",
    "temperature_c", "humidity_pct", "sound_db", "vibration_ms2"
]

# Human-readable sensor names for prose
SENSOR_NAMES = {
    "co_ppm":        ("Carbon monoxide (CO)",            "ppm",                   "parts per million"),
    "co2_ppm":       ("Carbon dioxide (CO2)",            "ppm",                   "parts per million"),
    "pm25_ugm3":     ("Particulate matter PM2.5",        "µg/m³",                 "micrograms per cubic metre"),
    "voc_ppb":       ("Volatile organic compounds (VOC)","ppb",                   "parts per billion"),
    "temperature_c": ("Air temperature",                 "°C",                    "degrees Celsius"),
    "humidity_pct":  ("Relative humidity",               "%",                     "percent"),
    "sound_db":      ("Sound pressure level",            "dB",                    "decibels"),
    "vibration_ms2": ("Vibration acceleration",          "m/s²",                  "metres per second squared"),
}


# ── C6: Inference-Forcing Structured ─────────────────────────────────────────
def fmt_c6_structured(scenario):
    """
    Structured table layout. No PRISM branding. No status labels.
    Just: sensor | measured value | safety threshold | standard.
    Model must compare value to threshold itself.
    """
    rows = []
    for key in SENSOR_KEYS:
        if key not in scenario or key not in SAFETY_THRESHOLDS:
            continue
        val   = scenario[key]
        t     = SAFETY_THRESHOLDS[key]
        name  = SENSOR_NAMES[key][0]
        unit  = SENSOR_NAMES[key][1]
        rows.append(
            f"  {name:<35} {val} {unit:<6}  |  threshold: {t['danger']} {unit}  ({t['standard']})"
        )

    sensor_block = "\n".join(rows)
    temporal = "Single-frame measurement. No prior readings available."

    return (
        f"You are an AI assistant. Below are raw sensor measurements "
        f"from an environmental monitoring system, along with the relevant "
        f"safety thresholds from international standards.\n\n"
        f"SENSOR MEASUREMENTS:\n{sensor_block}\n\n"
        f"NOTE: {temporal}\n\n"
        f"Question: {TASK_QUESTION}"
    )


# ── C7: Inference-Forcing Prose ───────────────────────────────────────────────
def fmt_c7_prose(scenario):
    """
    Identical information to C6 expressed as flowing prose.
    No status labels, no urgency, no pre-classification.
    Model must compare value to threshold itself.
    """
    sentences = []
    for key in SENSOR_KEYS:
        if key not in scenario or key not in SAFETY_THRESHOLDS:
            continue
        val          = scenario[key]
        t            = SAFETY_THRESHOLDS[key]
        long_name    = SENSOR_NAMES[key][0]
        long_unit    = SENSOR_NAMES[key][2]
        short_unit   = SENSOR_NAMES[key][1]
        sentences.append(
            f"{long_name} is currently {val} {short_unit}; "
            f"the {t['standard']} safety threshold for this parameter is {t['danger']} {short_unit}."
        )

    temporal = (
        "This represents a single-frame measurement. "
        "No prior readings are available for comparison."
    )

    body = " ".join(sentences) + " " + temporal

    return (
        f"You are an AI assistant. The following describes current environmental "
        f"sensor readings along with the applicable safety thresholds from "
        f"international standards.\n\n"
        f"CURRENT READINGS:\n{body}\n\n"
        f"Question: {TASK_QUESTION}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PARITY VERIFIER for C6/C7
# ═══════════════════════════════════════════════════════════════════════════════

def extract_c6c7_facts(scenario):
    """
    The fact set for C6/C7. Strictly smaller than C4/C5.
    Contains: value, threshold, standard, temporal.
    Does NOT contain: norm, status, urgency, event_tags.
    """
    facts = set()
    for key in SENSOR_KEYS:
        if key not in scenario or key not in SAFETY_THRESHOLDS:
            continue
        t = SAFETY_THRESHOLDS[key]
        facts.add((f"{key}.value",     str(scenario[key])))
        facts.add((f"{key}.threshold", str(t["danger"])))
        facts.add((f"{key}.standard",  t["standard"]))
    facts.add(("temporal", "single_snapshot"))
    return facts


def verify_c6c7_fact(fact_type, fact_val, text):
    """Check one fact is present in text. Returns True/False."""
    text_lower = text.lower()
    if ".value" in fact_type or ".threshold" in fact_type:
        return str(fact_val) in text
    elif ".standard" in fact_type:
        return fact_val.lower() in text_lower
    elif fact_type == "temporal":
        return "single" in text_lower and ("frame" in text_lower or "snapshot" in text_lower or "reading" in text_lower)
    return False


def check_no_preclass(text):
    """
    Verify no pre-classification artifacts leaked in.
    Returns list of violations found.
    """
    forbidden_patterns = [
        # Status labels
        (r'\bDANGER\b',                 'status label: DANGER'),
        (r'\bWARNING\b',                'status label: WARNING'),
        (r'\bCAUTION\b',                'status label: CAUTION'),
        (r'\bSAFE\b(?!\s+range\s+for)', 'status label: SAFE (standalone)'),
        # Urgency labels
        (r'\bCRITICAL\b',               'urgency: CRITICAL'),
        (r'\bAMBIENT\b',                'urgency: AMBIENT'),
        (r'\bCONTEXTUAL\b',             'urgency: CONTEXTUAL'),
        # Event tags
        (r'FIRE_RISK',                  'event_tag: FIRE_RISK'),
        (r'TOXIC_GAS',                  'event_tag: TOXIC_GAS'),
        (r'HEAT_STRESS',                'event_tag: HEAT_STRESS'),
        (r'AIR_QUALITY',                'event_tag: AIR_QUALITY'),
        (r'CONDITIONS_NORMAL',          'event_tag: CONDITIONS_NORMAL'),
        # PRISM artifacts
        (r'PRISM',                      'PRISM reference'),
        (r'normalized=',                'normalized= field'),
        (r'status=',                    'status= field'),
        (r'\[PRISM_PERCEPTUAL',         'PRISM block tag'),
        (r'EVENT_TAGS:',                'EVENT_TAGS header'),
        (r'URGENCY:',                   'URGENCY header'),
        # Normalized multiplier (e.g. "1.953 times")
        (r'\d+\.\d+\s+times\s+the',     'normalized multiplier phrase'),
        # In danger zone / safe range status phrases
        (r'in the danger zone',         'status prose: danger zone'),
        (r'within the safe range',      'status prose: safe range'),
        (r'approaching the.*threshold', 'status prose: approaching threshold'),
    ]
    violations = []
    for pattern, label in forbidden_patterns:
        if re.search(pattern, text):
            violations.append(label)
    return violations


def verify_pair(scenario, verbose=False):
    """Full parity + no-preclass verification for one scenario."""
    c6 = fmt_c6_structured(scenario)
    c7 = fmt_c7_prose(scenario)
    expected = extract_c6c7_facts(scenario)

    results = {}
    for label, text in [("C6", c6), ("C7", c7)]:
        missing   = set()
        present   = set()
        for fact in expected:
            if verify_c6c7_fact(fact[0], fact[1], text):
                present.add(fact)
            else:
                missing.add(fact)
        violations = check_no_preclass(text)
        parity     = len(present) / len(expected)
        results[label] = {
            "parity":     round(parity, 3),
            "present":    len(present),
            "missing":    len(missing),
            "missing_facts": [f"{t}={v}" for t,v in sorted(missing)],
            "violations": violations,
            "pass":       parity >= 1.0 and len(violations) == 0,
        }
        if verbose and not results[label]["pass"]:
            print(f"  {label} FAIL — parity={parity:.3f}  violations={violations}")
            print(f"    missing: {[f'{t}={v}' for t,v in sorted(missing)][:5]}")
    return results


# ── Main: generate, verify, save ─────────────────────────────────────────────
if __name__ == "__main__":
    import csv

    with open("/home/claude/prism_eval/data/scenarios.json") as f:
        scenarios = json.load(f)

    print("Verifying C6/C7 across all 100 scenarios...")
    all_results = []
    c6_pass = c7_pass = 0
    c6_violations_total = c7_violations_total = 0

    for s in scenarios:
        r = verify_pair(s, verbose=True)
        if r["C6"]["pass"]: c6_pass += 1
        if r["C7"]["pass"]: c7_pass += 1
        c6_violations_total += len(r["C6"]["violations"])
        c7_violations_total += len(r["C7"]["violations"])
        all_results.append((s, r))

    print(f"\n{'='*55}")
    print(f"  C6/C7 VERIFICATION RESULTS")
    print(f"{'='*55}")
    print(f"  C6 pass: {c6_pass}/100  violations: {c6_violations_total}")
    print(f"  C7 pass: {c7_pass}/100  violations: {c7_violations_total}")
    print(f"{'='*55}")

    # Build new prompts and append to prompts.json
    with open("/home/claude/prism_eval/data/prompts.json") as f:
        existing = json.load(f)

    # Remove stale C6/C7
    existing = [p for p in existing
                if p["condition"] not in ("C6_structured_infer", "C7_prose_infer")]

    new_prompts = []
    for s in scenarios:
        for cond, fn in [("C6_structured_infer", fmt_c6_structured),
                         ("C7_prose_infer",       fmt_c7_prose)]:
            new_prompts.append({
                "scenario_id":    s["scenario_id"],
                "condition":      cond,
                "scenario_type":  s["scenario_type"],
                "hazard_detected":s["hazard_detected"],
                "correct_action": s["correct_action"],
                "urgency_level":  s["urgency_level"],
                "threshold_name": s["threshold_name"],
                "threshold_value":s["threshold_value"],
                "threshold_unit": s["threshold_unit"],
                "prompt":         fn(s),
            })

    all_prompts = existing + new_prompts
    with open("/home/claude/prism_eval/data/prompts.json", "w") as f:
        json.dump(all_prompts, f, indent=2)

    print(f"\nprompts.json now has {len(all_prompts)} entries")

    # Spot-check: print one hazard + one safe scenario
    for sid in ["S003", "S007"]:
        s = next(x for x in scenarios if x['scenario_id'] == sid)
        r = verify_pair(s)
        print(f"\n{'─'*60}")
        print(f"SPOT-CHECK {sid} ({s['scenario_type']})")
        for label in ["C6", "C7"]:
            ri = r[label]
            print(f"  {label}: parity={ri['parity']}  present={ri['present']}  "
                  f"missing={ri['missing']}  violations={ri['violations']}")

    # Export to CSV for inspection
    csv_path = "/home/claude/prism_eval/data/c6c7_prompts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scenario_id","scenario_type","condition",
            "hazard_detected","parity","violations","prompt"
        ])
        w.writeheader()
        for s, r in all_results:
            for cond, fn, label in [
                ("C6_structured_infer", fmt_c6_structured, "C6"),
                ("C7_prose_infer",      fmt_c7_prose,      "C7"),
            ]:
                w.writerow({
                    "scenario_id":    s["scenario_id"],
                    "scenario_type":  s["scenario_type"],
                    "condition":      cond,
                    "hazard_detected":s["hazard_detected"],
                    "parity":         r[label]["parity"],
                    "violations":     "|".join(r[label]["violations"]),
                    "prompt":         fn(s).replace("\n", " "),
                })
    print(f"\nC6/C7 CSV → {csv_path}")
