"""
PRISM Prompt Formatter
Generates 7 prompt variants per scenario:
  C1: Baseline (no sensor data)
  C2: Raw sensor values only
  C3: Plain-text explanation
  C4: Full PRISM structured format
  A1: PRISM minus urgency tag
  A2: PRISM minus threshold normalization
  A3: PRISM minus multi-event correlation
"""

SAFETY_THRESHOLDS = {
    "co_ppm":        {"name": "CO",        "safe": 9,    "caution": 35,  "danger": 70,  "unit": "ppm",   "standard": "NIOSH IDLH"},
    "co2_ppm":       {"name": "CO₂",       "safe": 700,  "caution": 1000,"danger": 2000,"unit": "ppm",   "standard": "ASHRAE 62.1"},
    "pm25_ugm3":     {"name": "PM2.5",     "safe": 12,   "caution": 35,  "danger": 55,  "unit": "µg/m³", "standard": "WHO 2021"},
    "voc_ppb":       {"name": "VOC",       "safe": 200,  "caution": 500, "danger": 1000,"unit": "ppb",   "standard": "ASHRAE 62.1"},
    "temperature_c": {"name": "Temp",      "safe": 28,   "caution": 35,  "danger": 43,  "unit": "°C",    "standard": "OSHA heat stress"},
    "humidity_pct":  {"name": "Humidity",  "safe": 60,   "caution": 70,  "danger": 90,  "unit": "%",     "standard": "ASHRAE 55"},
    "sound_db":      {"name": "Sound",     "safe": 70,   "caution": 85,  "danger": 100, "unit": "dB",    "standard": "OSHA 1910.95"},
    "vibration_ms2": {"name": "Vibration", "safe": 0.3,  "caution": 0.5, "danger": 1.0, "unit": "m/s²",  "standard": "ISO 2631-1"},
}

def normalize(key, value):
    if key not in SAFETY_THRESHOLDS:
        return None, None
    t = SAFETY_THRESHOLDS[key]
    if value <= t["safe"]:
        level = "SAFE"
        norm = round(value / t["danger"], 3)
    elif value <= t["caution"]:
        level = "CAUTION"
        norm = round(value / t["danger"], 3)
    elif value <= t["danger"]:
        level = "WARNING"
        norm = round(value / t["danger"], 3)
    else:
        level = "DANGER"
        norm = round(min(value / t["danger"], 3.0), 3)
    return level, norm

def get_event_tags(scenario):
    tags = []
    co  = scenario.get("co_ppm", 0)
    tmp = scenario.get("temperature_c", 20)
    pm  = scenario.get("pm25_ugm3", 0)
    vib = scenario.get("vibration_ms2", 0)
    voc = scenario.get("voc_ppb", 0)
    co2 = scenario.get("co2_ppm", 400)
    snd = scenario.get("sound_db", 40)

    if co > 70 and pm > 100 and tmp > 45:
        tags.append("FIRE_RISK_HIGH")
    elif co > 70:
        tags.append("TOXIC_GAS_EXPOSURE")
    if tmp > 43:
        tags.append("HEAT_STRESS_CRITICAL")
    elif tmp > 35:
        tags.append("HEAT_STRESS_MODERATE")
    if pm > 55:
        tags.append("POOR_AIR_QUALITY_SEVERE")
    elif pm > 35:
        tags.append("POOR_AIR_QUALITY_MODERATE")
    if vib > 1.0:
        tags.append("STRUCTURAL_VIBRATION_HIGH")
    elif vib > 0.5:
        tags.append("STRUCTURAL_VIBRATION_MODERATE")
    if voc > 1000:
        tags.append("VOC_DANGEROUS")
    if co2 > 2000:
        tags.append("CO2_HIGH")
    if snd > 100:
        tags.append("NOISE_DANGEROUS")
    if not tags:
        tags.append("CONDITIONS_NORMAL")
    return tags

def get_urgency(scenario):
    stype = scenario.get("scenario_type", "")
    if stype in ("fire_risk", "toxic_gas", "heat_stress", "structural", "multi_hazard"):
        return "CRITICAL"
    elif stype == "air_quality":
        return "CONTEXTUAL"
    return "AMBIENT"


# ── F4: Three-question split ──────────────────────────────────────────────────
# Each question maps to exactly one scoring dimension.
# Q1 = threshold arithmetic   (primary causal measure for C6/C7)
# Q2 = hazard classification  (requires Q1 reasoning as input)
# Q3 = action recommendation  (protocol recall, scored separately)
#
# Presented in sequence as a numbered list so the model addresses each
# independently. "Based on your analysis above" in Q2 forces Q1 to be
# used as input rather than retrieved from memory.

TASK_QUESTION = """Answer each of the following three questions in order.

Q1. For each sensor reading provided, state whether the measured value exceeds its safety threshold, is approaching it, or is within the safe range. For any sensor that exceeds its threshold, state the measured value, the threshold value, and by how much it is exceeded.

Q2. Based on your analysis in Q1, is this environment currently safe to remain in? Answer yes or no, and identify which specific sensor readings, if any, drive your conclusion.

Q3. What immediate actions, if any, should be taken?"""


# ── Condition 1: Baseline ─────────────────────────────────────────────────────
def fmt_c1_baseline(scenario):
    return f"""You are an AI assistant. Answer the following question based on your general knowledge.

Question: {TASK_QUESTION}

Note: You have no information about the current physical environment."""


# ── Condition 2: Raw sensor values ────────────────────────────────────────────
def fmt_c2_raw(scenario):
    sensor_keys = ["temperature_c","pm25_ugm3","co_ppm","co2_ppm","voc_ppb",
                   "humidity_pct","sound_db","vibration_ms2","light_lux","heat_index_c"]
    readings = []
    for k in sensor_keys:
        if k in scenario:
            readings.append(f"{k}: {scenario[k]}")
    raw_block = "\n".join(readings)
    return f"""You are an AI assistant. Answer the following question using the sensor data provided.

SENSOR READINGS:
{raw_block}

Question: {TASK_QUESTION}"""


# ── Condition 3: Plain-text explanation ───────────────────────────────────────
def fmt_c3_plaintext(scenario):
    parts = []
    co  = scenario.get("co_ppm", 0)
    tmp = scenario.get("temperature_c", 20)
    pm  = scenario.get("pm25_ugm3", 0)
    vib = scenario.get("vibration_ms2", 0)
    voc = scenario.get("voc_ppb", 0)
    co2 = scenario.get("co2_ppm", 400)
    snd = scenario.get("sound_db", 40)
    hum = scenario.get("humidity_pct", 50)

    parts.append(f"The current temperature is {tmp}°C with {hum}% humidity.")
    parts.append(f"Carbon monoxide level is {co} ppm.")
    if co2:
        parts.append(f"Carbon dioxide level is {co2} ppm.")
    parts.append(f"PM2.5 particulate matter is {pm} µg/m³.")
    parts.append(f"VOC concentration is {voc} ppb.")
    parts.append(f"Sound level is {snd} dB.")
    parts.append(f"Vibration is {vib} m/s².")

    description = " ".join(parts)
    return f"""You are an AI assistant. Answer the following question using the environmental description provided.

CURRENT CONDITIONS:
{description}

Question: {TASK_QUESTION}"""


# ── Condition 4: Full PRISM structured format ─────────────────────────────────
def fmt_c4_prism(scenario):
    sensor_keys = ["co_ppm","co2_ppm","pm25_ugm3","voc_ppb","temperature_c",
                   "humidity_pct","sound_db","vibration_ms2"]
    readings = []
    for k in sensor_keys:
        if k not in scenario:
            continue
        val = scenario[k]
        if k not in SAFETY_THRESHOLDS:
            continue
        t = SAFETY_THRESHOLDS[k]
        level, norm = normalize(k, val)
        readings.append(
            f"  {t['name']}: {val} {t['unit']} | normalized={norm} | status={level} "
            f"| threshold={t['danger']} {t['unit']} ({t['standard']})"
        )
    sensor_block = "\n".join(readings)
    tags = get_event_tags(scenario)
    urgency = get_urgency(scenario)
    tag_str = ", ".join(tags)

    return f"""[PRISM_PERCEPTUAL_INPUT]
URGENCY: {urgency}
EVENT_TAGS: {tag_str}
SENSOR_READINGS:
{sensor_block}
TEMPORAL_CONTEXT: Single-frame snapshot. No prior perceptual history in this session.
[/PRISM_PERCEPTUAL_INPUT]

You are an AI assistant receiving structured real-time physical sensor context via PRISM middleware.

Question: {TASK_QUESTION}"""


# ── Ablation 1: PRISM minus urgency tag ───────────────────────────────────────
def fmt_a1_no_urgency(scenario):
    sensor_keys = ["co_ppm","co2_ppm","pm25_ugm3","voc_ppb","temperature_c",
                   "humidity_pct","sound_db","vibration_ms2"]
    readings = []
    for k in sensor_keys:
        if k not in scenario:
            continue
        val = scenario[k]
        if k not in SAFETY_THRESHOLDS:
            continue
        t = SAFETY_THRESHOLDS[k]
        level, norm = normalize(k, val)
        readings.append(
            f"  {t['name']}: {val} {t['unit']} | normalized={norm} | status={level} "
            f"| threshold={t['danger']} {t['unit']} ({t['standard']})"
        )
    sensor_block = "\n".join(readings)
    tags = get_event_tags(scenario)
    tag_str = ", ".join(tags)

    return f"""[PRISM_PERCEPTUAL_INPUT]
EVENT_TAGS: {tag_str}
SENSOR_READINGS:
{sensor_block}
TEMPORAL_CONTEXT: Single-frame snapshot. No prior perceptual history in this session.
[/PRISM_PERCEPTUAL_INPUT]

You are an AI assistant receiving structured real-time physical sensor context via PRISM middleware.

Question: {TASK_QUESTION}"""


# ── Ablation 2: PRISM minus threshold normalization ───────────────────────────
def fmt_a2_no_thresholds(scenario):
    sensor_keys = ["co_ppm","co2_ppm","pm25_ugm3","voc_ppb","temperature_c",
                   "humidity_pct","sound_db","vibration_ms2"]
    readings = []
    for k in sensor_keys:
        if k not in scenario:
            continue
        val = scenario[k]
        if k not in SAFETY_THRESHOLDS:
            continue
        t = SAFETY_THRESHOLDS[k]
        readings.append(f"  {t['name']}: {val} {t['unit']}")
    sensor_block = "\n".join(readings)
    tags = get_event_tags(scenario)
    urgency = get_urgency(scenario)
    tag_str = ", ".join(tags)

    return f"""[PRISM_PERCEPTUAL_INPUT]
URGENCY: {urgency}
EVENT_TAGS: {tag_str}
SENSOR_READINGS:
{sensor_block}
TEMPORAL_CONTEXT: Single-frame snapshot.
[/PRISM_PERCEPTUAL_INPUT]

You are an AI assistant receiving structured real-time physical sensor context via PRISM middleware.

Question: {TASK_QUESTION}"""


# ── Ablation 3: PRISM minus multi-event correlation (no event tags) ───────────
def fmt_a3_no_correlation(scenario):
    sensor_keys = ["co_ppm","co2_ppm","pm25_ugm3","voc_ppb","temperature_c",
                   "humidity_pct","sound_db","vibration_ms2"]
    readings = []
    for k in sensor_keys:
        if k not in scenario:
            continue
        val = scenario[k]
        if k not in SAFETY_THRESHOLDS:
            continue
        t = SAFETY_THRESHOLDS[k]
        level, norm = normalize(k, val)
        readings.append(
            f"  {t['name']}: {val} {t['unit']} | normalized={norm} | status={level} "
            f"| threshold={t['danger']} {t['unit']} ({t['standard']})"
        )
    sensor_block = "\n".join(readings)
    urgency = get_urgency(scenario)

    return f"""[PRISM_PERCEPTUAL_INPUT]
URGENCY: {urgency}
SENSOR_READINGS:
{sensor_block}
TEMPORAL_CONTEXT: Single-frame snapshot.
[/PRISM_PERCEPTUAL_INPUT]

You are an AI assistant receiving structured real-time physical sensor context via PRISM middleware.

Question: {TASK_QUESTION}"""


CONDITION_FORMATTERS = {
    "C1_baseline":      fmt_c1_baseline,
    "C2_raw":           fmt_c2_raw,
    "C3_plaintext":     fmt_c3_plaintext,
    "C4_prism":         fmt_c4_prism,
    "A1_no_urgency":    fmt_a1_no_urgency,
    "A2_no_thresholds": fmt_a2_no_thresholds,
    "A3_no_correlation":fmt_a3_no_correlation,
}


def build_all_prompts(scenarios):
    prompts = []
    for s in scenarios:
        for cond, fn in CONDITION_FORMATTERS.items():
            prompts.append({
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
    return prompts


if __name__ == "__main__":
    import json
    with open("/home/claude/prism_eval/data/scenarios.json") as f:
        scenarios = json.load(f)
    prompts = build_all_prompts(scenarios)
    out = "/home/claude/prism_eval/data/prompts.json"
    with open(out, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Generated {len(prompts)} prompts ({len(scenarios)} scenarios × {len(CONDITION_FORMATTERS)} conditions)")
