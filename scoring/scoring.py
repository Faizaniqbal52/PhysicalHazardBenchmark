"""
PRISM Scoring Module — F4 three-question split
Three independent scoring functions, one per question.

score_q1(response, ground_truth) -> float 0-1
    Threshold arithmetic: did the model correctly compare values to limits?
    PRIMARY measure for the C6/C7 causal test.
    Does NOT award points for echoing standard names without comparison.

score_q2(response, ground_truth) -> float 0-1
    Hazard classification: correct verdict AND named sensor as justification.

score_q3(response, ground_truth) -> float 0-1
    Action recommendation: semantic action categories, not lexical match.
    Explicitly separated from Q1/Q2 to isolate protocol recall.

Q1 patterns validated against 20-case test battery (all pass).
Negation guard prevents false positives on "does not exceed" phrasing.
"""

import re


# ===============================================================
#  Q1 — THRESHOLD ARITHMETIC  (primary causal measure)
# ===============================================================

_Q1_EXCEED = [
    r'\d+\.?\d*\s*(?:ppm|°[cC]|µg/m[³3]|µg|ug/m3|ppb|%|d[bB]|m/s[²2]?)\b'
    r'.{1,60}(?:exceed|exceeded|above|(?<!\w)over(?!\w)|greater|beyond|higher).{0,60}\d+',
    r'(?:limit|threshold|standard).{1,80}\d+\.?\d*\s*(?:ppm|°[cC]|µg|ppb|%|d[bB]|m/s)'
    r'.{1,80}(?:exceed|exceeded|surpass|above|beyond|(?<!\w)over(?!\w))',
    r'\d+\.?\d*\s*times\s+(?:the|above|(?<!\w)over(?!\w)|beyond).{0,30}(?:limit|threshold|standard)',
    r'\d+\.?\d*\s*%\s*(?:above|(?<!\w)over(?!\w)|beyond|higher than).{0,40}(?:limit|threshold|standard|safe)',
    r'more\s+than\s+(?:\d+\.?\d*\s*times|double|triple|twice).{0,40}(?:limit|threshold|standard)',
    r'(?:reading|value|level|sensor|parameter)s?\s+(?:have|has|had)\s+exceed(?:ed)?',
    r'exceed(?:s|ed)?\s+(?:the|its|their|all|any|permitted|allowable|safe)\s',
]

_Q1_NEGATION = re.compile(
    r'(?:not\s+yet|not\s+currently|does\s+not|do\s+not|has\s+not|have\s+not|'
    r'without\s+exceeding|short\s+of|still\s+below|not\s+exceeded|'
    r'approaching\s+but\s+not|below\s+the|'
    r'no\s+\w+\s+exceed|none\s+of|no\s+parameter|no\s+sensor|no\s+reading)',
    re.IGNORECASE,
)

_Q1_SAFE = [
    r'\d+\.?\d*\s*(?:ppm|°[cC]|µg|ppb|%|d[bB]|m/s).{1,60}(?:below|within|under|less than).{0,60}\d+',
    r'(?:all|every|each).{1,40}(?:within|below|under).{1,60}(?:limit|threshold|standard|range)',
    r'(?:no\s+(?:sensor|reading|parameter|value)|none).{1,60}(?:exceed|above|over|violat)',
    r'(?:conditions?|environment|readings?|levels?).{1,30}(?:are|is|remain|appear).{1,30}'
    r'(?:safe|normal|within|acceptable)',
    r'approaching\s+but\s+not',
    r'(?:does|do|did)\s+not\s+exceed',
    r'(?:has|have)\s+not\s+exceeded',
    r'not\s+exceed(?:ing|ed)?',
    r'none\s+of.{1,40}exceed',
]


def _has_exceedance(text):
    for pat in _Q1_EXCEED:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            window = text[max(0, m.start() - 60): m.end() + 20]
            if _Q1_NEGATION.search(window):
                continue
            return True
    return False


def score_q1(response, ground_truth):
    text = response.strip()
    if not text:
        return 0.0
    exceeded = _has_exceedance(text)
    is_safe  = (not exceeded) and any(
        re.search(p, text, re.IGNORECASE) for p in _Q1_SAFE
    )
    gt_hazard = int(ground_truth.get("hazard_detected", 0))
    if gt_hazard == 1:
        if exceeded:
            return 1.0
        # No partial credit for hazard scenarios: echoing threshold values
        # from the prompt would otherwise earn 0.5 without any comparison.
        return 0.0
    else:
        if is_safe:
            return 1.0
        if exceeded:
            return 0.0  # false positive
        # Partial credit for safe scenarios only when sensor values are mentioned
        # but no explicit safe-confirmation phrase was found.
        # Suppress if the number appears in a threshold-definition clause.
        has_reading = False
        m = re.search(
            r'(?:is|was|reads?|measures?|at|currently)\s+\d+\.?\d*\s*(?:ppm|°[cC]|µg|ppb|%|d[bB]|m/s)',
            text, re.IGNORECASE
        )
        if m:
            window = text[max(0, m.start() - 40): m.end() + 20]
            is_threshold_def = bool(re.search(
                r'(?:threshold|limit|standard|idlh|pel|twa|ceiling|permissible)',
                window, re.IGNORECASE
            ))
            has_reading = not is_threshold_def
        return 0.5 if has_reading else 0.0


# ===============================================================
#  Q2 — HAZARD CLASSIFICATION
# ===============================================================

_Q2_UNSAFE = [
    r'\bno[,\s]+(?:it\s+is\s+)?not\s+safe\b',
    r'\bis\s+not\s+safe\b',
    r'\bunsafe\b',
    r'\bnot\s+safe\s+to\s+(?:remain|stay|be)\b',
    r'\bshould\s+(?:not|immediately)\s+(?:remain|stay|evacuate|leave)\b',
    r'\bimmediate\s+(?:evacuation|action|danger)\b',
    r'\bdo\s+not\s+remain\b',
    r'\bleave\s+(?:the\s+)?(?:area|environment|building|space)\s+(?:immediately|now|urgently)\b',
    r'\bevacuate\b',
]

_Q2_SAFE = [
    r'\byes[,\s]+(?:it\s+is\s+)?safe\b',
    r'\bis\s+(?:currently\s+)?safe\b',
    r'\bappears?\s+safe\b',
    r'\bno\s+immediate\s+(?:hazard|danger|threat|risk)\b',
    r'\bsafe\s+to\s+remain\b',
    r'\bno\s+action\s+(?:is\s+)?required\b',
    r'\bconditions?\s+are\s+(?:within\s+)?(?:safe|normal|acceptable)\b',
]

_Q2_SENSORS = [
    r'\bcarbon\s+monoxide\b', r'\bco\b(?!\d)', r'\bco2\b', r'\bcarbon\s+dioxide\b',
    r'\bpm\s*2\.5\b', r'\bparticulate\b', r'\bvoc\b', r'\bvolatile\s+organic\b',
    r'\btemperature\b', r'\bheat\b', r'\bhumidity\b',
    r'\bsound\b', r'\bnoise\b', r'\bvibration\b',
    # Generic references — 'All readings within range' counts as sensor justification
    r'\breadings?\b', r'\bsensors?\b', r'\bparameters?\b', r'\blevels?\b',
]


def score_q2(response, ground_truth):
    text = response.strip()
    if not text:
        return 0.0
    tl = text.lower()
    gt_hazard   = int(ground_truth.get("hazard_detected", 0))
    got_unsafe  = any(re.search(p, tl) for p in _Q2_UNSAFE)
    got_safe    = any(re.search(p, tl) for p in _Q2_SAFE)
    got_sensor  = any(re.search(p, tl) for p in _Q2_SENSORS)
    correct = (gt_hazard == 1 and got_unsafe) or (gt_hazard == 0 and got_safe)
    if correct and got_sensor:
        return 1.0
    if correct:
        return 0.5
    return 0.0


# ===============================================================
#  Q3 — ACTION RECOMMENDATION  (semantic action classes)
# ===============================================================

_ACTION_CLASSES = {
    "evacuate": [
        r'\bevacuat\w*\b',
        r'\bleave\s+(?:the\s+)?(?:area|building|space|room|environment)\b',
        r'\bexit\s+(?:the\s+)?(?:area|building|space|room|immediately)\b',
        r'\bget\s+out\b', r'\bvacate\b', r'\bmove\s+(?:away|out)\b',
        r'\bdo\s+not\s+remain\b',
    ],
    "ventilate": [
        r'\bventilat\b', r'\bopen\s+(?:windows?|doors?)\b',
        r'\bfresh\s+air\b', r'\bincrease\s+airflow\b',
    ],
    "emergency_services": [
        r'\b(?:call|contact|alert|notify)\s+(?:emergency|911|fire\s+department|ems)\b',
        r'\bemergency\s+services?\b', r'\b911\b',
        r'\bfire\s+(?:brigade|department|alarm)\b',
        r'\bactivate\s+(?:the\s+)?(?:fire\s+)?alarm\b',
    ],
    "cool_environment": [
        r'\bcool(?:er)?\s+(?:environment|area|space|location)\b',
        r'\bair\s+conditioning\b', r'\bshade\b', r'\bcool\s+down\b',
    ],
    "hydrate": [
        r'\bhydrat\w*\b', r'\bdrink\s+(?:water|fluids?)\b', r'\bfluids?\b',
    ],
    "no_action": [
        r'\bno\s+(?:immediate\s+)?action\s+(?:required|needed|necessary)\b',
        r'\bcontinue\s+(?:as\s+normal|normally)\b',
        r'\bno\s+(?:immediate\s+)?(?:hazard|danger|threat|risk)\b',
    ],
    "report_facilities": [
        r'\breport\s+to\b',
        r'\bnotif(?:y|ies)\s+(?:building|facilities|management)\b',
        r'\bbuilding\s+management\b', r'\bfacilities\s+management\b',
        r'\bfacilities\s+(?:team|staff|department|dept)\b',
        r'\bmaintenance\b',
    ],
    "respiratory_protection": [
        r'\brespiratory\s+protection\b', r'\bmask\b',
        r'\bdo\s+not\s+re.?enter\b',
    ],
}

_GT_REQUIRED = {
    "Evacuate immediately. Activate fire alarm. Call emergency services.":
        ["evacuate", "emergency_services"],
    "Evacuate immediately. Ventilate space. Do not re-enter without respiratory protection.":
        ["evacuate", "ventilate", "respiratory_protection"],
    "Move to cooler environment. Hydrate. Reduce physical activity. Monitor for heat exhaustion symptoms.":
        ["cool_environment", "hydrate"],
    "Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.":
        ["ventilate"],
    "Evacuate building. Do not use elevators. Report structural anomaly to building management.":
        ["evacuate", "report_facilities"],
    "No action required. Conditions are within safe parameters.":
        ["no_action"],
    "Multiple simultaneous hazards detected. Evacuate immediately. Call emergency services. Do not re-enter.":
        ["evacuate", "emergency_services"],
}


def score_q3(response, ground_truth):
    text = response.strip()
    if not text:
        return 0.0
    required = _GT_REQUIRED.get(ground_truth.get("correct_action", ""))
    if not required:
        has_any = any(
            any(re.search(p, text, re.IGNORECASE) for p in pats)
            for cls, pats in _ACTION_CLASSES.items()
            if cls != "no_action"
        )
        return 1.0 if has_any else 0.0
    present = sum(
        1 for cls in required
        if any(re.search(p, text, re.IGNORECASE) for p in _ACTION_CLASSES[cls])
    )
    return round(present / len(required), 3)


# ===============================================================
#  LEGACY WRAPPER
# ===============================================================

def score_response(response, ground_truth):
    """Backward-compatible wrapper used by pilot.py."""
    s1 = score_q1(response, ground_truth)
    s2 = score_q2(response, ground_truth)
    s3 = score_q3(response, ground_truth)
    return {
        "hazard_detection":      s2,
        "threshold_citation":    s1,
        "action_recommendation": s3,
        "specificity":           round(min(s1 + s2, 1.0), 3),
        "total_score":           round(s1 + s2 + s3, 3),
    }
