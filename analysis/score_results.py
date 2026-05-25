"""
score_results.py
Scores raw benchmark_results.csv → scored_results.json
Usage: python analysis/score_results.py
"""
import csv, json, sys, re
sys.path.insert(0, 'scoring')
from scoring import score_q1, score_q2, score_q3

# Ground truth definitions
GT = {}
for i in range(1, 21):
    GT[f'A{i}'] = {
        'hazard_detected': 1,
        'correct_action': 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'
    }

marginal_b = ['B1p','B2p','B3p','B4p','B11p','B12p','B13p','B15p','B18p','B20p']
evac_b     = ['B5p','B6p','B7p','B8p','B9p','B10p','B14p','B17p','B19p']
for sid in marginal_b:
    GT[sid] = {'hazard_detected': 1,
               'correct_action': 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'}
for sid in evac_b:
    GT[sid] = {'hazard_detected': 1,
               'correct_action': 'Evacuate immediately. Ventilate space. Do not re-enter without respiratory protection.'}
GT['B16p'] = {'hazard_detected': 1,
              'correct_action': 'Move to cooler environment. Hydrate. Reduce physical activity. Monitor for heat exhaustion symptoms.'}

c_gt = {
    'C1p':  (1, 'Move to cooler environment. Hydrate. Reduce physical activity. Monitor for heat exhaustion symptoms.'),
    'C2p':  (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C3p':  (1, 'Evacuate immediately. Ventilate space. Do not re-enter without respiratory protection.'),
    'C4p':  (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C5p':  (1, 'Evacuate building. Do not use elevators. Report structural anomaly to building management.'),
    'C6p':  (1, 'Evacuate immediately. Activate fire alarm. Call emergency services.'),
    'C7p':  (1, 'Move to cooler environment. Hydrate. Reduce physical activity. Monitor for heat exhaustion symptoms.'),
    'C8p':  (1, 'Evacuate immediately. Ventilate space. Do not re-enter without respiratory protection.'),
    'C9p':  (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C10p': (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C11p': (1, 'Evacuate building. Do not use elevators. Report structural anomaly to building management.'),
    'C12p': (1, 'Evacuate building. Do not use elevators. Report structural anomaly to building management.'),
    'C13p': (1, 'Evacuate building. Do not use elevators. Report structural anomaly to building management.'),
    'C14p': (0, 'No action required. Conditions are within safe parameters.'),
    'C15p': (1, 'Evacuate immediately. Ventilate space. Do not re-enter without respiratory protection.'),
    'C16p': (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C17p': (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C18p': (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C19p': (1, 'Improve ventilation. Use air purifier if available. Avoid strenuous activity indoors.'),
    'C20p': (0, 'No action required. Conditions are within safe parameters.'),
}
for sid, (haz, action) in c_gt.items():
    GT[sid] = {'hazard_detected': haz, 'correct_action': action}


def strip_question_prefix(text, qnum):
    """Remove echoed question text before scoring."""
    patterns = [
        rf'Q{qnum}\.\s+For each sensor.*?exceeded\.?\**\s*',
        rf'\*\*Q{qnum}\.[^*]+\*\*\s*',
        rf'Q{qnum}\.\s+Based on your analysis.*?conclusion\.\s*',
        rf'Q{qnum}\.\s+What immediate.*?taken\?\s*',
        rf'Q{qnum}\.[^\n]*\n',
    ]
    result = text
    for pat in patterns:
        result = re.sub(pat, '', result, flags=re.IGNORECASE | re.DOTALL, count=1)
    return result.strip() or text


results = []
with open('data/benchmark_results.csv', newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row['scenario_id'].strip()
        if sid not in GT:
            continue
        gt = GT[sid]
        q1r = strip_question_prefix(row['q1_response'].replace('\\n', ' '), 1)
        q2r = strip_question_prefix(row['q2_response'].replace('\\n', ' '), 2)
        q3r = strip_question_prefix(row['q3_response'].replace('\\n', ' '), 3)
        s1 = score_q1(q1r, gt)
        s2 = score_q2(q2r, gt)
        s3 = score_q3(q3r, gt)
        results.append({
            'scenario_id': sid,
            'category':    row['category'],
            'model':       row['model'],
            'condition':   row['condition'],
            'run':         row['run_number'],
            'q1': s1, 'q2': s2, 'q3': s3,
            'total': round(s1 + s2 + s3, 3)
        })

with open('data/scored_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Scored {len(results)} rows → data/scored_results.json')
