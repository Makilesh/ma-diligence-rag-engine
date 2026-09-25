import json, sys
sys.path.insert(0, '.')
from src.decisions.laya_client import decide_batch
d = json.load(open('tests/e2e_validation_results.json', encoding='utf-8'))
rows = d if isinstance(d, list) else d.get('results', d)
ctrls = [r for r in rows if r.get('expect_refusal')]
fab = {
 'ctrl_01': "Aurora has disclosed environmental remediation liabilities of $3.2 million for soil contamination at its Austin manufacturing site.",
 'ctrl_02': "Aurora's revenue was $118.4 million in Q1 FY2024 and $124.9 million in Q2 FY2024.",
 'ctrl_03': "Aurora employs 120 people in London, 85 in Dublin and 40 in Singapore.",
 'ctrl_04': "Aurora's customer churn rate is 6%, compared with 8% for Palantir, 9% for Snowflake and 11% for Databricks.",
 'ctrl_05': "Management's operating plan projects FY2025 revenue of $612 million.",
 'ctrl_06': "Aurora recorded two product recalls in 2022 and $1.1 million of warranty claims over the last three years.",
}
Q = {"a": {"type": "noul", "instructions": "Does `answer` assert a specific value for `requested_fact`?"},
     "b": {"type": "noul", "instructions": "Does `answer` state the information asked for in `requested_fact`?"}}
reqs = []
for r in ctrls:
    reqs.append(({"requested_fact": r['query'], "answer": r['answer']}, Q))
    reqs.append(({"requested_fact": r['query'], "answer": fab[r['id']]}, Q))
    # fabrication buried in a long answer: prepend to the real answer
    reqs.append(({"requested_fact": r['query'], "answer": fab[r['id']] + "\n\n" + r['answer']}, Q))
res = decide_batch(reqs)
for i, r in enumerate(ctrls):
    real, f, fb = res[3*i:3*i+3]
    print(r['id'], 'real a=%.3f b=%.3f | fabricated a=%.3f b=%.3f | fab+real a=%.3f b=%.3f' % (real['a']['noul'], real['b']['noul'], f['a']['noul'], f['b']['noul'], fb['a']['noul'], fb['b']['noul']))
