import json, sys, time, re
from pathlib import Path
sys.path.insert(0, '.')
from src.decisions.laya_client import decide_batch
PHRASINGS = {
 "P1": "Does `passage` state the information needed to answer `question`?",
 "P2": "Does `passage` contain the answer to `question`?",
 "P3": "Can `question` be answered using only the facts stated in `passage`?",
}
d = json.load(open('.scratch_laya/contexts.json', encoding='utf-8'))
def norm(s): return ' '.join(s.split())
dev = {q['id']: q for q in json.load(open('eval/answerability_dev.json', encoding='utf-8'))['questions']}
out = {}
for pname, instr in PHRASINGS.items():
    reqs, keys = [], []
    for i, x in enumerate(d):
        facets = [x['query']] + list(x['sub_questions'])
        for fi, f in enumerate(facets):
            for ci, c in enumerate(x['reranked']):
                reqs.append(({"question": f, "passage": c['text']}, {"a": {"type": "noul", "instructions": instr}}))
                keys.append((i, fi, ci))
    t = time.time()
    res = decide_batch(reqs)
    print(pname, len(reqs), 'pairs', round(time.time() - t, 2), 's', file=sys.stderr)
    for (i, fi, ci), r in zip(keys, res):
        out.setdefault(pname, {}).setdefault(i, {}).setdefault(fi, {})[ci] = r['a']['noul']
# evidence-in-context flag for dev answerables
for i, x in enumerate(d):
    if x['set'] == 'dev' and x['answerable']:
        ev = norm(dev[x['id']]['evidence'])
        x['evidence_in_context'] = any(ev in norm(c['text']) for c in x['reranked'])
res = {"items": [{k: v for k, v in x.items() if k not in ('reranked', 'expanded')} | {"rr": [c['reranker_score'] for c in x['reranked']]} for x in d], "scores": out}
Path('.scratch_laya/scores.json').write_text(json.dumps(res), encoding='utf-8')
