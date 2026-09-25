import json, sys, time
from pathlib import Path
sys.path.insert(0, '.')
from src.decisions.laya_client import decide_batch
d = json.load(open('.scratch_laya/contexts.json', encoding='utf-8'))
d = [x for x in d if x['set'] in ('dev', 'golden')]
def windows(text, size=450):
    lines = [l for l in text.split('\n') if l.strip()]
    out, cur = [], ''
    for l in lines:
        if cur and len(cur) + len(l) > size:
            out.append(cur); cur = cur.split('\n')[-1] + '\n' if False else ''
        cur += l + '\n'
    if cur: out.append(cur)
    return out or [text]
V = {
 "P1": ({"type": "noul", "instructions": "Does `passage` state the information needed to answer `question`?"}, 'passage', 'noul'),
 "V3": ({"type": "noul", "instructions": "Does `passage` state the exact fact or figure that `question` asks for, for the same period, party and metric?"}, 'passage', 'noul'),
 "V5": ({"type": "noul", "instructions": "Does `document` state the information needed to answer `question`?"}, 'document', 'noul'),
 "V2": ({"type": "choice", "instructions": "How does `passage` relate to `question`?", "criteria": {"answers": "states the specific fact the question asks for", "related": "same topic but lacks the specific fact asked for", "unrelated": "different topic"}}, 'passage', 'choice'),
 "V4": ({"type": "score", "instructions": "How completely does `passage` answer `question`?", "criteria": ["none: does not answer", "partial: related but lacks the asked-for fact", "full: states the asked-for fact"]}, 'passage', 'score'),
}
out = {}
for name, (q, field, kind) in V.items():
    reqs, keys = [], []
    for i, x in enumerate(d):
        for ci, c in enumerate(x['reranked']):
            reqs.append(({"question": x['query'], field: c['text']}, {"a": q})); keys.append((i, ci))
    t = time.time(); res = decide_batch(reqs); print(name, len(reqs), round(time.time()-t,1), file=sys.stderr)
    for (i, ci), r in zip(keys, res):
        a = r['a']
        v = a['noul'] if kind == 'noul' else (a['probabilities']['answers'] if kind == 'choice' else a['probabilities']['2'])
        out.setdefault(name, {}).setdefault(i, {})[ci] = v
# windows variant with P1
reqs, keys = [], []
for i, x in enumerate(d):
    for ci, c in enumerate(x['reranked']):
        for w in windows(c['text']):
            reqs.append(({"question": x['query'], "passage": w}, {"a": V['P1'][0]})); keys.append((i, ci))
t = time.time(); res = decide_batch(reqs); print('W1', len(reqs), round(time.time()-t,1), file=sys.stderr)
for (i, ci), r in zip(keys, res):
    cur = out.setdefault('W1', {}).setdefault(i, {})
    cur[ci] = max(cur.get(ci, 0), r['a']['noul'])
Path('.scratch_laya/scores2.json').write_text(json.dumps({"items": [{k: v for k, v in x.items() if k not in ('reranked','expanded')} | {"rr": [c['reranker_score'] for c in x['reranked']]} for x in d], "scores": out}), encoding='utf-8')
