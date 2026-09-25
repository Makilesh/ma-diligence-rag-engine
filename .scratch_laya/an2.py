import json
S = json.load(open('.scratch_laya/scores.json', encoding='utf-8'))
items, scores = S['items'], S['scores']
d = json.load(open('.scratch_laya/contexts.json', encoding='utf-8'))
for i, x in enumerate(items):
    if x['set'] != 'dev': continue
    ch = scores['P1'][str(i)]['0']
    best = max(ch, key=lambda k: ch[k])
    print(x['id'], x['answerable'], x.get('evidence_in_context'), x['heuristic'], 'P1max', round(ch[best],3), 'rr', round(x['rr'][int(best)],3), '|', ' '.join(d[i]['reranked'][int(best)]['text'].split())[:150])
