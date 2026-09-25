import json
S = json.load(open('.scratch_laya/scores.json', encoding='utf-8'))
items, scores = S['items'], S['scores']
def qmax(p, i, fi=0, topn=None):
    ch = scores[p][str(i)][str(fi)]
    vals = [v for ci, v in ch.items() if topn is None or int(ci) < topn]
    return max(vals) if vals else 0.0
for p in scores:
    print('=====', p)
    for set_ in ('dev', 'golden', 'golden_decomp'):
        rows = [(i, x) for i, x in enumerate(items) if x['set'] == set_]
        un = sorted(round(qmax(p, i), 3) for i, x in rows if not x['answerable'])
        an = sorted(round(qmax(p, i), 3) for i, x in rows if x['answerable'])
        print(set_, 'UNANS', un)
        print(set_, 'ANS  ', an)
print()
def auc(pos, neg):
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return s / (len(pos) * len(neg))
for p in scores:
    for set_ in ('dev', 'golden'):
        rows = [(i, x) for i, x in enumerate(items) if x['set'] == set_ and x['id'] != 'dev_u27']
        print(p, set_, 'AUC', round(auc([qmax(p, i) for i, x in rows if x['answerable']], [qmax(p, i) for i, x in rows if not x['answerable']]), 3),
              'top3 AUC', round(auc([qmax(p, i, topn=3) for i, x in rows if x['answerable']], [qmax(p, i, topn=3) for i, x in rows if not x['answerable']]), 3))
for i, x in enumerate(items):
    if x['set'] in ('golden','golden_decomp') and (not x['answerable'] or qmax('P1', i) < 0.55):
        print(x['set'], x['id'], x['heuristic'], 'rr', round(max(x['rr']),3), {p: round(qmax(p, i), 3) for p in scores}, [round(qmax('P1', i, fi),3) for fi in range(1, 1+len(x['sub_questions']))])
