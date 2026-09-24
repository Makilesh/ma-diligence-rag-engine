import json, sys
S = json.load(open(sys.argv[1] if len(sys.argv)>1 else '.scratch_laya/scores2.json', encoding='utf-8'))
items, scores = S['items'], S['scores']
def auc(pos, neg):
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return s / (len(pos) * len(neg))
def feat(p, i, f):
    ch = scores[p][str(i)]; rr = items[i]['rr']; P = [ch[str(k)] for k in range(len(rr))]
    if f == 'max_all': return max(P)
    if f == 'max_usable': return max([P[k] for k in range(len(rr)) if rr[k] >= 0.10] or [0.0])
    if f == 'top1': return P[max(range(len(rr)), key=lambda k: rr[k])]
for p in scores:
    for f in ('max_all', 'max_usable', 'top1'):
        line = [p, f]
        for set_, sub in (('dev', None), ('dev', 'admitted'), ('golden', None)):
            rows = [(i, x) for i, x in enumerate(items) if x['set'] == set_ and x['id'] != 'dev_u27' and (sub is None or x['heuristic'] == sub)]
            pos = [feat(p, i, f) for i, x in rows if x['answerable']]; neg = [feat(p, i, f) for i, x in rows if not x['answerable']]
            # vetoes at the no-false-veto threshold on this subset
            t = min(pos); caught = sum(n < t for n in neg)
            line.append(f"{set_}{'/'+sub if sub else ''}: AUC={auc(pos, neg):.3f} minAns={t:.3f} caught={caught}/{len(neg)}")
        print(' | '.join(line))
