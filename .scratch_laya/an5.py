import json, sys
S = json.load(open('.scratch_laya/scores2.json', encoding='utf-8'))
items, scores = S['items'], S['scores']
MISS = {'dev_a01', 'dev_a35', 'dev_a36'}
def auc(pos, neg):
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return s / (len(pos) * len(neg))
def feat(p, i, f):
    ch = scores[p][str(i)]; rr = items[i]['rr']; P = [ch[str(k)] for k in range(len(rr))]
    if f == 'max_all': return max(P)
    if f == 'max_usable': return max([P[k] for k in range(len(rr)) if rr[k] >= 0.10] or [0.0])
    if f == 'top1': return P[max(range(len(rr)), key=lambda k: rr[k])]
    if f == 'top2': o = sorted(range(len(rr)), key=lambda k: -rr[k]); return max(P[k] for k in o[:2])
ens = lambda i, f: sum(feat(p, i, f) for p in ('P1', 'V5', 'W1')) / 3
for p in list(scores) + ['ENS']:
    for f in ('max_all', 'top1', 'top2'):
        F = (lambda i: ens(i, f)) if p == 'ENS' else (lambda i: feat(p, i, f))
        dev = [(i, x) for i, x in enumerate(items) if x['set'] == 'dev' and x['id'] != 'dev_u27' and x['id'] not in MISS]
        adm = [(i, x) for i, x in dev if x['heuristic'] == 'admitted']
        pos = [F(i) for i, x in dev if x['answerable']]; neg = [F(i) for i, x in dev if not x['answerable']]
        t = min(pos)
        negadm = [F(i) for i, x in adm if not x['answerable']]
        g = [(i, x) for i, x in enumerate(items) if x['set'] == 'golden']
        gctl = {x['id']: round(F(i), 3) for i, x in g if not x['answerable']}
        gans_vetoed = [x['id'] for i, x in g if x['answerable'] and F(i) < t]
        print(f"{p:4} {f:8} devAUC={auc(pos,neg):.3f} t={t:.3f} dev caught {sum(n<t for n in neg)}/{len(neg)} (admitted {sum(n<t for n in negadm)}/{len(negadm)}) | golden ctrl<t: {[k for k,v in gctl.items() if v < t]} ans vetoed {gans_vetoed}")
