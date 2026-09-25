import json
S = json.load(open('.scratch_laya/scores.json', encoding='utf-8'))
items, scores = S['items'], S['scores']
def auc(pos, neg):
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return s / (len(pos) * len(neg))
def feats(p, i, fi=0):
    ch = scores[p][str(i)][str(fi)]; rr = items[i]['rr']
    P = [ch[str(k)] for k in range(len(rr))]
    usable = [P[k] for k in range(len(rr)) if rr[k] >= 0.10]
    order = sorted(range(len(rr)), key=lambda k: -rr[k])
    return {
        'max_all': max(P),
        'max_usable': max(usable) if usable else 0.0,
        'max_rr05': max([P[k] for k in range(len(rr)) if rr[k] >= 0.05] or [0.0]),
        'top1': P[order[0]],
        'top3': max(P[k] for k in order[:3]),
        'max_P_x_rr': max(P[k] * rr[k] for k in range(len(rr))),
    }
for p in scores:
    for set_ in ('dev', 'golden'):
        rows = [(i, x) for i, x in enumerate(items) if x['set'] == set_ and x['id'] != 'dev_u27']
        F = {i: feats(p, i) for i, _ in rows}
        line = []
        for f in F[rows[0][0]]:
            line.append(f"{f}={auc([F[i][f] for i,x in rows if x['answerable']], [F[i][f] for i,x in rows if not x['answerable']]):.3f}")
        print(p, set_, ' '.join(line))
# heuristic-admitted subset, dev
print()
for p in scores:
    rows = [(i, x) for i, x in enumerate(items) if x['set']=='dev' and x['id']!='dev_u27' and x['heuristic']=='admitted']
    F = {i: feats(p, i) for i, _ in rows}
    for f in ('max_all','max_usable','top1','top3'):
        un = sorted(round(F[i][f],3) for i,x in rows if not x['answerable']); an = sorted(round(F[i][f],3) for i,x in rows if x['answerable'])
        print(p, f, 'admitted-dev AUC', round(auc([F[i][f] for i,x in rows if x['answerable']],[F[i][f] for i,x in rows if not x['answerable']]),3))
        if p=='P1': print('   UN', un); print('   AN', an)
