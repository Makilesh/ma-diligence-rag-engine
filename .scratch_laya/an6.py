import json
S = json.load(open('.scratch_laya/scores.json', encoding='utf-8'))
items, scores = S['items'], S['scores']
MISS = {'dev_a01', 'dev_a35', 'dev_a36'}
def F(p, i, K, fi='0'):
    ch = scores[p][str(i)][fi]; rr = items[i]['rr']; o = sorted(range(len(rr)), key=lambda k: -rr[k])[:K]
    return max(ch[str(k)] for k in o)
for p in ('P1','P2','P3'):
  for K in (1,2,3,4,5,10):
    dev = [(i,x) for i,x in enumerate(items) if x['set']=='dev' and x['id']!='dev_u27']
    pos = sorted(F(p,i,K) for i,x in dev if x['answerable'] and x['id'] not in MISS)
    neg = [(F(p,i,K), x['heuristic']) for i,x in dev if not x['answerable']]
    miss = [round(F(p,i,K),3) for i,x in dev if x['id'] in MISS]
    row=[]
    for t in (0.30,0.33,0.35,0.37):
        row.append(f"t={t}: fv={sum(v<t for v in pos)} caught={sum(v<t for v,h in neg)} (adm {sum(v<t for v,h in neg if h=='admitted')}, amb {sum(v<t for v,h in neg if h=='llm_fallback')})")
    print(p, 'K', K, 'minpos', round(pos[0],3), round(pos[1],3), 'miss', miss, ' | '.join(row))
