import torch, os, csv, json
OUT="out/"; DM,NH,BLK=512,8,2048
SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
def L(p): return torch.load(OUT+p,map_location="cpu",mmap=True,weights_only=False)
def per(m): return m*DM//NH
# territory: phase1(en)=blocks0-3 (base/shared); phase p>=2 owns block p+2
owner={}
for i,l in enumerate(SEQ): owner[l]= ("base(0-3)" if i==0 else f"blk{i+2}")
M={}
for r in csv.DictReader(open(os.path.expanduser("~/bdh-review/reports/ra2b_matrix.csv"))):
    M.setdefault(r["checkpoint"],{})[r["eval_lang"]]=float(r["ppl"])
print("=== EXP3a: which OLD blocks actually received gradient in each phase (full nonzero count, no sampling) ===")
rows=[]
for i,lang in enumerate(SEQ[1:],start=1):
    par=SEQ[i-1]
    c=L(f"bdh_europarl_ladRA2b-{lang}_last.pt"); p=L(f"bdh_europarl_ladRA2b-{par}_last.pt")
    pc=per(int(c["cfg"]["mlp_internal_dim_multiplier"])); pp=per(int(p["cfg"]["mlp_internal_dim_multiplier"]))
    v=c["optimizer_state"]["state"][0]["exp_avg_sq"]; dec=c["model_state"]["decoder"]; dpp=p["model_state"]["decoder"]
    lpw=pc//BLK; active=[]; churn={}
    for b in range(lpw-1):                      # old blocks only
        nz=sum(int((v[h*pc+b*BLK:h*per(int(c['cfg']['mlp_internal_dim_multiplier']))-(pc-per(int(c['cfg']['mlp_internal_dim_multiplier'])))][:0].numel()) ) if False else int((v[h*pc+b*BLK:h*pc+(b+1)*BLK]!=0).sum()) for h in range(NH))
        if nz>0: active.append((b,nz))
    # weight churn, sampled rows (every 16th) - stated explicitly
    for b in range(lpw-1):
        num=den=0.0
        for h in range(NH):
            rc=torch.arange(h*pc+b*BLK, h*pc+(b+1)*BLK, 16); rp=torch.arange(h*pp+b*BLK, h*pp+(b+1)*BLK, 16)
            a=dpp[rp].float(); bb=dec[rc].float()
            num+=float(((bb-a)**2).sum()); den+=float((a**2).sum())
        churn[b]=(num/den)**0.5
    del c,p,v,dec,dpp
    rows.append((lang,i,active,churn))
    tot=len(range(lpw-1)); nact=len(active)
    top=sorted(churn.items(), key=lambda kv:-kv[1])[:3]
    print(f" {lang:>2s} (ph{i+1}): old blocks={tot} gradient-active={nact} {[b for b,_ in active][:6]}{'...' if nact>6 else ''} | largest churn: " + ", ".join(f"b{b}:{r:.4f}" for b,r in top))
print("\n=== EXP3b: does churn in a language's OWN territory predict its serving damage? ===")
print("(churn = relative ||dW||/||W|| of that language's block during a LATER phase; damage = ppl rise of that domain)")
pairs=[]
for lang,i,active,churn in rows:
    for b,r in churn.items():
        own=None
        for k,l in enumerate(SEQ):
            if l==lang: continue
            if (k==0 and b<=3) or (k>=2 and b==k+2): own=l
        if own and own!=lang and own in M.get(lang,{}) and own in M.get(SEQ[i-1],{}):
            before=M[SEQ[i-1]][own]; after=M[lang][own]
            pairs.append((lang,own,b,r,after-before,(after-before)/before*100))
import statistics as S
byc=[p for p in pairs if p[5]>4]; byd=[p for p in pairs if p[5]<-4]
print(f" (own,other,block,churn,delta,rel%) samples={len(pairs)}")
if len(pairs)>3:
    xs=[p[3] for p in pairs]; ys=[p[5] for p in pairs]
    mx,my=S.mean(xs),S.mean(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); sx=(sum((x-mx)**2 for x in xs))**.5; sy=(sum((y-my)**2 for y in ys))**.5
    print(f" Pearson r(churn, relative damage) = {cov/(sx*sy):+.3f} over {len(pairs)} pairs")
    print(f" median churn when domain HURT (>4%): {S.median([p[3] for p in byc]):.4f} (n={len(byc)})")
    print(f" median churn when domain HELPED (<-4%): {S.median([p[3] for p in byd]):.4f} (n={len(byd)})")
    print(f" median churn when within +-4% floor: {S.median([p[3] for p in pairs if abs(p[5])<=4]):.4f}")
print("\n=== EXP3c: base blocks 0-3 (shared English-era territory) churn per phase ===")
print("  " + "  ".join(f"{l}:{c.get(0,float('nan')):.4f}" for l,i,a,c in rows[:8]))
