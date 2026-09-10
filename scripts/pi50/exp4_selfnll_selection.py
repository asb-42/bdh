"""
KNOWN-ISSUE (2026-09-10): FAILS its own sanity check - unmasked en PPL ~198 vs 31.07
from the matrix. Numbers from this script are NOT quotable until its crop/target path matches
scripts/lang_eval.py. Kept in-tree as WIP because the design (label-free argmin-NLL over
cumulative prefixes, calibration/test split) is the intended A2 arm.
"""A2: label-free prefix selection by self-supervised NLL. For each domain crop set, score every
cumulative prefix width with neuron_mask, pick argmin NLL on a CALIBRATION slice, then evaluate
PPL on a disjoint TEST slice under (a) selected prefix, (b) oracle own prefix, (c) free/no mask.
Reports P(correct prefix | x) and ppl achieved vs oracle - no labels used for selection."""
import sys, math, numpy as np, torch
sys.path.insert(0,".")
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks
SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
POS={l:i for i,l in enumerate(SEQ)}; BLK,NH=2048,8
CK=sys.argv[1] if len(sys.argv)>1 else "out/bdh_europarl_ladRA2b-lt_last.pt"
IT=int(sys.argv[2]) if len(sys.argv)>2 else 30
dev=torch.device("cuda"); model,cfg=_load_model(CK); model=model.to(dev).eval(); bs=cfg["block_size"]
blocks=_europarl_blocks("data",30_000_000,langs=tuple(SEQ))
N=int(model.decoder.shape[0])//NH; LPW=N//BLK
WIDTHS=[(j+1)*BLK for j in range(LPW)]
BLOCK_OF={b:("en" if b<=3 else SEQ[b-3]) for b in range(LPW)}
def data(lang):
    raw=blocks[lang]["val"]; d=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).astype(np.int64))
    half=(len(d)-bs-1)//2
    return d[:half], d[half:]
def nll(x_all,y_all,mask,iters):
    g=torch.Generator().manual_seed(9021); ls=[]
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for _ in range(iters):
            ix=torch.randint(len(x_all)-bs-1,(1,),generator=g)
            x=x_all[ix:ix+bs].unsqueeze(0).to(dev); y=y_all[ix:ix+bs].unsqueeze(0).to(dev)
            lo,_,_=model(x,None,None,neuron_mask=mask) if mask is not None else model(x)
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1,lo.size(-1)),y.reshape(-1)).item())
    return sum(ls)/len(ls)
print(f"ckpt={CK.split('-')[-1]} LPW={LPW} iters={IT}")
print("\ndomain | sel_width | sel_lang | oracle | correct | ppl_sel | ppl_oracle | ppl_free")
ok=0; rows=[]
for l in SEQ:
    xc,yc=data(l)
    scores=[]
    for w in WIDTHS:
        m=torch.ones(N,device=dev); m[w:]=0.0
        scores.append(nll(xc,yc,m,IT))
    best=int(np.argmin(scores)); bw=WIDTHS[best]
    own=(POS[l]+3)*BLK if l!="en" else 4*BLK
    correct = BLOCK_OF[bw//BLK-1]==l; ok+=correct
    xd,yd=data(l)
    msel=torch.ones(N,device=dev); msel[bw:]=0.0
    morc=torch.ones(N,device=dev); morc[own:]=0.0
    p_sel=math.exp(nll(xd,yd,msel,IT)); p_orc=math.exp(nll(xd,yd,morc,IT)); p_free=math.exp(nll(xd,yd,None,IT))
    rows.append((l,bw,BLOCK_OF[bw//BLK-1],BLOCK_OF[own//BLK-1],correct,p_sel,p_orc,p_free))
    print(f"  {l}   |  {bw:6d}   | {BLOCK_OF[bw//BLK-1]:>3s}     |  {BLOCK_OF[own//BLK-1]:>3s}  |  {'Y' if correct else 'n'}    | {p_sel:7.2f}  | {p_orc:7.2f}    | {p_free:7.2f}")
print(f"\nP(correct prefix | x) = {ok}/20 = {ok*5}%   (labels used only to SCORE correctness)")
gap=[(r[5]-r[6])/r[6]*100 for r in rows]
print(f"mean ppl penalty of selected vs oracle prefix: {np.mean(gap):+.1f}%  (0% = selection as good as language ID)")
print(f"free eval vs oracle mean penalty: {np.mean([(r[7]-r[6])/r[6]*100 for r in rows]):+.0f}%")
