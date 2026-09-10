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
BB=int(sys.argv[3]) if len(sys.argv)>3 else 8   # crops per forward; validated against lang_eval at BB=8
dev=torch.device("cuda"); model,cfg=_load_model(CK); model=model.to(dev).eval(); bs=cfg["block_size"]
blocks=_europarl_blocks("data",30_000_000,langs=tuple(SEQ))
N=int(model.decoder.shape[0])//NH; LPW=N//BLK
WIDTHS=[(j+1)*BLK for j in range(LPW)]
BLOCK_OF={b:("en" if b<=3 else SEQ[b-3]) for b in range(LPW)}
def data(lang):
    raw=blocks[lang]["val"]+blocks[lang]["test"]
    d=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).astype(np.int64))
    half=(len(d)-bs-1)//2
    return d[:half], d[half:]
def nll(d,mask,iters):
    """x and y MUST come from the same crop: d[i:i+bs] -> d[i+1:i+1+bs].
    Earlier versions paired x from one buffer half with y from another, which drove
    every NLL to chance level (~e^5.3) and made argmin pick the smallest model."""
    g=torch.Generator().manual_seed(9021); ls=[]
    m=None
    if mask is not None:
        m=torch.ones(N,device=dev); m[mask:]=0.0
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for _ in range(iters):
            ix=torch.randint(len(d)-bs-1,(BB,),generator=g)
            x=torch.stack([d[int(i):int(i)+bs] for i in ix]).to(dev)
            y=torch.stack([d[int(i)+1:int(i)+1+bs] for i in ix]).to(dev)
            lo,_,_=model(x,None,None,neuron_mask=m) if m is not None else model(x)
            ls.append(torch.nn.functional.cross_entropy(lo.reshape(-1,lo.size(-1)),y.reshape(-1)).item())
    return sum(ls)/len(ls)
print(f"ckpt={CK.split('-')[-1]} LPW={LPW} iters={IT}")
print("\ndomain | sel_width | sel_lang | oracle | correct | ppl_sel | ppl_oracle | ppl_free")
ok=0; rows=[]; gate_ok=True
for l in SEQ:
    xc,_=data(l)
    scores=[nll(xc,w,IT) for w in WIDTHS]
    best=int(np.argmin(scores)); bw=WIDTHS[best]
    own=(POS[l]+3)*BLK if l!="en" else 4*BLK
    correct = BLOCK_OF[bw//BLK-1]==l; ok+=correct
    xd,_=data(l)
    p_sel=math.exp(nll(xd,bw,IT)); p_orc=math.exp(nll(xd,own,IT)); p_free=math.exp(nll(xd,None,IT))
    if l=="en":
        # instrument gate: must reproduce the matrix (free ~31, oracle-masked ~2.3) or stop
        if not (15.0 < p_free < 60.0 and p_orc < 4.0):
            gate_ok=False
            print(f"SANITY GATE FAILED: en free={p_free:.2f} (expect ~31) oracle={p_orc:.2f} (expect ~2.3). Aborting.")
            break
    rows.append((l,bw,BLOCK_OF[bw//BLK-1],BLOCK_OF[own//BLK-1],correct,p_sel,p_orc,p_free))
    print(f"  {l}   |  {bw:6d}   | {BLOCK_OF[bw//BLK-1]:>3s}     |  {BLOCK_OF[own//BLK-1]:>3s}  |  {'Y' if correct else 'n'}    | {p_sel:7.2f}  | {p_orc:7.2f}    | {p_free:7.2f}")
if not gate_ok: raise SystemExit(1)
print(f"\nP(correct prefix | x) = {ok}/20 = {ok*5}%   (labels used only to SCORE correctness)")
gap=[(r[5]-r[6])/r[6]*100 for r in rows]
print(f"mean ppl penalty of selected vs oracle prefix: {np.mean(gap):+.1f}%  (0% = selection as good as language ID)")
print(f"free eval vs oracle mean penalty: {np.mean([(r[7]-r[6])/r[6]*100 for r in rows]):+.0f}%")
