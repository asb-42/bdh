import math,sys,numpy as np,torch
sys.path.insert(0,".")
from pipeline.analyze import _load_model
from pipeline.data import _europarl_blocks
SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
CK=sys.argv[1] if len(sys.argv)>1 else "out/bdh_europarl_ladRA2b-lt_last.pt"
dev=torch.device("cuda"); blocks=_europarl_blocks("data",30_000_000,langs=tuple(SEQ))
model,cfg=_load_model(CK); model=model.to(dev).eval()
bs=cfg["block_size"]; NH=8; DM=512; BLK=2048
Ntot=int(model.decoder.shape[0])//NH     # per-head width of final ckpt
print(f"ckpt={CK.split('-')[-1]} N/head={Ntot} block={bs}")
def ev(lang,mask=None,iters=100,batch=1):
    raw=blocks[lang]["val"]+blocks[lang]["test"]
    d=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).astype(np.int64))
    g=torch.Generator().manual_seed(1234); ls=[]
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for _ in range(iters):
            ix=torch.randint(len(d)-bs-1,(batch,),generator=g)
            x=torch.stack([d[i:i+bs] for i in ix]).to(dev); y=torch.stack([d[i+1:i+1+bs] for i in ix]).to(dev)
            logits,_,_=model(x,None,None,neuron_mask=mask) if mask is not None else model(x)
            ls.append(torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1)).item())
    return math.exp(sum(ls)/len(ls))
ones=torch.ones(Ntot,device=dev)
print("\ndomain | exit(matrix) ppl | FREE eval | MASKED to own prefix | masked recovers?")
rec=0
for i,l in enumerate(SEQ):
    w=8192+2048*i
    m=ones.clone(); m[w:]=0.0
    fr=ev(l); mk=ev(l,m)
    print(f"  {l}   |  {w:>6d} wide  | {fr:8.2f} | {mk:8.2f} | {'YES' if mk<fr*0.7 else 'partial' if mk<fr*0.95 else 'no'}")
    rec+= mk<fr*0.7
print(f"\n{rec}/20 domains recover >30% under own-prefix masking")
