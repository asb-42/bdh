import torch
OUT="out/"; DM,NH,BLK=512,8,2048
SEQ="en es pl fr de cs da pt fi hu bg it et el sk sv ro nl sl lt".split()
def L(p): return torch.load(OUT+p,map_location="cpu",mmap=True,weights_only=False)
print("transition | decoder-old-bitwise-equal | enc churn(old cols) | enc_v churn | enc-old v-nonzero | new-block v-nonzero")
for i in [1,4,9,14,19]:
    lang=SEQ[i]; par=SEQ[i-1]
    c=L(f"bdh_europarl_ladRA2b-{lang}_last.pt"); p=L(f"bdh_europarl_ladRA2b-{par}_last.pt")
    pc=int(c["cfg"]["mlp_internal_dim_multiplier"])*DM//NH; pp=int(p["cfg"]["mlp_internal_dim_multiplier"])*DM//NH
    st=c["optimizer_state"]["state"]
    dec_eq=float((c["model_state"]["decoder"][:pp*NH]==p["model_state"]["decoder"]).float().mean())
    out={}
    for nm,idx in [("encoder",1),("encoder_v",2)]:
        A=p["model_state"][nm][:,:,:pp]; B=c["model_state"][nm][:,:,:pp]
        out[nm+"_churn"]=float(((B.float()-A.float()).pow(2).sum()/A.float().pow(2).sum())**.5)
        out[nm+"_voldnz"]=int((st[idx]["exp_avg_sq"][:,:,:pp]!=0).sum())
    newnz=int((st[0]["exp_avg_sq"][:, :][:NH*BLK*0].numel() and sum(int((st[0]["exp_avg_sq"][h*pc+pp:h*pc+pc]!=0).sum()) for h in range(NH))))
    print(f" {par}->{lang:<4s} | {dec_eq:.6f} | {out['encoder_churn']:.6f} | {out['encoder_v_churn']:.6f} | {out['encoder_voldnz']:>9d} | {newnz:>9d}")
    del c,p,st
