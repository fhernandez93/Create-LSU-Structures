"""Parity test: scipy L-BFGS(100) vs on-device Adam vs on-device Barzilai-Borwein
GD, on a battery of post-SW relaxes from a near-annealed config. BB uses secant
step sizes (quasi-Newton, O(N)) and often matches L-BFGS. Gate: mean dE ~ 0
(no systematic under-relaxation). Frozen atoms must stay put.

Usage: python -m Claude_Helpers._dev_relax_methods [start_rodfile] [nmoves]
"""
import sys
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
import tools, lsu_network as lsu

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4)
SHELL = 4
START = sys.argv[1] if len(sys.argv) > 1 else None
NTEST = int(sys.argv[2]) if len(sys.argv) > 2 else 10
vg = lsu._value_and_grad_jit


@partial(jax.jit, static_argnames=("n_iters",))
def relax_adam(x0, e_, t_, q_, bx, d0, w, mask, n_iters, lr):
    b1,b2,eps=0.9,0.999,1e-8
    def body(i,c):
        x,m,v=c; _,g=vg(x,e_,t_,q_,bx,d0,w); g=g*mask
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g; t=i.astype(x.dtype)+1.0
        return (x-lr*(m/(1-b1**t))/(jnp.sqrt(v/(1-b2**t))+eps), m, v)
    z=jnp.zeros_like(x0); x,_,_=jax.lax.fori_loop(0,n_iters,body,(x0,z,z))
    e,_=vg(x,e_,t_,q_,bx,d0,w); return x,e


@partial(jax.jit, static_argnames=("n_iters",))
def relax_bb(x0, e_, t_, q_, bx, d0, w, mask, n_iters, alpha0):
    _,g0=vg(x0,e_,t_,q_,bx,d0,w); g0=g0*mask
    x1=x0-alpha0*g0
    def body(i,c):
        xp,gp,x=c
        _,g=vg(x,e_,t_,q_,bx,d0,w); g=g*mask
        s=x-xp; y=g-gp
        sy=jnp.sum(s*y); yy=jnp.sum(y*y); ss=jnp.sum(s*s)
        # BB1 step (ss/sy) with safeguards; fall back to alpha0 if ill-defined
        alpha=jnp.where((sy>1e-30)&(yy>1e-30), ss/jnp.maximum(sy,1e-30), alpha0)
        alpha=jnp.clip(alpha,1e-7,0.5)
        return (x,g,x-alpha*g)
    xp,gp,x=jax.lax.fori_loop(0,n_iters,body,(x0,g0,x1))
    e,_=vg(x,e_,t_,q_,bx,d0,w); return x,e


# starting config
rng=np.random.default_rng(42)
if START:
    rods=np.loadtxt(START); pos,edges=tools.rods_to_network(rods,box)
else:
    pos,edges,_=lsu.random_seed_network_bm2000(N,box,D0,rng,verbose=False)
nb=lsu.build_neighbors(N,edges)
ctx=lsu._RelaxContext(N,box,D0,W,use_jax=True,use_jaxopt=False); ctx.update_topology(edges,nb)
if not START:
    pos,_=lsu.settle_seed_with_repulsion(pos,ctx,edges,box,D0,verbose=False)
ctx.set_moving_mask(None); pos,_,_=lsu.relax(pos,ctx,max_iter=1500)
print(f"start E/atom={float(ctx.energy(pos.ravel()))/N:.4f}", flush=True)
boxj=jnp.asarray(box); d0j=jnp.float64(D0); wj=jnp.asarray(W)

rng2=np.random.default_rng(1)
res={'adam':[], 'bb':[]}
print(f"{'mv':>3} {'E_scipy':>10} {'E_adam':>10} {'E_bb':>10} {'dE_adam':>8} {'dE_bb':>8} {'froz_bb':>9}", flush=True)
for k in range(NTEST):
    move=lsu.stone_wales_propose(edges,nb,rng2,max_tries=40)
    if move is None: continue
    _e1,(i,c,j,d),_e2=move; pb=pos.copy()
    lsu.stone_wales_apply(edges,nb,move)
    if not lsu.is_connected(N,edges): lsu.stone_wales_revert(edges,nb,move); continue
    ctx.update_topology(edges,nb)
    shell=lsu.compute_local_shell_mask(np.array([i,c,j,d]),nb,SHELL,N); ctx.set_moving_mask(shell)
    mflat=jnp.asarray(np.broadcast_to(shell[:,None],(N,3)).reshape(-1).astype(np.float64))
    ej=jnp.asarray(ctx.edges,jnp.int32); tj=jnp.asarray(ctx.triples,jnp.int32); qj=jnp.asarray(ctx.quads,jnp.int32)
    x0=jnp.asarray(pb.ravel())
    _,Es,_=lsu.relax(pb,ctx,max_iter=100,E_threshold=float("inf"))
    _,Ea=relax_adam(x0,ej,tj,qj,boxj,d0j,wj,mflat,1000,2e-3); Ea=float(Ea)
    xb,Eb=relax_bb(x0,ej,tj,qj,boxj,d0j,wj,mflat,150,1e-3); Eb=float(Eb)
    fro=float(np.abs(lsu.pbc_displacement(np.asarray(xb).reshape(N,3)[~shell]-pb[~shell],box)).max())
    res['adam'].append(Ea-Es); res['bb'].append(Eb-Es)
    print(f"{k:>3} {Es:>10.4f} {Ea:>10.4f} {Eb:>10.4f} {Ea-Es:>+8.4f} {Eb-Es:>+8.4f} {fro:>9.1e}", flush=True)
    lsu.stone_wales_revert(edges,nb,move); ctx.update_topology(edges,nb); pos=pb

for m in ('adam','bb'):
    a=np.array(res[m]);
    if len(a): print(f"{m}: mean dE={a.mean():+.4f}  <=scipy on {int((a<=1e-3).sum())}/{len(a)}", flush=True)
