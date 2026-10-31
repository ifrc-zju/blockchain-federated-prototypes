"""Independent NumPy implementation of paper equations (6)-(13).

No paper scores or historical logs are read. Global priors are detached inputs.
"""
import numpy as np

def normalize(x):
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    return x/np.maximum(norms,1e-12)

def prototype_similarity(prototypes):
    p=normalize(prototypes)
    return np.maximum(0,p@p.T)  # Eq. (6)

def wpc_loss(z,y,similarity,alpha,tau):
    """Eqs. (7)-(10): weighted sample-sample SupCon using global class S.

    Off-diagonal only. Anchors with zero weight sum contribute zero; the mean
    retains the full batch denominator N. No gradients enter the prior S.
    """
    n=len(y)
    if n<2: return 0.,np.zeros_like(z)
    u=normalize(z)
    mask=~np.eye(n,dtype=bool)
    weights=np.where(y[:,None]==y[None,:],1.,alpha*similarity[y[:,None],y[None,:]])
    weights=np.where(mask,weights,0)
    sums=weights.sum(axis=1,keepdims=True)
    targets=weights/np.maximum(sums,1e-12)
    valid=(sums[:,0]>0)
    logits=(u@u.T)/tau
    logits=np.where(mask,logits,-1e30)
    shift=logits.max(axis=1,keepdims=True)
    exp=np.where(mask,np.exp(logits-shift),0)
    probs=exp/exp.sum(axis=1,keepdims=True)
    logp=logits-shift-np.log(exp.sum(axis=1,keepdims=True))
    loss=float(-(targets*np.where(mask,logp,0)).sum()/n)
    a=(probs-targets)*valid[:,None]/(n*tau)
    du=(a+a.T)@u
    norms=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    dz=(du-u*(du*u).sum(axis=1,keepdims=True))/norms
    return loss,dz

def shrink_prototypes(raw,counts,prior,mu):
    """Eq. (13); absent classes are masked, never counted as observations."""
    counts=np.asarray(counts)
    denom=np.maximum(counts+mu,1e-12)[:,None]
    result=counts[:,None]/denom*raw+mu/denom*prior
    return np.where(counts[:,None]>0,result,0)

def local_prototypes(z,y,classes):
    counts=np.bincount(y,minlength=classes)
    sums=np.zeros((classes,z.shape[1]),dtype=z.dtype)
    np.add.at(sums,y,z)
    return sums/np.maximum(counts[:,None],1),counts

def aggregate_prototypes(local,counts):
    """Eq. (12), using class counts, including only actually present classes."""
    return (local*counts[:,:,None]).sum(axis=0)/np.maximum(counts.sum(axis=0)[:,None],1)

def init_model(inputs,hidden,embedding,classes,rng,dtype=np.float32):
    return {'w1':(rng.normal(size=(inputs,hidden))*np.sqrt(1/inputs)).astype(dtype),
            'b1':np.zeros(hidden,dtype=dtype),
            'w2':(rng.normal(size=(hidden,embedding))*np.sqrt(1/hidden)).astype(dtype),
            'b2':np.zeros(embedding,dtype=dtype),
            'wc':(rng.normal(size=(embedding,classes))*np.sqrt(1/embedding)).astype(dtype),
            'bc':np.zeros(classes,dtype=dtype)}

def forward(model,x):
    h=np.tanh(x@model['w1']+model['b1'])
    z=h@model['w2']+model['b2']
    return h,z,z@model['wc']+model['bc']

def objective(model,x,y,similarity,weight,alpha,tau,decay):
    h,z,logits=forward(model,x)
    logits=logits-logits.max(axis=1,keepdims=True)
    exp=np.exp(logits); probs=exp/exp.sum(axis=1,keepdims=True)
    ce=float((-logits[np.arange(len(y)),y]+np.log(exp.sum(axis=1))).mean())
    dl=probs.copy(); dl[np.arange(len(y)),y]-=1; dl/=len(y)
    contrast,dzcontrast=wpc_loss(z,y,similarity,alpha,tau) if weight else (0.,np.zeros_like(z))
    dz=dl@model['wc'].T+weight*dzcontrast
    dh=(dz@model['w2'].T)*(1-h*h)
    grads={'wc':z.T@dl,'bc':dl.sum(axis=0),'w2':h.T@dz,'b2':dz.sum(axis=0),'w1':x.T@dh,'b1':dh.sum(axis=0)}
    reg=0.
    for k in model:
        if k.startswith('w'):
            reg+=.5*decay*float((model[k]*model[k]).sum())
            grads[k]+=decay*model[k]
        grads[k]=grads[k].astype(model[k].dtype)
    return ce+weight*contrast+reg,grads,{'ce':ce,'wpc':contrast,'regularization':reg}

def predictions(model,x,prototypes):
    _,z,logits=forward(model,x)
    return np.argmax(normalize(z)@normalize(prototypes).T,axis=1),np.argmax(logits,axis=1)
