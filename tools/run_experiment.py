import argparse
import shutil
import time
import traceback
import uuid
import numpy as np
from common import *
from learning import *
from chain_transport import Chain,ZERO,Web3

MODEL_KEYS=('w1','b1','w2','b2','wc','bc')

def validate_artifact(path,manifest,expected_counts,prior,mu):
    if filehash(path)!=manifest['artifact_sha256']:raise ValueError('ARTIFACT_HASH')
    with np.load(path,allow_pickle=False) as a:
        for key in a.files:
            if not np.all(np.isfinite(a[key])):raise ValueError('NONFINITE')
        if not np.array_equal(a['counts'],expected_counts):raise ValueError('CLASS_COUNTS')
        expected=shrink_prototypes(a['raw_proto'],a['counts'],prior,mu)
        if not np.allclose(expected,a['shrunk_proto'],rtol=1e-6,atol=1e-7):raise ValueError('SHRINK_RULE')
        return {k:a[k].copy() for k in a.files}

def test_revert(chain,account,method,args,label,out):
    rec=chain.settle([chain.send(account,method,args,label)],expected=0)[0]
    append(out/'negative_cases.jsonl',{'utc':now(),'case':label,'transaction_hash':rec['hash'],'expected_rejected':True,'receipt_status':rec['receipt']['status']})

def run_arm(out,chain,cfg,arm,seed,rounds,do_negatives):
    arm_id=arm['id'];rid=f'seed{seed}_{arm_id}';dest=out/rid;dest.mkdir()
    samples=np.load(BASE/'I23-INPUT/emnist_subset.npz',allow_pickle=False)
    x=samples['train_x'].reshape(-1,784).astype(np.float32)/255
    tx=samples['test_x'].reshape(-1,784).astype(np.float32)/255
    y=samples['train_y'].astype(np.int64);ty=samples['test_y'].astype(np.int64)
    partition=np.load(BASE/f'I23-INPUT/partition_{seed}.npz',allow_pickle=False)
    assign=partition['train_client'];test_assign=partition['test_client']
    counts=np.array([np.bincount(y[assign==k],minlength=cfg['classes']) for k in range(cfg['clients'])])
    rng=np.random.default_rng(seed);model=init_model(784,cfg['hidden'],cfg['embedding'],cfg['classes'],rng)
    prior=np.zeros((cfg['classes'],cfg['embedding']),dtype=np.float64)
    task=bytes.fromhex(digest((out.name+'/'+rid).encode('utf-8')))
    input_hash=digest((filehash(BASE/'I23-INPUT/emnist_subset.npz')+filehash(BASE/f'I23-INPUT/partition_{seed}.npz')).encode('ascii'))
    np.savez(dest/'initial_model.npz',**model)
    init_model_hash=filehash(dest/'initial_model.npz')
    save(dest/'run.json',{'utc':now(),'run_id':out.name+'/'+rid,'suite_id':out.name,'task_id':'0x'+task.hex(),'seed':seed,'arm':arm,'rounds':rounds,'config_sha256':filehash(BASE/'I23-INPUT/config.json'),'input_hash':input_hash,'initial_model_sha256':init_model_hash,'test_use':'evaluation_only','classification':'new_actual_run'})
    chain.settle([chain.send(chain.owner,'create',[task,bytes.fromhex(filehash(BASE/'I23-INPUT/config.json')),bytes.fromhex(input_hash),bytes.fromhex(init_model_hash),[a.address for a in chain.clients]],rid+'/create')])
    prior_hash=ZERO
    for rnd in range(rounds+1):
        start=time.perf_counter();started=now();rd=dest/f'round_{rnd:02d}';rd.mkdir()
        current=chain.state(task)
        assert current['next_round']==rnd and current['prototype_hash']==prior_hash and current['model_hash']=='0x'+init_model_hash
        s=prototype_similarity(prior)
        submissions=[];manifests=[];locals_=[];small_metrics=[]
        if do_negatives and rnd==0:
            dummy=b'\x01'*32
            test_revert(chain,chain.outsider,'submit',[task,rnd,bytes.fromhex(prior_hash[2:]),dummy],'unauthorized_client',out)
            test_revert(chain,chain.clients[0],'submit',[task,rnd+1,bytes.fromhex(prior_hash[2:]),dummy],'wrong_round',out)
            test_revert(chain,chain.clients[0],'submit',[task,rnd,dummy,dummy],'wrong_prior',out)
            test_revert(chain,chain.owner,'finalize',[task,rnd,dummy,dummy,dummy],'incomplete_submissions',out)
        for k,account in enumerate(chain.clients):
            local={key:model[key].copy() for key in MODEL_KEYS}
            ix=np.flatnonzero(assign==k);client_started=now();loss_sums={'ce':0.,'wpc':0.,'regularization':0.,'objective':0.};steps=0
            if rnd:
                order_rng=np.random.default_rng(np.random.SeedSequence([seed,rnd,k,723]))
                for epoch in range(cfg['local_epochs']):
                    order=order_rng.permutation(ix)
                    for pos in range(0,len(order),cfg['batch_size']):
                        b=order[pos:pos+cfg['batch_size']]
                        loss,grads,parts=objective(local,x[b],y[b],s,arm['wpc'],arm['alpha'],cfg['tau'],cfg['weight_decay'])
                        if not np.isfinite(loss):raise FloatingPointError('NONFINITE_LOSS')
                        for key in MODEL_KEYS:local[key]-=cfg['learning_rate']*grads[key]
                        for key,v in parts.items():loss_sums[key]+=v*len(b)
                        loss_sums['objective']+=loss*len(b);steps+=1
            _,z,_=forward(local,x[ix]);raw,n=local_prototypes(z,y[ix],cfg['classes'])
            mu=arm['mu'] if rnd else 0.
            shrunk=shrink_prototypes(raw,n,prior,mu)
            extra={}
            if rnd==rounds:
                _,test_z,_=forward(local,tx);test_ref,test_counts=local_prototypes(test_z,ty,cfg['classes'])
                extra={'test_reference_proto':test_ref,'test_reference_counts':test_counts}
                raw_mse=((raw-test_ref)**2).mean(axis=1);shrunk_mse=((shrunk-test_ref)**2).mean(axis=1)
                for c in np.flatnonzero((n>0)&(n<=5)):
                    small_metrics.append({'client':k,'class':int(c),'n':int(n[c]),'raw_mse':float(raw_mse[c]),'shrunk_mse':float(shrunk_mse[c])})
            path=rd/f'client_{k}.npz'
            np.savez(path,**local,raw_proto=raw,shrunk_proto=shrunk,counts=n,**extra)
            manifest={'suite_id':out.name,'run_id':rid,'task_id':'0x'+task.hex(),'round':rnd,'client':k,'address':account.address,'started_utc':client_started,'finished_utc':now(),'artifact_path':path.relative_to(out).as_posix(),'artifact_sha256':filehash(path),'config_sha256':filehash(BASE/'I23-INPUT/config.json'),'input_hash':input_hash,'prior_prototype_sha256':prior_hash[2:],'prior_model_sha256':init_model_hash,'n_samples':len(ix),'class_counts':n.tolist(),'mu_effective':mu,'wpc_weight':arm['wpc'],'steps':steps,'loss_mean':{key:v/(len(ix)*cfg['local_epochs']) if rnd else None for key,v in loss_sums.items()}}
            mp=rd/f'client_{k}.json';save(mp,manifest);manifests.append((mp,manifest))
            submissions.append(chain.send(account,'submit',[task,rnd,bytes.fromhex(prior_hash[2:]),bytes.fromhex(filehash(mp))],f'{rid}/r{rnd}/client{k}'))
        receipts=chain.settle(submissions)
        if do_negatives and rnd==0:
            mp=manifests[0][0]
            test_revert(chain,chain.clients[0],'submit',[task,rnd,bytes.fromhex(prior_hash[2:]),bytes.fromhex(filehash(mp))],'duplicate_client',out)
            test_revert(chain,chain.owner,'finalize',[task,rnd,b'\x01'*32,b'\x01'*32,b'\x01'*32],'wrong_submission_root',out)
        for k,(mp,manifest) in enumerate(manifests):
            onchain=chain.call(0,'updates',[task,rnd,chain.clients[k].address],['bytes32'])[0]
            assert onchain=='0x'+filehash(mp)
            a=validate_artifact(out/manifest['artifact_path'],manifest,counts[k],prior,arm['mu'] if rnd else 0.)
            # Recompute the claimed local prototypes from actual local images and update.
            ix=np.flatnonzero(assign==k);_,z,_=forward(a,x[ix]);check_raw,check_counts=local_prototypes(z,y[ix],cfg['classes'])
            assert np.allclose(check_raw,a['raw_proto'],rtol=1e-6,atol=1e-7)
            locals_.append(a)
        if do_negatives and rnd==0:
            mp,m=manifests[0]
            for case,alt_counts,alt_m in [('wrong_artifact_hash',counts[0],{**m,'artifact_sha256':'00'*32}),('wrong_sample_count',counts[0]+1,m)]:
                try:validate_artifact(out/m['artifact_path'],alt_m,alt_counts,prior,0.)
                except ValueError as e:append(out/'negative_cases.jsonl',{'utc':now(),'case':case,'expected_rejected':True,'exception':str(e),'artifact_path':m['artifact_path']})
                else:raise AssertionError(case)
        if rnd:
            total=counts.sum()
            model={key:sum(a[key]*(counts[k].sum()/total) for k,a in enumerate(locals_)).astype(np.float32) for key in MODEL_KEYS}
        prior=aggregate_prototypes(np.array([a['shrunk_proto'] for a in locals_]),counts)
        cp=rd/'global_model.npz';pp=rd/'global_prototypes.npy'
        np.savez(cp,**model);np.save(pp,prior,allow_pickle=False)
        root=Web3.keccak(b''.join(bytes.fromhex(filehash(mp)) for mp,m in manifests))
        rc=chain.settle([chain.send(chain.owner,'finalize',[task,rnd,bytes.fromhex(filehash(cp)),bytes.fromhex(filehash(pp)),root],f'{rid}/r{rnd}/finalize')])[0]
        tag=rc['receipt']['blockNumber'];chain.sync(tag)
        states=[chain.state(task,tag,i) for i in range(2)]
        assert states[0]==states[1]
        assert states[0]['model_hash']=='0x'+filehash(cp) and states[0]['prototype_hash']=='0x'+filehash(pp) and states[0]['next_round']==rnd+1
        onchain_updates=[chain.call(1,'updates',[task,rnd,chain.clients[k].address],['bytes32'],tag)[0] for k in range(cfg['clients'])]
        save(rd/'chain_link.json',{'utc':now(),'block_number':tag,'finalize_transaction':rc['hash'],'client_transactions':[r['hash'] for r in receipts],'submissions_root':'0x'+root.hex(),'node_states':states,'node2_update_commitments':onchain_updates,'model_sha256':filehash(cp),'prototype_sha256':filehash(pp),'prior_sha256':prior_hash[2:]})
        pred,ce=predictions(model,tx,prior)
        np.savez_compressed(rd/'test_predictions.npz',prototype_pred=pred,ce_pred=ce)
        score=float(np.mean(pred==ty));ce_score=float(np.mean(ce==ty))
        record={'utc':now(),'started_utc':started,'run_id':rid,'round':rnd,'prototype_accuracy':score,'ce_accuracy':ce_score,'client_test_accuracy':[float(np.mean(pred[test_assign==k]==ty[test_assign==k])) for k in range(cfg['clients'])],'elapsed_seconds':time.perf_counter()-start,'model_sha256':filehash(cp),'prototype_sha256':filehash(pp),'chain_block':tag,'finalize_transaction':rc['hash']}
        append(dest/'rounds.jsonl',record)
        print(json.dumps(record),flush=True)
        prior_hash='0x'+filehash(pp);init_model_hash=filehash(cp)
    save(dest/'small_sample_estimates.json',small_metrics)
    save(dest/'completed.json',{'utc':now(),'rounds':rounds,'final_prototype_accuracy':score,'final_ce_accuracy':ce_score})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    cfg=read(BASE/'I23-INPUT/config.json')
    suite=('smoke_' if args.smoke else 'main_')+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    out=BASE/'I23-RESULTS'/suite;out.mkdir(parents=True,exist_ok=False)
    code=out/'code';code.mkdir()
    hashes={}
    for p in sorted((BASE/'tools').iterdir()):
        if p.suffix in {'.py','.sol'}:
            shutil.copy2(p,code/p.name);hashes[p.name]=filehash(p)
    save(out/'start.json',{'utc':now(),'suite_id':suite,'purpose':'integration_smoke_not_independent_main_repeat' if args.smoke else 'frozen_main_experiment','command':sys.argv,'config_sha256':filehash(BASE/'I23-INPUT/config.json'),'code_hashes':hashes,'code_version_sha256':digest(json.dumps(hashes,sort_keys=True).encode('utf-8')),'seeds':cfg['seeds'][:1] if args.smoke else cfg['seeds'],'arms':['WS'] if args.smoke else [a['id'] for a in cfg['arms']],'rounds':1 if args.smoke else cfg['rounds']})
    chain=Chain(out,cfg['ports'],cfg['chain_id'],cfg['clients'])
    try:
        chain.start();chain.deploy();first=True
        for seed in cfg['seeds'][:1] if args.smoke else cfg['seeds']:
            for arm in cfg['arms'][-1:] if args.smoke else cfg['arms']:
                run_arm(out,chain,cfg,arm,seed,1 if args.smoke else cfg['rounds'],first)
                first=False
        save(out/'completed.json',{'utc':now(),'status':'completed'})
    except BaseException as exc:
        save(out/'failure.json',{'utc':now(),'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:chain.stop()
    print('SUITE_COMPLETED '+str(out),flush=True)

if __name__=='__main__':main()
