from runtime import executable
"""Independent forward pass, gradients, aggregation, statistics and chain replay.

Does NOT import learning.py or run_experiment.py. Reopens the archived second
Geth node without mining or peers and queries historical contract state.
"""
import argparse
import csv
import math
import shutil
import subprocess
import time
import urllib.request
import uuid
import numpy as np
from common import *
from web3 import Web3
from eth_account import Account

KEYS=('w1','b1','w2','b2','wc','bc')

def unit(v):return v/np.maximum(np.sqrt(np.sum(v*v,axis=1,keepdims=True)),1e-12)

def infer(m,x):
    h=np.tanh(np.dot(x,m['w1'])+m['b1']);z=np.dot(h,m['w2'])+m['b2']
    return h,z,np.dot(z,m['wc'])+m['bc']

def centers(z,y,c):
    # Deliberately different reduction from the implementation's np.add.at.
    out=np.zeros((c,z.shape[1]),dtype=np.float64)
    for label in range(c):
        if np.any(y==label):out[label]=z[y==label].mean(axis=0,dtype=np.float64)
    return out

def replay_client(initial,x,y,prior,cfg,arm,seed,rnd,k):
    m={key:initial[key].copy() for key in KEYS}
    if rnd==0:return m
    rng=np.random.default_rng(np.random.SeedSequence([seed,rnd,k,723]))
    sim=np.maximum(0,unit(prior)@unit(prior).T)
    for epoch in range(cfg['local_epochs']):
        order=rng.permutation(len(x))
        for start in range(0,len(x),cfg['batch_size']):
            ix=order[start:start+cfg['batch_size']];xb=x[ix];yb=y[ix];n=len(ix)
            h,z,logits=infer(m,xb);logits-=logits.max(axis=1,keepdims=True)
            q=np.exp(logits);q/=q.sum(axis=1,keepdims=True);q[np.arange(n),yb]-=1;q/=n
            dz=q@m['wc'].T
            if arm['wpc'] and n>1:
                u=unit(z);off=~np.eye(n,dtype=bool)
                weights=np.where(yb[:,None]==yb[None,:],1.,arm['alpha']*sim[yb[:,None],yb[None,:]])*off
                denom=weights.sum(axis=1,keepdims=True)
                target=weights/np.maximum(denom,1e-12)
                a=np.where(off,(u@u.T)/cfg['tau'],-1e30)
                a-=a.max(axis=1,keepdims=True);p=np.where(off,np.exp(a),0);p/=p.sum(axis=1,keepdims=True)
                d=(p-target)*(denom>0)/(n*cfg['tau'])
                v=(d+d.T)@u
                g=(v-u*(v*u).sum(axis=1,keepdims=True))/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
                dz=dz+arm['wpc']*g
            dh=(dz@m['w2'].T)*(1-h*h)
            grads={'w1':xb.T@dh,'b1':dh.sum(axis=0),'w2':h.T@dz,'b2':dz.sum(axis=0),'wc':z.T@q,'bc':q.sum(axis=0)}
            for key in KEYS:
                if key.startswith('w'):grads[key]+=cfg['weight_decay']*m[key]
                m[key]-=cfg['learning_rate']*grads[key].astype(m[key].dtype)
    return m

def write_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def rpc(method,params=None):
    req=urllib.request.Request('http://127.0.0.1:19625',json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params or []}).encode('utf-8'),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as f:reply=json.load(f)
    if 'error' in reply:raise RuntimeError(reply['error'])
    return reply['result']

def near(a,b,label,atol=6e-6,rtol=2e-5):
    if not np.allclose(a,b,atol=atol,rtol=rtol):raise AssertionError((label,float(np.max(np.abs(a-b)))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('suite');args=ap.parse_args()
    suite=BASE/'I23-RESULTS'/args.suite
    assert (suite/'completed.json').exists()
    out=suite/f'audit_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_{uuid.uuid4().hex[:6]}';out.mkdir()
    save(out/'start.json',{'utc':now(),'command':sys.argv,'audit_code_sha256':filehash(__file__),'policy':'independent numpy formulas plus non-mining archived node historical calls'})
    shutil.copy2(__file__,out/'audit_code.py')
    cfg=read(BASE/'I23-INPUT/config.json');start=read(suite/'start.json')
    assert start['config_sha256']==filehash(BASE/'I23-INPUT/config.json')
    for name,h in start['code_hashes'].items():assert filehash(suite/'code'/name)==h
    for rec in read(BASE/'I23-INPUT/source_manifest.json'):assert filehash(ROOT/rec['path'])==rec['sha256']
    with np.load(BASE/'I23-INPUT/emnist_subset.npz',allow_pickle=False) as archive:
        data={key:archive[key] for key in archive.files}
    x=data['train_x'].reshape(-1,784).astype(np.float32)/255;tx=data['test_x'].reshape(-1,784).astype(np.float32)/255
    y=data['train_y'].astype(np.int64);ty=data['test_y'].astype(np.int64)
    source=ROOT/cfg['data_source']
    for split in ['train','test']:
        labels=np.memmap(source/f'emnist-balanced-{split}-labels-idx1-ubyte',mode='r',dtype=np.uint8,offset=8)
        pixels=np.memmap(source/f'emnist-balanced-{split}-images-idx3-ubyte',mode='r',dtype=np.uint8,offset=16,shape=(len(labels),28,28))
        ix=data[split+'_source_index'];assert len(ix)==len(set(ix.tolist()))
        np.testing.assert_array_equal(data[split+'_x'],pixels[ix]);np.testing.assert_array_equal(data[split+'_y'],labels[ix])
    with (BASE/'I23-INPUT/sample_index.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):assert digest(data[row['split']+'_x'][int(row['subset_index'])].tobytes())==row['image_sha256']
    deployments=read(suite/'chain/deployment.json');contract=Web3().eth.contract(abi=deployments['abi'])
    def call(fn,args,types,tag):
        raw=rpc('eth_call',[{'to':deployments['address'],'data':contract.encode_abi(fn,args=args)},tag])
        return clean(Web3().codec.decode(types,bytes.fromhex(raw[2:])))
    cmd=[executable('geth'),'--datadir',str(suite/'chain/node2'),'--networkid',str(cfg['chain_id']),'--syncmode','full','--gcmode','archive','--state.scheme','hash','--cache','64','--maxpeers','0','--nodiscover','--nat','none','--netrestrict','127.0.0.0/8','--port','0','--http','--http.addr','127.0.0.1','--http.port','19625','--http.api','eth,net,web3','--http.vhosts','localhost,127.0.0.1','--ipcdisable','--authrpc.addr','127.0.0.1','--authrpc.port','39625','--verbosity','2']
    save(out/'node_command.json',{'utc':now(),'command':cmd,'mining':False,'peers':0})
    log=(out/'archived_node.log').open('xb');proc=subprocess.Popen(cmd,stdout=log,stderr=log,creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    summary=[];curve=[];replay_count=0;max_replay_error=0.;chain_checks=0
    try:
        deadline=time.monotonic()+35
        while True:
            try:version=rpc('web3_clientVersion');break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(.2)
        assert rpc('eth_getCode',[deployments['address'],'latest'])==deployments['runtime_code'][1]
        assert rpc('net_peerCount')=='0x0'
        receipts=[json.loads(line) for line in (suite/'chain/receipts.jsonl').read_text(encoding='utf-8').splitlines()]
        submitted=[json.loads(line) for line in (suite/'chain/submitted.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(receipts)==len(submitted)==len({r['hash'] for r in receipts})
        byhash={r['hash']:r for r in receipts}
        for rec in receipts:
            h='0x'+Web3.keccak(bytes.fromhex(rec['raw'][2:])).hex()
            assert h==rec['hash']
            assert Account.recover_transaction(rec['raw'])==rec['from']
            real=rpc('eth_getTransactionReceipt',[h]);assert real==rec['receipt']
            tx_real=rpc('eth_getTransactionByHash',[h]);assert tx_real['input'].lower()==rec['tx_input']['data'].lower()
            assert tx_real['from'].lower()==rec['from'].lower()
            assert int(real['status'],16)==rec['expected_status']
            b=rpc('eth_getBlockByHash',[real['blockHash'],False]);assert h in b['transactions'] and b['stateRoot']==rec['block']['stateRoot']
            chain_checks+=1
        for dest in sorted(suite.glob('seed*')):
            run=read(dest/'run.json');seed=run['seed'];arm=run['arm'];task=bytes.fromhex(run['task_id'][2:]);rounds=run['rounds']
            part=np.load(BASE/f'I23-INPUT/partition_{seed}.npz',allow_pickle=False);assign=part['train_client'];test_assign=part['test_client']
            counts=np.array([np.bincount(y[assign==k],minlength=cfg['classes']) for k in range(cfg['clients'])])
            records=[json.loads(line) for line in (dest/'rounds.jsonl').read_text(encoding='utf-8').splitlines()]
            assert [r['round'] for r in records]==list(range(rounds+1))
            prior=np.zeros((cfg['classes'],cfg['embedding']));prior_hash='00'*32
            with np.load(dest/'initial_model.npz') as a:initial={k:a[k].copy() for k in KEYS}
            model_hash=filehash(dest/'initial_model.npz');small=[]
            for rnd in range(rounds+1):
                rd=dest/f'round_{rnd:02d}';link=read(rd/'chain_link.json');tag=link['block_number'];model_arrays=[];proto_arrays=[];commitments=[]
                for k in range(cfg['clients']):
                    manifest=read(rd/f'client_{k}.json');path=rd/f'client_{k}.npz'
                    assert manifest['artifact_sha256']==filehash(path) and manifest['prior_prototype_sha256']==prior_hash and manifest['prior_model_sha256']==model_hash
                    assert manifest['class_counts']==counts[k].tolist() and manifest['n_samples']==int(counts[k].sum())
                    commitment='0x'+filehash(rd/f'client_{k}.json');commitments.append(commitment)
                    assert call('updates',[task,rnd,manifest['address']],['bytes32'],tag)==[commitment]
                    tx_rec=byhash[link['client_transactions'][k]]
                    fn,arguments=contract.decode_function_input(tx_rec['transaction']['input'])
                    assert fn.fn_name=='submit' and clean(arguments['id'])==run['task_id'] and arguments['round']==rnd
                    assert clean(arguments['prior'])=='0x'+prior_hash and clean(arguments['commitment'])==commitment
                    assert tx_rec['from']==manifest['address']
                    with np.load(path,allow_pickle=False) as a:
                        local={key:a[key].copy() for key in KEYS};n=a['counts'];raw=a['raw_proto'];shrunk=a['shrunk_proto']
                        np.testing.assert_array_equal(n,counts[k]);assert all(np.isfinite(a[key]).all() for key in a.files)
                        replay=replay_client(initial,x[assign==k],y[assign==k],prior,cfg,arm,seed,rnd,k)
                        error=max(float(np.max(np.abs(local[key]-replay[key]))) for key in KEYS)
                        for key in KEYS:near(local[key],replay[key],'TRAINING_REPLAY/'+key,atol=8e-6,rtol=3e-5)
                        max_replay_error=max(max_replay_error,error);replay_count+=int(rnd>0)
                        append(out/'training_replay.jsonl',{'run_id':dest.name,'round':rnd,'client':k,'max_abs_parameter_error':error,'passed':True})
                        _,z,_=infer(local,x[assign==k]);expected_raw=centers(z,y[assign==k],cfg['classes']);near(raw,expected_raw,'RAW_PROTO')
                        mu=arm['mu'] if rnd else 0.;weight=n/np.maximum(n+mu,1e-12)
                        expected=weight[:,None]*raw+(1-weight[:,None])*prior;expected[n==0]=0
                        near(shrunk,expected,'SHRINK')
                        model_arrays.append(local);proto_arrays.append(shrunk.copy())
                        if rnd==rounds:
                            _,ztest,_=infer(local,tx);reference=centers(ztest,ty,cfg['classes']);near(a['test_reference_proto'],reference,'TEST_CENTROID')
                            for c in np.flatnonzero((n>0)&(n<=5)):
                                small.append({'client':k,'class':int(c),'n':int(n[c]),'raw_mse':float(np.mean((raw[c]-reference[c])**2)),'shrunk_mse':float(np.mean((shrunk[c]-reference[c])**2))})
                with np.load(rd/'global_model.npz',allow_pickle=False) as a:glob={key:a[key].copy() for key in KEYS}
                if rnd:
                    for key in KEYS:
                        expected=np.average(np.stack([m[key] for m in model_arrays]),axis=0,weights=counts.sum(axis=1))
                        near(glob[key],expected,'FEDAVG',atol=1e-6,rtol=1e-5)
                else:
                    for key in KEYS:np.testing.assert_array_equal(glob[key],initial[key])
                expected_proto=sum(proto_arrays[k]*counts[k,:,None] for k in range(cfg['clients']))/counts.sum(axis=0)[:,None]
                global_proto=np.load(rd/'global_prototypes.npy',allow_pickle=False);near(global_proto,expected_proto,'GLOBAL_PROTO',atol=1e-7,rtol=1e-6)
                _,zt,logits=infer(glob,tx);pred=np.argmax(unit(zt)@unit(global_proto).T,axis=1);ce=np.argmax(logits,axis=1)
                with np.load(rd/'test_predictions.npz') as a:np.testing.assert_array_equal(pred,a['prototype_pred']);np.testing.assert_array_equal(ce,a['ce_pred'])
                acc=float(np.mean(pred==ty));ceacc=float(np.mean(ce==ty));client_acc=[float(np.mean(pred[test_assign==k]==ty[test_assign==k])) for k in range(cfg['clients'])]
                assert acc==records[rnd]['prototype_accuracy'] and ceacc==records[rnd]['ce_accuracy'] and client_acc==records[rnd]['client_test_accuracy']
                state=call('tasks',[task],['bytes32','bytes32','bytes32','bytes32','uint32','uint32','uint32','bool'],tag)
                expected_state=['0x'+run['config_sha256'],'0x'+run['input_hash'],'0x'+filehash(rd/'global_model.npz'),'0x'+filehash(rd/'global_prototypes.npy'),rnd+1,cfg['clients'],0,True]
                assert state==expected_state and list(link['node_states'][0].values())==state and link['node_states'][0]==link['node_states'][1]
                assert commitments==link['node2_update_commitments']
                root='0x'+Web3.keccak(b''.join(bytes.fromhex(c[2:]) for c in commitments)).hex();assert root==link['submissions_root']
                fn,arguments=contract.decode_function_input(byhash[link['finalize_transaction']]['transaction']['input'])
                assert fn.fn_name=='finalize' and clean(arguments['id'])==run['task_id'] and arguments['round']==rnd and clean(arguments['root'])==root
                assert clean(arguments['model'])==expected_state[2] and clean(arguments['prototype'])==expected_state[3]
                append(out/'historical_state_checks.jsonl',{'run_id':dest.name,'round':rnd,'block':tag,'state':state,'submissions_root':root,'passed':True})
                curve.append({'run_id':dest.name,'seed':seed,'arm':arm['id'],'round':rnd,'prototype_accuracy':acc,'ce_accuracy':ceacc})
                initial=glob;prior=global_proto;prior_hash=filehash(rd/'global_prototypes.npy');model_hash=filehash(rd/'global_model.npz')
            original_small=read(dest/'small_sample_estimates.json');assert len(original_small)==len(small)
            for a,b in zip(original_small,small):
                assert (a['client'],a['class'],a['n'])==(b['client'],b['class'],b['n'])
                near(a['raw_mse'],b['raw_mse'],'RAW_MSE',atol=1e-7);near(a['shrunk_mse'],b['shrunk_mse'],'SHRUNK_MSE',atol=1e-7)
            summary.append({'run_id':dest.name,'seed':seed,'arm':arm['id'],'final_prototype_accuracy':acc,'final_ce_accuracy':ceacc,'mean_client_accuracy':float(np.mean(client_acc)),'small_cells':len(small),'small_raw_mse':float(np.mean([s['raw_mse'] for s in small])),'small_shrunk_mse':float(np.mean([s['shrunk_mse'] for s in small]))})
            print('AUDITED '+dest.name+' accuracy='+str(acc)+' replay_max='+str(max_replay_error),flush=True)
        write_csv(out/'per_run_statistics.csv',summary);write_csv(out/'round_curves.csv',curve)
        aggregate=[]
        for arm in sorted({r['arm'] for r in summary}):
            rows=[r for r in summary if r['arm']==arm]
            rec={'arm':arm,'independent_runs':len(rows)}
            for metric in ['final_prototype_accuracy','final_ce_accuracy','mean_client_accuracy','small_raw_mse','small_shrunk_mse']:
                vals=[r[metric] for r in rows];rec[metric+'_mean']=float(np.mean(vals));rec[metric+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else ''
            aggregate.append(rec)
        write_csv(out/'aggregate_statistics.csv',aggregate)
        paired=[]
        for plus,minus in [('SC','F'),('W','F'),('W','SC'),('S','F'),('WS','W'),('WS','S')]:
            ds=[]
            for seed in cfg['seeds']:
                a=next((r for r in summary if r['seed']==seed and r['arm']==plus),None);b=next((r for r in summary if r['seed']==seed and r['arm']==minus),None)
                if a and b:ds.append(100*(a['final_prototype_accuracy']-b['final_prototype_accuracy']))
            if ds:paired.append({'comparison':plus+'-'+minus,'n':len(ds),'paired_delta_percentage_points':json.dumps(ds),'mean_delta_pp':float(np.mean(ds)),'sample_sd_pp':float(np.std(ds,ddof=1)) if len(ds)>1 else ''})
        if paired:write_csv(out/'paired_comparisons.csv',paired)
        result={'utc':now(),'status':'passed','config_sha256':start['config_sha256'],'audited_runs':len(summary),'audited_rounds':len(curve),'replayed_training_client_updates':replay_count,'max_abs_replayed_parameter_error':max_replay_error,'verified_signed_transactions_and_archived_receipts':chain_checks,'successful_transactions':sum(r['expected_status']==1 for r in receipts),'expected_reverted_transactions':sum(r['expected_status']==0 for r in receipts),'real_archived_node_client_version':version,'input_source_hashes_and_selected_pixels_checked':True,'prediction_and_statistics_recomputed':True,'blockchain_historical_state_reopened_and_queried':True,'scientific_interpretation':'execution_and_numeric_agreement_do_not_imply_superiority'}
        save(out/'audit_result.json',result);print(json.dumps(result),flush=True)
    except BaseException as e:
        import traceback
        save(out/'audit_failure.json',{'utc':now(),'error':str(e),'traceback':traceback.format_exc()});raise
    finally:
        if proc.poll() is None:proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close()

if __name__=='__main__':main()
