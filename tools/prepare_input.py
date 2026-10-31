import csv
import importlib.metadata
import struct
import subprocess
import argparse
import numpy as np
from common import *

def idx(path):
    with path.open('rb') as f:
        magic,n=struct.unpack('>II',f.read(8))
        if magic==2051:
            rows,cols=struct.unpack('>II',f.read(8))
            assert (rows,cols)==(28,28)
            assert path.stat().st_size==16+n*rows*cols
            return np.memmap(path,dtype=np.uint8,mode='r',offset=16,shape=(n,rows,cols))
        assert magic==2049 and path.stat().st_size==8+n
        return np.memmap(path,dtype=np.uint8,mode='r',offset=8,shape=(n,))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing uncompressed EMNIST Balanced IDX files')
    source=parser.parse_args().data_dir.resolve()
    required=[source/f'emnist-balanced-{split}-{kind}' for split in ['train','test'] for kind in ['images-idx3-ubyte','labels-idx1-ubyte']] + [source/'emnist-balanced-mapping.txt']
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    inp=BASE/'I23-INPUT'; inp.mkdir(exist_ok=False)
    config={'frozen_utc':now(),'implementation':'independent_numpy_mechanism_supplement_v1','data_source':source.as_posix(),'classes':47,'train_per_class':100,'test_per_class':40,'sample_seed':2300,'seeds':[2301,2302,2303],'clients':6,'dirichlet_alpha':.2,'rounds':10,'local_epochs':1,'batch_size':32,'hidden':64,'embedding':32,'optimizer':'SGD_no_momentum','learning_rate':.03,'weight_decay':1e-5,'tau':.15,'alpha':.5,'lambda_wpc':.7,'shrink_mu':10.,'ird_enabled':False,'mhn_enabled':False,'arms':[{'id':'F','wpc':0.,'alpha':.5,'mu':0.},{'id':'SC','wpc':.7,'alpha':0.,'mu':0.},{'id':'W','wpc':.7,'alpha':.5,'mu':0.},{'id':'S','wpc':0.,'alpha':.5,'mu':10.},{'id':'WS','wpc':.7,'alpha':.5,'mu':10.}], 'preprocessing':'IDX raw 28x28 orientation kept, float32 /255, flatten; no augmentation or fitted transform','model':'784->64 tanh->32 linear embedding->47 linear CE head','primary_metric':'final_round_global_cosine_prototype_accuracy_on_fixed_official_test_subset','secondary_metrics':['final_CE_head_accuracy','equal_weight_mean_client_test_accuracy','final_small_n_1_to_5_prototype_MSE_to_same_local_model_test_class_centroid'],'repeat_unit':'three independent initialization, partition and minibatch order seeds; same fixed sample subset; arms paired within seed','estimator_readout':'cosine nearest global prototype, common to all arms, supplementary implementation choice beyond paper Algorithm 1 output wT','round0':'all clients submit initial-model raw prototypes; count-weighted prior; no shrink before a prior exists','chain':'local two-node Geth Clique, six simulated client accounts, off-chain numeric validation and aggregation gated by on-chain submissions and state','no_score_threshold':True,'ports':[19623,19624],'chain_id':23230910}
    save(inp/'config.json',config)
    rng=np.random.default_rng(config['sample_seed']); rows=[]; arrays={}; source_manifest=[]
    for p in sorted(source.iterdir()):
        if p.is_file(): source_manifest.append({'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':filehash(p)})
    for split,nper in [('train',100),('test',40)]:
        x=idx(source/f'emnist-balanced-{split}-images-idx3-ubyte'); y=idx(source/f'emnist-balanced-{split}-labels-idx1-ubyte')
        assert len(x)==len(y) and set(np.unique(y))==set(range(47))
        selected=np.concatenate([rng.choice(np.flatnonzero(y==c),nper,replace=False) for c in range(47)])
        arrays[split+'_x']=np.array(x[selected]); arrays[split+'_y']=np.array(y[selected]); arrays[split+'_source_index']=selected
        for j,i in enumerate(selected): rows.append({'split':split,'subset_index':j,'source_index':int(i),'label':int(y[i]),'image_sha256':digest(x[i].tobytes())})
    np.savez_compressed(inp/'emnist_subset.npz',**arrays)
    with (inp/'sample_index.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for seed in config['seeds']:
        rng=np.random.default_rng(seed); train_client=np.full(len(arrays['train_y']),-1,dtype=np.int16); test_client=np.full(len(arrays['test_y']),-1,dtype=np.int16)
        proportions=[]
        for c in range(47):
            probs=rng.dirichlet(np.full(6,.2));proportions.append(probs.tolist())
            for split,dest in [('train',train_client),('test',test_client)]:
                ix=rng.permutation(np.flatnonzero(arrays[split+'_y']==c)); counts=rng.multinomial(len(ix),probs)
                for k,chunk in enumerate(np.split(ix,np.cumsum(counts)[:-1])):dest[chunk]=k
        assert np.all(train_client>=0) and np.all(test_client>=0)
        assert min(np.bincount(train_client,minlength=6))>0
        np.savez_compressed(inp/f'partition_{seed}.npz',train_client=train_client,test_client=test_client)
        counts=np.array([np.bincount(arrays['train_y'][train_client==k],minlength=47) for k in range(6)])
        save(inp/f'partition_{seed}.json',{'seed':seed,'proportions':proportions,'train_counts_by_client_class':counts.tolist(),'train_client_sizes':np.bincount(train_client,minlength=6).tolist(),'test_client_sizes':np.bincount(test_client,minlength=6).tolist(),'small_class_cells_1_to_5':int(((counts>0)&(counts<=5)).sum())})
    train_hash={r['image_sha256'] for r in rows if r['split']=='train'}
    overlap=[r for r in rows if r['split']=='test' and r['image_sha256'] in train_hash]
    save(inp/'source_manifest.json',source_manifest)
    save(inp/'input_checks.json',{'utc':now(),'official_train':112800,'official_test':18800,'selected_train':4700,'selected_test':1880,'class_count':47,'source_indices_unique_within_split':True,'identical_pixel_train_test_overlap':overlap,'subset_sha256':filehash(inp/'emnist_subset.npz'),'config_sha256':filehash(inp/'config.json'),'mapping':(source/'emnist-balanced-mapping.txt').read_text(encoding='utf-8'),'external_reference':'https://www.nist.gov/itl/products-and-services/emnist-dataset'})
    print(json.dumps({'frozen_config_sha256':filehash(inp/'config.json'),'train':4700,'test':1880,'seeds':config['seeds'],'overlap_count':len(overlap)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
