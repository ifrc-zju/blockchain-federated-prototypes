from runtime import executable
"""New I23 transport adapted after reading I21/tools/chain.py (hash in environment).

Uses existing Geth and solc read-only; creates fresh loopback-only chain stores.
"""
import subprocess
import time
import urllib.request
from common import *
from eth_account import Account
from web3 import Web3

ZERO='0x'+'00'*32

class RPC:
    def __init__(self,url,log):self.url=url;self.log=log
    def call(self,method,params=None):
        rec={'utc':now(),'method':method,'params':params or []}
        try:
            req=urllib.request.Request(self.url,json.dumps({'jsonrpc':'2.0','id':1,'method':method,'params':params or []}).encode('utf-8'),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=15) as f:reply=json.load(f)
            rec['response']=reply
            if 'error' in reply:raise RuntimeError(reply['error'])
            return reply['result']
        except Exception as exc:
            rec['error']=repr(exc);raise
        finally:append(self.log,rec)

class Chain:
    def __init__(self,out,ports,chain_id,nclients):
        self.out=out; self.dir=out/'chain';self.dir.mkdir()
        self.ports=ports;self.chain_id=chain_id;self.processes=[];self.nonces={};self.raws=[]
        self.accounts=[Account.create() for _ in range(nclients+2)]
        self.nodes=[Account.create() for _ in range(2)]
        self.owner=self.accounts[0];self.clients=self.accounts[1:nclients+1];self.outsider=self.accounts[-1]
        save(self.dir/'TEST_ONLY_accounts.json',[{'address':a.address,'key':a.key.hex()} for a in self.accounts+self.nodes])
        config={k:0 for k in ['homesteadBlock','eip150Block','eip155Block','eip158Block','byzantiumBlock','constantinopleBlock','petersburgBlock','istanbulBlock','muirGlacierBlock','berlinBlock','londonBlock']}
        config.update(chainId=chain_id,clique={'period':1,'epoch':30000})
        genesis={'config':config,'nonce':'0x0','timestamp':hex(int(time.time())-10),'extraData':'0x'+'00'*32+''.join(sorted(a.address[2:].lower() for a in self.nodes))+'00'*65,'gasLimit':'0x5f5e100','difficulty':'0x1','mixHash':ZERO,'coinbase':'0x'+'00'*20,'alloc':{a.address[2:]:{'balance':str(10**25)} for a in self.accounts+self.nodes},'baseFeePerGas':'0x1'}
        save(self.dir/'genesis.json',genesis)
        self.rpcs=[RPC(f'http://127.0.0.1:{p}',self.dir/f'node{i+1}_rpc.jsonl') for i,p in enumerate(ports)]

    def start(self):
        commands=[]
        for i,a in enumerate(self.nodes):
            d=self.dir/f'node{i+1}';d.mkdir()
            (d/'password.txt').write_text('I23-local-test-only',encoding='utf-8')
            (d/'signer.key').write_text(a.key.hex(),encoding='utf-8')
            cmd=[executable('geth'),'--datadir',str(d)]
            with (self.dir/f'node{i+1}_init.log').open('xb') as f:
                for tail in [['init',str(self.dir/'genesis.json')],['account','import','--password',str(d/'password.txt'),str(d/'signer.key')]]:
                    subprocess.run(cmd+tail,stdout=f,stderr=f,check=True,creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            cmd+=['--networkid',str(self.chain_id),'--syncmode','full','--gcmode','archive','--state.scheme','hash','--cache','64','--maxpeers','2','--port',str(self.ports[i]+10000),'--nodiscover','--nat','none','--netrestrict','127.0.0.0/8','--http','--http.addr','127.0.0.1','--http.port',str(self.ports[i]),'--http.api','eth,net,web3,admin,clique','--http.vhosts','localhost,127.0.0.1','--ipcdisable','--authrpc.addr','127.0.0.1','--authrpc.port',str(self.ports[i]+20000),'--mine','--miner.etherbase',a.address,'--miner.gasprice','1','--unlock',a.address,'--password',str(d/'password.txt'),'--allow-insecure-unlock','--verbosity','3']
            f=(self.dir/f'node{i+1}.log').open('xb');p=subprocess.Popen(cmd,stdout=f,stderr=f,creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.processes.append((p,f));commands.append({'pid':p.pid,'command':cmd})
        save(self.dir/'commands.json',commands)
        for rpc in self.rpcs:
            deadline=time.monotonic()+35
            while True:
                try:rpc.call('web3_clientVersion');break
                except Exception:
                    if time.monotonic()>deadline:raise
                    time.sleep(.2)
        infos=[r.call('admin_nodeInfo') for r in self.rpcs]
        for i in range(2):
            enode=infos[1-i]['enode'].split('@')[0]+f'@127.0.0.1:{self.ports[1-i]+10000}'
            self.rpcs[i].call('admin_addPeer',[enode])
        deadline=time.monotonic()+30
        while any(int(r.call('net_peerCount'),16)<1 for r in self.rpcs):
            if time.monotonic()>deadline:raise RuntimeError('PEER_TIMEOUT')
            time.sleep(.2)
        save(self.dir/'network_start.json',{'utc':now(),'physical_hosts':1,'nodes':2,'client_accounts':clean([a.address for a in self.clients]),'node_infos':infos,'peers':[r.call('admin_peers') for r in self.rpcs],'genesis_hashes':[r.call('eth_getBlockByNumber',['0x0',False])['hash'] for r in self.rpcs]})

    def submit(self,account,data,label,to=None):
        nonce=self.nonces.get(account.address,0)
        tx={'chainId':self.chain_id,'nonce':nonce,'gas':7000000,'gasPrice':2000000000,'value':0,'data':data}
        if to:tx['to']=Web3.to_checksum_address(to)
        signed=Account.sign_transaction(tx,account.key);raw='0x'+signed.raw_transaction.hex()
        h=self.rpcs[0].call('eth_sendRawTransaction',[raw]);self.nonces[account.address]=nonce+1
        rec={'utc':now(),'label':label,'hash':h,'from':account.address,'raw':raw,'tx_input':clean(tx)}
        append(self.dir/'submitted.jsonl',rec);self.raws.append(raw);return rec

    def settle(self,items,expected=1):
        receipts=[]
        for item in items:
            deadline=time.monotonic()+45
            while True:
                rc=self.rpcs[0].call('eth_getTransactionReceipt',[item['hash']])
                if rc:break
                if time.monotonic()>deadline:raise RuntimeError('TX_TIMEOUT')
                time.sleep(.08)
            block=self.rpcs[0].call('eth_getBlockByHash',[rc['blockHash'],False])
            transaction=self.rpcs[0].call('eth_getTransactionByHash',[item['hash']])
            rec={**item,'observed_utc':now(),'expected_status':expected,'receipt':rc,'block':block,'transaction':transaction}
            append(self.dir/'receipts.jsonl',rec)
            assert int(rc['status'],16)==expected,(item['label'],rc)
            receipts.append(rec)
        return receipts

    def sync(self,block_number):
        a=self.rpcs[0].call('eth_getBlockByNumber',[block_number,False]);deadline=time.monotonic()+30
        while True:
            b=self.rpcs[1].call('eth_getBlockByNumber',[block_number,False])
            if b and b['hash']==a['hash']:break
            if time.monotonic()>deadline:raise RuntimeError('STATE_SYNC_TIMEOUT')
            time.sleep(.1)
        append(self.dir/'synchronization.jsonl',{'utc':now(),'block_number':block_number,'node1_hash':a['hash'],'node2_hash':b['hash'],'node1_state_root':a['stateRoot'],'node2_state_root':b['stateRoot']})

    def deploy(self):
        cmd=[executable('solc'),'--combined-json','abi,bin,bin-runtime','--optimize','--evm-version','paris',str(BASE/'tools/PrototypeRegistry.sol')]
        p=subprocess.run(cmd,capture_output=True,check=True,creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        compiled=json.loads(p.stdout);entry=next(iter(compiled['contracts'].values()))
        save(self.dir/'compiled_contract.json',compiled)
        self.abi=entry['abi'];self.contract=Web3().eth.contract(abi=self.abi)
        rc=self.settle([self.submit(self.owner,'0x'+entry['bin'],'deploy')])[0]
        self.address=rc['receipt']['contractAddress'];tag=rc['receipt']['blockNumber'];self.sync(tag)
        codes=[r.call('eth_getCode',[self.address,tag]) for r in self.rpcs]
        assert codes[0]==codes[1]
        assert len(codes[0])>2
        save(self.dir/'deployment.json',{'utc':now(),'address':self.address,'transaction_hash':rc['hash'],'runtime_code':codes,'source_sha256':filehash(BASE/'tools/PrototypeRegistry.sol'),'compiler_command':cmd,'abi':self.abi})

    def send(self,account,method,args,label):
        return self.submit(account,self.contract.encode_abi(method,args=args),label,self.address)

    def call(self,node,method,args,types,tag='latest'):
        data=self.contract.encode_abi(method,args=args)
        raw=self.rpcs[node].call('eth_call',[{'to':self.address,'data':data},tag])
        return clean(Web3().codec.decode(types,bytes.fromhex(raw[2:])))

    def state(self,task,tag='latest',node=0):
        v=self.call(node,'tasks',[task],['bytes32','bytes32','bytes32','bytes32','uint32','uint32','uint32','bool'],tag)
        return dict(zip(['config_hash','input_hash','model_hash','prototype_hash','next_round','client_count','submitted','exists'],v))

    def stop(self):
        for p,f in self.processes:
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            f.close()
        save(self.dir/'stopped.json',{'utc':now(),'process_exit_codes':[p.returncode for p,f in self.processes]})
