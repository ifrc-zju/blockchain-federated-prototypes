// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Local research component: transactions gate off-chain numeric aggregation.
/// Commitments prove linkage; this contract does not prove truthful training.
contract PrototypeRegistry {
    address public immutable owner;
    struct Task {
        bytes32 configHash;
        bytes32 inputHash;
        bytes32 modelHash;
        bytes32 prototypeHash;
        uint32 round;
        uint32 clientCount;
        uint32 submitted;
        bool exists;
    }
    mapping(bytes32 => Task) public tasks;
    mapping(bytes32 => address[]) private participants;
    mapping(bytes32 => mapping(address => bool)) public allowed;
    mapping(bytes32 => mapping(uint32 => mapping(address => bytes32))) public updates;
    event Created(bytes32 indexed task, bytes32 configHash, bytes32 inputHash);
    event Submitted(bytes32 indexed task, uint32 indexed round, address indexed client, bytes32 prior, bytes32 commitment);
    event Finalized(bytes32 indexed task, uint32 indexed round, bytes32 model, bytes32 prototype, bytes32 submissionsRoot);

    constructor() { owner=msg.sender; }
    function create(bytes32 id, bytes32 cfg, bytes32 input, bytes32 model, address[] calldata clients) external {
        require(msg.sender==owner,"OWNER");
        require(!tasks[id].exists && clients.length>0,"TASK");
        tasks[id]=Task(cfg,input,model,bytes32(0),0,uint32(clients.length),0,true);
        for(uint i=0;i<clients.length;i++) {
            require(clients[i]!=address(0) && !allowed[id][clients[i]],"CLIENT");
            allowed[id][clients[i]]=true;
            participants[id].push(clients[i]);
        }
        emit Created(id,cfg,input);
    }
    function submit(bytes32 id,uint32 round,bytes32 prior,bytes32 commitment) external {
        Task storage t=tasks[id];
        require(t.exists && allowed[id][msg.sender],"CLIENT");
        require(round==t.round,"ROUND");
        require(prior==t.prototypeHash,"PRIOR");
        require(updates[id][round][msg.sender]==bytes32(0),"DUPLICATE");
        require(commitment!=bytes32(0),"EMPTY");
        updates[id][round][msg.sender]=commitment;
        t.submitted++;
        emit Submitted(id,round,msg.sender,prior,commitment);
    }
    function submissionsRoot(bytes32 id,uint32 round) public view returns(bytes32) {
        bytes memory packed;
        for(uint i=0;i<participants[id].length;i++)
            packed=bytes.concat(packed,updates[id][round][participants[id][i]]);
        return keccak256(packed);
    }
    function finalize(bytes32 id,uint32 round,bytes32 model,bytes32 prototype,bytes32 root) external {
        require(msg.sender==owner,"OWNER");
        Task storage t=tasks[id];
        require(t.exists && round==t.round,"ROUND");
        require(t.submitted==t.clientCount,"INCOMPLETE");
        require(model!=bytes32(0) && prototype!=bytes32(0),"EMPTY");
        require(root==submissionsRoot(id,round),"ROOT");
        t.modelHash=model;
        t.prototypeHash=prototype;
        t.round++;
        t.submitted=0;
        emit Finalized(id,round,model,prototype,root);
    }
}
