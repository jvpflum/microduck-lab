"""Strict local ONNX MLP inspection and NumPy replay; no pickle or executable model loading.

This deliberately supports only the exported Sub/Div/Gemm/Elu family. It is not
an ONNX runtime replacement or proof of cross-simulator behavioral equivalence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

def require(value,message):
    if not value:raise ValueError(message)
def varint(data,start):
    value=0
    for shift in range(0,70,7):
        require(start<len(data),'Truncated varint');byte=data[start];start+=1;value|=(byte&127)<<shift
        if not byte&128:return value,start
    raise ValueError('Overlong varint')
def fields(data):
    result={};position=0
    while position<len(data):
        key,position=varint(data,position);tag,wire=key>>3,key&7
        require(tag>0,'Invalid field number')
        if wire==0:value,position=varint(data,position)
        elif wire in (1,5):
            size=8 if wire==1 else 4;require(position+size<=len(data),'Truncated fixed field');value=data[position:position+size];position+=size
        elif wire==2:
            size,position=varint(data,position);require(position+size<=len(data),'Truncated bytes field');value=data[position:position+size];position+=size
        else:raise ValueError('Unsupported protobuf wire type')
        result.setdefault(tag,[]).append((wire,value))
    return result
def one(message,tag,default=None):
    values=message.get(tag,[]);require(len(values)<=1,'Duplicate singleton field')
    return values[0][1] if values else default
def text(value):return value.decode('utf-8')
def ints(message,tag):
    values=[]
    for wire,value in message.get(tag,[]):
        if wire==0:values.append(value)
        else:
            require(wire==2,'Invalid packed integer');i=0
            while i<len(value):v,i=varint(value,i);values.append(v)
    return values
def tensor(blob):
    m=fields(blob);shape=ints(m,1);name=text(one(m,8,b''))
    require(name and one(m,2)==1,'Only named float32 tensors supported')
    require(not m.get(13) and one(m,14,0)==0,'External tensor data prohibited')
    require(all(0<x<=1000000 for x in shape),'Invalid tensor dimensions')
    raw=one(m,9)
    if raw is None:
        raw=b''.join(value for wire,value in m.get(4,[]) if wire in (2,5))
    value=np.frombuffer(raw,dtype='<f4').copy();require(value.size==int(np.prod(shape)),'Tensor byte shape mismatch')
    value=value.reshape(shape);require(np.isfinite(value).all(),'Nonfinite model tensor')
    return name,value
def value_info(blob):
    m=fields(blob);typ=fields(one(m,2,b''));t=fields(one(typ,1,b''));shape=fields(one(t,2,b''))
    dims=[]
    for _,blob in shape.get(1,[]):
        d=fields(blob);dims.append(one(d,1,None) if 1 in d else text(one(d,2,b'')))
    return {'name':text(one(m,1,b'')),'dtype':one(t,1),'shape':dims}
def attr(blob):
    m=fields(blob);name=text(one(m,1,b''));typ=one(m,20)
    if typ==1:return name,struct.unpack('<f',one(m,2))[0]
    if typ==2:return name,one(m,3)
    raise ValueError('Unsupported node attribute '+name)

class MLP:
    def __init__(self,path,expected_sha256):
        raw=Path(path).read_bytes();self.sha256=hashlib.sha256(raw).hexdigest()
        require(self.sha256==expected_sha256,'ONNX byte identity differs')
        model=fields(raw);graph=fields(one(model,7,b''));self.parameters={}
        for _,blob in graph.get(5,[]):
            name,value=tensor(blob);require(name not in self.parameters,'Duplicate initializer');self.parameters[name]=value
        self.inputs=[value_info(blob)for _,blob in graph.get(11,[])];self.outputs=[value_info(blob)for _,blob in graph.get(12,[])]
        require(len(self.inputs)==len(self.outputs)==1,'Expected one input/output')
        require(self.inputs[0]['dtype']==self.outputs[0]['dtype']==1,'Expected float32 IO')
        require(self.inputs[0]['shape']==[1,61] and self.outputs[0]['shape']==[1,14],'Expected exact source 1x61→1x14')
        self.metadata={}
        for _,blob in model.get(14,[]):
            m=fields(blob);key=text(one(m,1));require(key not in self.metadata,'Duplicate metadata');self.metadata[key]=text(one(m,2))
        self.nodes=[];defined=set(self.parameters)|{self.inputs[0]['name']}
        for _,blob in graph.get(1,[]):
            m=fields(blob);node={'op':text(one(m,4,b'')),'inputs':[text(v)for _,v in m.get(1,[])],
                'outputs':[text(v)for _,v in m.get(2,[])],'attributes':dict(attr(v)for _,v in m.get(5,[]))}
            require(text(one(m,7,b'')) in ('','ai.onnx'),'Custom operator domain prohibited')
            require(node['op'] in ('Sub','Div','Gemm','Elu'),'Unsupported executable operator '+node['op'])
            require(set(node['inputs'])<=defined and len(node['outputs'])==1 and node['outputs'][0] not in defined,'Invalid graph topology')
            allow={'Sub':set(),'Div':set(),'Elu':{'alpha'},'Gemm':{'alpha','beta','transA','transB'}}[node['op']]
            require(set(node['attributes'])<=allow,'Unsupported operator attribute')
            defined.update(node['outputs']);self.nodes.append(node)
        require(self.outputs[0]['name'] in defined,'Missing graph output')
        require(Counter(n['op']for n in self.nodes)=={'Sub':1,'Div':1,'Gemm':4,'Elu':3},'Expected exact normalized three-hidden-layer MLP')
        sub,div=self.nodes[:2];require(sub['op']=='Sub' and div['op']=='Div','Normalizer must precede network')
        require(sub['inputs'][0]==self.inputs[0]['name'] and sub['inputs'][1] in self.parameters,'Normalizer mean binding differs')
        require(div['inputs'][0]==sub['outputs'][0] and div['inputs'][1] in self.parameters,'Normalizer divisor binding differs')
        self.mean=self.parameters[sub['inputs'][1]];self.divisor=self.parameters[div['inputs'][1]]
        require(self.mean.shape==self.divisor.shape==(1,61) and (self.divisor>0).all(),'Invalid embedded normalizer')
        self.report={'sha256':self.sha256,'bytes':len(raw),'inputs':self.inputs,'outputs':self.outputs,'metadata':self.metadata,
            'operators':dict(Counter(n['op']for n in self.nodes)),'normalization':'exact embedded (raw - mean) / exported_divisor; no additional epsilon or normalization',
            'mean_sha256':hashlib.sha256(self.mean.tobytes()).hexdigest(),'divisor_sha256':hashlib.sha256(self.divisor.tobytes()).hexdigest(),
            'mean':self.mean.tolist(),'divisor':self.divisor.tolist(),'tensors':{k:{'shape':list(v.shape),'raw_sha256':hashlib.sha256(v.tobytes()).hexdigest()}for k,v in self.parameters.items()},
            'native_runtime_parity_verified':False,'physical_transfer_verified':False}
    def __call__(self,raw_observation):
        raw=np.asarray(raw_observation,dtype=np.float32);require(raw.shape==(1,61) and np.isfinite(raw).all(),'Exact finite raw source observation required')
        values={**self.parameters,self.inputs[0]['name']:raw}
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for node in self.nodes:
                x=[values[k]for k in node['inputs']];p=node['attributes'];op=node['op']
                if op=='Sub':out=x[0]-x[1]
                elif op=='Div':out=x[0]/x[1]
                elif op=='Elu':
                    out=x[0].copy();mask=out<0;out[mask]=np.float32(p.get('alpha',1))*(np.exp(out[mask])-np.float32(1))
                else:
                    require(len(x)==3,'Gemm requires bias');left=x[0].T if p.get('transA',0) else x[0];right=x[1].T if p.get('transB',0) else x[1]
                    out=np.float32(p.get('alpha',1))*(left@right)+np.float32(p.get('beta',1))*x[2]
                require(np.isfinite(out).all(),'Nonfinite graph output');values[node['outputs'][0]]=out
        out=values[self.outputs[0]['name']];require(out.shape==(1,14),'Output dimensions differ');return out

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();model=MLP(args.model,args.sha256);report=model.report
    report['zero_input_output']=model(np.zeros((1,61),dtype=np.float32)).tolist()
    with args.output.open('x')as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'sha256':model.sha256,'inspected':True,'runtime_qualified':False}))
if __name__=='__main__':main()
