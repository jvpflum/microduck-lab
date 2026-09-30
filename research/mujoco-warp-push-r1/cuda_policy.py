"""Audited operator subset for the frozen V89 graph; all arithmetic on CUDA."""
import onnx,torch
from onnx import numpy_helper,helper
class Policy(torch.nn.Module):
    def __init__(self,path,device):
        super().__init__();m=onnx.load(path);self.input=m.graph.input[0].name;self.output=m.graph.output[0].name
        self.keys={};self.nodes=[]
        def add(name,a):
            key='tensor_'+str(len(self.keys));self.register_buffer(key,torch.from_numpy(a.copy()).to(device));self.keys[name]=key
        for v in m.graph.initializer:add(v.name,numpy_helper.to_array(v))
        for n in m.graph.node:
            attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
            if n.op_type=='Constant':add(n.output[0],numpy_helper.to_array(attrs['value']))
            else:self.nodes.append((n.op_type,list(n.input),n.output[0],attrs))
    def forward(self,x):
        v={name:getattr(self,key) for name,key in self.keys.items()};v[self.input]=x
        ints=lambda a:[int(z) for z in a.detach().cpu().reshape(-1).tolist()]
        for op,names,out,a in self.nodes:
            z=[v[n] if n else None for n in names];p=z[0]
            if op=='Identity':y=p
            elif op=='Add':y=p+z[1]
            elif op=='Sub':y=p-z[1]
            elif op=='Mul':y=p*z[1]
            elif op=='Div':y=p/z[1]
            elif op=='Abs':y=p.abs()
            elif op=='Tanh':y=p.tanh()
            elif op=='Elu':y=torch.nn.functional.elu(p,alpha=a.get('alpha',1.))
            elif op=='Gemm':
                aa=p.T if a.get('transA',0) else p;bb=z[1].T if a.get('transB',0) else z[1]
                y=a.get('alpha',1.)*(aa@bb)+a.get('beta',1.)*z[2]
            elif op=='MatMul':y=p@z[1]
            elif op=='Clip':y=torch.clamp(p,min=float(z[1]) if len(z)>1 and z[1] is not None else a.get('min'),max=float(z[2]) if len(z)>2 and z[2] is not None else a.get('max'))
            elif op=='Greater':y=p>z[1]
            elif op=='Less':y=p<z[1]
            elif op=='Equal':y=p==z[1]
            elif op=='And':y=p&z[1]
            elif op=='Where':y=torch.where(p,z[1],z[2])
            elif op=='Cast':y=p.to({1:torch.float32,6:torch.int32,7:torch.int64,9:torch.bool,11:torch.float64}[a['to']])
            elif op=='Concat':y=torch.cat(z,dim=a['axis'])
            elif op=='Unsqueeze':
                y=p
                for dim in sorted(ints(z[1]) if len(z)>1 else a['axes']):y=y.unsqueeze(dim)
            elif op=='Gather':
                axis=a.get('axis',0);axis=axis if axis>=0 else p.ndim+axis;idx=z[1].long();idx=torch.where(idx<0,idx+p.shape[axis],idx)
                y=torch.index_select(p,axis,idx.reshape(-1)).reshape(*p.shape[:axis],*idx.shape,*p.shape[axis+1:])
            elif op=='Slice':
                starts=ints(z[1]);ends=ints(z[2]);axes=ints(z[3]) if len(z)>3 else list(range(len(starts)));steps=ints(z[4]) if len(z)>4 else [1]*len(starts);sl=[slice(None)]*p.ndim
                for axis,start,end,step in zip(axes,starts,ends,steps):sl[axis]=slice(start,end,step)
                y=p[tuple(sl)]
            elif op.startswith('Reduce'):
                axes=tuple(ints(z[1]) if len(z)>1 else a.get('axes',range(p.ndim)));keep=bool(a.get('keepdims',1))
                if op=='ReduceMean':y=p.mean(dim=axes,keepdim=keep)
                elif op=='ReduceMax':y=torch.amax(p,dim=axes,keepdim=keep)
                elif op=='ReduceSum':y=p.sum(dim=axes,keepdim=keep)
                else:raise ValueError(op)
            elif op=='CumSum':
                axis=int(z[1]);u=p.flip((axis,)) if a.get('reverse',0) else p;y=u.cumsum(axis)
                if a.get('exclusive',0):y=y-u
                if a.get('reverse',0):y=y.flip((axis,))
            elif op=='Reshape':
                shape=ints(z[1]);shape=[p.shape[i] if s==0 else s for i,s in enumerate(shape)];y=p.reshape(shape)
            elif op=='Flatten':y=p.flatten(start_dim=a.get('axis',1))
            else:raise ValueError('Unsupported ONNX op '+op)
            v[out]=y
        return v[self.output]
