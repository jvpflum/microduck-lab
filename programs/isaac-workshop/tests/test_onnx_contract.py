"""Small synthetic protobuf graphs, never robot-policy performance fixtures."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from isaac_workshop import onnx_contract as graph


def vi(n):
    out=bytearray()
    while n>127:out.append((n&127)|128);n>>=7
    out.append(n);return bytes(out)


def field(tag,value):
    return vi(tag*8)+vi(value) if isinstance(value,int) else vi(tag*8+2)+vi(len(value))+value


def tensor(name,value,external=False):
    value=np.asarray(value,dtype='<f4')
    return b''.join(field(1,int(n))for n in value.shape)+field(2,1)+field(8,name.encode())+field(9,value.tobytes())+(field(14,1)if external else b'')


def info(name,width):
    shape=field(1,field(1,1))+field(1,field(1,width))
    return field(1,name.encode())+field(2,field(1,field(1,1)+field(2,shape)))


def node(op,inputs,output):
    return b''.join(field(1,s.encode())for s in inputs)+field(2,output.encode())+field(4,op.encode())


def fixture(*,external=False,op='Elu'):
    rng=np.random.default_rng(13)
    params={'mean':np.full((1,61),.05,np.float32),'divisor':np.full((1,61),2.,np.float32)}
    nodes=[node('Sub',['input','mean'],'centered'),node('Div',['centered','divisor'],'norm')]
    previous='norm'
    for i,(a,b)in enumerate(((61,3),(3,3),(3,3),(3,14))):
        params[f'w{i}']=rng.normal(0,.1,(a,b)).astype(np.float32)
        params[f'b{i}']=rng.normal(0,.1,(b,)).astype(np.float32)
        output='output' if i==3 else f'g{i}'
        nodes.append(node('Gemm',[previous,f'w{i}',f'b{i}'],output));previous=output
        if i<3:nodes.append(node(op,[previous],f'e{i}'));previous=f'e{i}'
    g=b''.join(field(1,n)for n in nodes)+field(2,b'synthetic')
    g+=b''.join(field(5,tensor(n,v,external and n=='mean'))for n,v in params.items())
    g+=field(11,info('input',61))+field(12,info('output',14))
    return field(1,9)+field(7,g)+field(8,field(2,13)),params


class OnnxContractTests(unittest.TestCase):
    def load(self,raw):
        p=Path(self.temp.name)/'synthetic.onnx';p.write_bytes(raw)
        return graph.MLP(p,hashlib.sha256(raw).hexdigest())

    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def test_normalizer_and_mlp_match_independent_array_formula(self):
        raw,p=fixture();model=self.load(raw);x=np.linspace(-1,1,61,dtype=np.float32)[None,:]
        expected=(x-p['mean'])/p['divisor']
        for i in range(4):
            expected=expected@p[f'w{i}']+p[f'b{i}']
            if i<3:expected=np.where(expected>=0,expected,np.expm1(expected))
        np.testing.assert_allclose(model(x),expected,atol=1e-7,rtol=1e-6)
        np.testing.assert_array_equal(model(x),model(x))
        self.assertFalse(model.report['native_runtime_parity_verified'])

    def test_hash_mismatch_precedes_parsing(self):
        p=Path(self.temp.name)/'bad';p.write_bytes(b'not a model')
        with self.assertRaisesRegex(ValueError,'identity'):graph.MLP(p,'0'*64)

    def test_external_data_and_unsupported_ops_reject(self):
        for kwargs in ({'external':True},{'op':'Relu'}):
            with self.assertRaises(ValueError):self.load(fixture(**kwargs)[0])

    def test_nonfinite_wrong_width_and_batch_reject(self):
        model=self.load(fixture()[0])
        for x in (np.zeros((2,61)),np.zeros((1,60)),np.full((1,61),np.nan)):
            with self.assertRaises(ValueError):model(x)

    def test_truncated_and_unknown_wire_reject(self):
        for raw in (b'\x0a\x80',b'\x0a\x04ab',b'\x0f',b'\x00'):
            with self.assertRaises(ValueError):graph.fields(raw)


if __name__=='__main__':unittest.main()
