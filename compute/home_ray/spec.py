"""Validate specifications before accepting any part of a batch."""
import math
import re

CAPABILITIES = {'arch:x86_64', 'arch:arm64', 'gpu:rtx5090', 'gpu:gb10', 'cuda', 'high_memory'}

def validate(raw):
    allowed = {'command', 'cpus', 'gpus', 'capabilities', 'memory_gb', 'timeout_s',
               'system_retries', 'idempotent', 'env', 'files', 'artifacts', 'name', 'retry_of'}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError('Unknown fields: ' + ', '.join(sorted(unknown)))
    s = dict(command=raw.get('command'), cpus=1, gpus=0, capabilities=[], memory_gb=1,
             timeout_s=3600, system_retries=0, idempotent=False, env={}, files={}, artifacts=[], name='job')
    s.update(raw)
    if not isinstance(s['command'], list) or not s['command'] or not all(isinstance(x, str) and '\0' not in x for x in s['command']):
        raise ValueError('command must be a nonempty argv list')
    for field in ('cpus', 'memory_gb', 'timeout_s'):
        if not isinstance(s[field], (int, float)) or not math.isfinite(s[field]) or s[field] <= 0:
            raise ValueError(field + ' must be finite and positive')
    if s['gpus'] not in (0, 1):
        raise ValueError('Use gpus=0 or 1; full-GPU jobs cannot share a GPU')
    if not isinstance(s['capabilities'], list) or not all(c in CAPABILITIES for c in s['capabilities']):
        raise ValueError('Unsupported capability')
    if s['system_retries'] not in (0, 1, 2) or (s['system_retries'] and s['idempotent'] is not True):
        raise ValueError('Up to 2 system retries require idempotent=true')
    if not isinstance(s['env'], dict) or not all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) and '\0' not in v for k,v in s['env'].items()):
        raise ValueError('env must contain valid string environment variables')
    if set(s['env']) & {'CUDA_VISIBLE_DEVICES', 'HOME_RAY_API_TOKEN', 'RAY_AUTH_TOKEN'}:
        raise ValueError('Reserved environment override')
    if not isinstance(s['files'], dict) or not isinstance(s['artifacts'], list):
        raise ValueError('files is a text mapping; artifacts is a path list')
    for name in list(s['files']) + s['artifacts']:
        if not isinstance(name, str) or name.startswith('/') or any(p in ('..','') for p in name.split('/')) or '\\' in name or '\0' in name:
            raise ValueError('Paths must be relative and cannot traverse directories')
    if not all(isinstance(v, str) for v in s['files'].values()) or sum(len(v.encode()) for v in s['files'].values()) > 2*1024*1024:
        raise ValueError('Inline source files limited to 2 MiB of UTF-8 text')
    return s
