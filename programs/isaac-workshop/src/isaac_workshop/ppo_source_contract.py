"""Version-independent, exact source admission for the inspected public PPO.

Python 3.12 and 3.14 differ in ast.dump's empty-field formatting. Serialize
every AST field explicitly instead; do not omit, normalize, or ignore syntax.
The independently pinned source text and whole module bytes remain required.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

UPSTREAM_MODULE_SHA256 = '01a82cea45fa9d8ccb161bd75e07d8623d25450ff92982e9c9d59ca8a9b78393'
UPSTREAM_UPDATE_SOURCE_SHA256 = '8609eb05e3ec0aec1a0229cc17a260eb3dea2a3e07a0de75b1a6a5d4b55ac881'
UPSTREAM_UPDATE_CANONICAL_AST_SHA256 = 'a3ee28fc35255e92a61e24207e1edee3fc99b1394f070f68a5c38f4ac16d8132'


def canonical_ast(value):
    if isinstance(value, ast.AST):
        return [type(value).__name__, {key: canonical_ast(child) for key, child in ast.iter_fields(value)}]
    if isinstance(value, list):
        return [canonical_ast(child) for child in value]
    return value


def canonical_ast_sha256(node):
    encoded = json.dumps(canonical_ast(node), sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_method_source(source):
    text = textwrap.dedent(source.replace('\r\n', '\n')).strip()
    tree = ast.parse(text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != 'update':
        raise RuntimeError('Installed PPO.update source is not the expected method')
    node = tree.body[0]
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    ast_hash = canonical_ast_sha256(node)
    if source_hash != UPSTREAM_UPDATE_SOURCE_SHA256 or ast_hash != UPSTREAM_UPDATE_CANONICAL_AST_SHA256:
        raise RuntimeError('Installed PPO.update differs from inspected pinned source: '
                           f'text={source_hash}, canonical_ast={ast_hash}')
    return {'verified': True, 'update_source_sha256': source_hash, 'update_canonical_ast_sha256': ast_hash,
            'python_version': sys.version.split()[0],
            'serialization': 'explicit ast.iter_fields including empty values; sorted compact JSON',
            'default_ast_dump_used_for_admission': False}


def verify_ppo_update(update):
    source_path = inspect.getsourcefile(update)
    if source_path is None:
        raise RuntimeError('Installed PPO.update source file unavailable')
    module_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    if module_hash != UPSTREAM_MODULE_SHA256:
        raise RuntimeError('Installed PPO module differs from inspected pinned source: '+module_hash)
    result = verify_method_source(inspect.getsource(update))
    result['module_sha256'] = module_hash
    return result
