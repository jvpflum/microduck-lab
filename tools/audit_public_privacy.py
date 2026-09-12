"""Review public files without printing potentially sensitive matching values."""
import argparse
import re
import subprocess
from pathlib import Path

PATTERNS = {
    'private_absolute_path': re.compile(r'/(?:home|Users)/(?:juice|jarro)/'),
    'private_overlay_address': re.compile(r'\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.(?:\d{1,3})\.(?:\d{1,3})\b'),
    'tailscale_authorization': re.compile(r'https://login\.tailscale\.com/a/\w+'),
    'credential_token': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{20,}|tskey-[A-Za-z0-9_-]{16,})'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'literal_password_or_secret': re.compile(r'(?i)(?:password|passwd|secret_access_key|api_key|access_token|root_password)\s*[=:]\s*[\"\x27]([A-Za-z0-9_+/=-]{12,})[\"\x27]'),
}


def findings(data):
    return [category for category, pattern in PATTERNS.items() if pattern.search(data)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    violations = 0
    if args.history:
        # Include removed files and changed content, without displaying matched lines.
        stream = subprocess.check_output(['git', '-C', str(root), 'log', '--all', '-p', '--no-ext-diff', '--format=commit %H'], text=True, errors='replace')
        for category in findings(stream):
            print('history:', category)
            violations += 1
    else:
        names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-c', '-o', '--exclude-standard', '-z']).decode().split('\0')
        for name in sorted(set(names)-{''}):
            path = root / name
            if not path.is_file() or name.startswith('upstream/'):
                continue
            if path.suffix == '.onnx':
                import onnx
                model = onnx.load(path, load_external_data=False)
                data = '\n'.join(p.value for p in model.metadata_props)
            else:
                data = path.read_bytes().decode('utf-8', errors='replace')
            for category in findings(data):
                print(name + ': ' + category)
                violations += 1
    print(f'Findings requiring review: {violations}')
    return bool(violations)

if __name__ == '__main__':
    raise SystemExit(main())
