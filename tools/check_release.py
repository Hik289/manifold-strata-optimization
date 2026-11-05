"""Check source syntax, local documentation links and common credential/path leaks."""
import argparse
import ast
import ipaddress
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forbid-token', action='append', default=[], help='Additional private identifiers; never stored')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    skipped = {'.git', '.venv', 'venv', '__pycache__', '.latex-build', '.pytest_cache', 'build', 'dist'}
    files = [p for p in root.rglob('*') if p.is_file() and not any(x in skipped or x.endswith('.egg-info') for x in p.relative_to(root).parts)]
    errors = []
    secret = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    private_path = re.compile(r'/(?:Users|home)/[^/\s\"\x27]+|/nas[0-9]+/|/data[0-9]+/')
    for path in files:
        rel = str(path.relative_to(root))
        if path.name == '.env' or path.name.startswith('.env.') or path.suffix in {'.pem', '.key', '.p12', '.pfx'}:
            errors.append((rel, 'credential file'))
        if path.is_symlink():
            errors.append((rel, 'symlink not permitted in release'))
        if path.suffix in {'.pdf', '.png', '.jpg', '.jpeg'}:
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            errors.append((rel, 'unexpected binary file'))
            continue
        if secret.search(content):
            errors.append((rel, 'possible credential; value redacted'))
        if private_path.search(content):
            errors.append((rel, 'absolute private home/storage path'))
        for token in args.forbid_token:
            if token.lower() in content.lower() or token.lower() in rel.lower():
                errors.append((rel, 'blocked identifier; value redacted'))
        for match in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])', content):
            try:
                address = ipaddress.ip_address(match)
            except ValueError:
                continue
            if address.is_private and not address.is_loopback:
                errors.append((rel, 'private IP address'))
        if path.suffix == '.py':
            try:
                ast.parse(content, filename=rel)
            except SyntaxError as error:
                errors.append((rel, f'Python syntax at line {error.lineno}'))
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True)
            if result.returncode:
                errors.append((rel, 'shell syntax'))
        elif path.suffix == '.json':
            try:
                json.loads(content)
            except ValueError:
                errors.append((rel, 'JSON syntax'))
        elif path.suffix == '.md':
            for target in re.findall(r'\]\(([^)\s]+)\)', content):
                if target.startswith(('http:', 'https:', '#', 'mailto:')):
                    continue
                local = unquote(target.split('#')[0])
                if local and not (path.parent / local).exists():
                    errors.append((rel, 'missing local link: ' + local))
    recipes = json.loads((root / 'configs/experiments.json').read_text())
    if len({j['id'] for j in recipes}) != len(recipes):
        errors.append(('configs/experiments.json', 'duplicate recipe ID'))
    for job in recipes:
        if not (root / 'experiments' / job['args'][0]).is_file():
            errors.append(('configs/experiments.json', 'missing trainer'))
        if any(k in job for k in ['worker_pid', 'started', 'ended', 'gpu', 'status']):
            errors.append(('configs/experiments.json', 'live execution metadata'))
    if errors:
        for filename, reason in errors:
            print(f'FAIL {filename}: {reason}')
        raise SystemExit(1)
    print(f'PASS: {len(files)} release files, {len(recipes)} recipes, syntax, local links and credential/path checks.')


if __name__ == '__main__':
    main()
