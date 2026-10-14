"""Run all tests and integrity/parity checks, preserving each exit status and log."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def main():
    output = REPO / 'reports' / 'portable_verification' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(REPO / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    checks = [
        ('integrity', ['scripts/verify_reproducibility.py']),
        ('pytest', ['-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider', '--junitxml', str(output / 'pytest.xml')]),
        ('parity', ['scripts/verify_step8_parity.py']),
    ]
    results = []
    for name, args in checks:
        print(f'\n=== {name} ===', flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, *args], cwd=REPO, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                log.write(line)
                print(line, end='', flush=True)
            results.append({'check': name, 'exit_code': process.wait()})
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f'\nResults: {output.relative_to(REPO)}')
    return int(any(row['exit_code'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
