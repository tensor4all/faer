#!/usr/bin/env python3
"""Print the run's new and stale failures as GitHub Actions step outputs.

Reads target/lapack-validation/summary.json (written by run-lapack-validation.py) and emits
`unexpected=` and `stale=` lines, which the failure reporter puts in its issue. Missing or
unreadable evidence is reported as a line the reporter shows, never as silence.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('target/lapack-validation'))
    args = parser.parse_args()
    summary = args.output / 'summary.json'
    try:
        report = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f'unexpected=the validation summary could not be read: {error}')
        print('stale=')
        return 0
    unexpected = report.get('unexpected') or []
    stale = report.get('stale') or []
    unexplained = report.get('unexplained') or []
    lines = [f'{entry["family"]}: {entry["test"]}' for entry in unexpected]
    lines += [f'{family}: the family failed without a listed test, see its log' for family in unexplained]
    if report.get('error'):
        lines.insert(0, f'the run did not complete: {report["error"]}')
    print('unexpected=' + ('\n'.join(lines) if lines else ''))
    print('stale=' + '\n'.join(f'#{entry["issue"]} {entry["test"]}' for entry in stale))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
