#!/usr/bin/env python3
"""Build once, run every bounded LAPACK family, and retain failures/evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

KNOWN_FILE = 'lapack-known-failures.json'
FAMILIES = (
    'common', 'svd', 'hessenberg', 'schur', 'nonsymmetric_evd',
    'schur_reorder', 'balance', 'tridiagonal_evd', 'self_adjoint_evd',
    'lu', 'cholesky', 'indefinite', 'qr', 'triangular', 'lstsq', 'svd_large',
)
# Local measurements are documented in docs/lapack-validation.md. Hosted
# timings remain unmeasured; these bounds leave several times the local runtime.
LIMITS = dict(common=30, svd=120, hessenberg=60, schur=90,
              nonsymmetric_evd=60, schur_reorder=30, balance=30,
              tridiagonal_evd=90, self_adjoint_evd=120, lu=180,
              cholesky=60, indefinite=120, qr=90, triangular=60, lstsq=60,
              svd_large=300)
# Upstream's large SVD matrix tests: too slow for every test run (about a minute each
# locally), so they are ignored there and run here, in a family of their own.
LARGE_SVD = ('linalg::svd::bidiag_svd::tests::test_josef',)
CASE = re.compile(
    r'LAPACK_CASES family=(\S+) dtype=(\S+) planned=(\d+) attempted=(\d+) '
    r'call_failures=(\d+) diagnostics=(\d+)(?: auxiliary_attempted=(\d+))?'
)
SUMMARY = re.compile(r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;')


def classify(name):
    if name in LARGE_SVD:
        return 'svd_large'
    if name.startswith('linalg::svd::tests::'):
        return 'svd'
    leaf = name.rsplit('::', 1)[-1]
    if leaf.startswith(('lapack_common_', 'lapack_lin_')):
        return 'common'
    for family in sorted(FAMILIES, key=len, reverse=True):
        if leaf.startswith('lapack_' + family + '_'):
            return family
    if '::lapack::' in name:
        raise ValueError('unclassified validation test: ' + name)
    return None


def selected_tests(listing):
    groups = {family: [] for family in FAMILIES}
    for line in listing.splitlines():
        if line.endswith(': test'):
            name = line[:-6]
            family = classify(name)
            if family:
                groups[family].append(name)
    empty = [family for family, names in groups.items() if not names]
    if empty:
        raise ValueError('empty test selection: ' + ', '.join(empty))
    return groups


def run_command(command, log, timeout):
    start = time.perf_counter()
    with log.open('w') as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            stream.write('\nValidation command timed out.\n')
            code = 124
    return code, time.perf_counter() - start


def run_families(binary, groups, output, command=run_command, limit_scale=1.0):
    rows = []
    for family in FAMILIES:
        names = groups[family]
        if not names:
            raise ValueError('empty test selection: ' + family)
        log = output / (family + '.log')
        code, seconds = command(
            [str(binary), '--exact', *names, '--include-ignored', '--nocapture', '--test-threads=1'],
            log, LIMITS[family] * limit_scale,
        )
        contents = log.read_text()
        summaries = SUMMARY.findall(contents)
        executed = sum(map(int, summaries[-1])) if summaries else 0
        rows.append(dict(family=family, selected=len(names), executed=executed,
                         exit_code=code, seconds=round(seconds, 3),
                         complete=executed == len(names), log=log.name))
        print(f'{family}: selected={len(names)} executed={executed} exit={code} {seconds:.2f}s', flush=True)
    return rows


def load_known(root):
    """The known-failing tests and the issue that tracks each one.

    A test listed here is reported as known rather than as a new failure; a listed test that
    passes fails the run, so a fix has to drop its entry in the same change.
    """
    path = Path(root) / 'scripts' / KNOWN_FILE
    document = json.loads(path.read_text())
    known = {}
    for entry in document['known']:
        issue = entry['issue']
        if not isinstance(issue, int):
            raise ValueError(f'known failure without an issue number: {entry}')
        for test in entry['tests']:
            if test in known:
                raise ValueError(f'test listed twice in the known failures: {test}')
            known[test] = issue
    return known


def failed_tests(contents):
    """The test names libtest lists in the final `failures:` block.

    The output carries two such blocks when a test fails - one before the per-test stdout
    sections and the list itself at the end - so the last one before `test result:` is the
    list of failing names.
    """
    lines = contents.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == 'failures:']
    for start in reversed(starts):
        names = []
        for line in lines[start + 1:]:
            if not line.strip():
                if names:
                    break
                continue
            if line.startswith('test result:'):
                break
            names.append(line.strip())
        if names:
            return names
    return []


def classify_failures(rows, groups, known, output):
    """Split the failing tests into known (with their issue) and unexpected ones."""
    known_failing, unexpected, stale = [], [], []
    failing = {}
    for row in rows:
        names = failed_tests((output / row['log']).read_text())
        failing[row['family']] = names
        for name in names:
            if name in known:
                known_failing.append(dict(test=name, issue=known[name], family=row['family']))
            else:
                unexpected.append(dict(test=name, family=row['family']))
    listed = {test for names in groups.values() for test in names}
    for test in sorted(set(known) & listed):
        if not any(test in names for names in failing.values()):
            stale.append(dict(test=test, issue=known[test]))
    return known_failing, unexpected, stale


def case_records(contents):
    records = []
    for family, dtype, *numbers in CASE.findall(contents):
        values = [int(n or 0) for n in numbers]
        records.append(dict(family=family, dtype=dtype, **dict(zip(
            ('planned', 'attempted', 'call_failures', 'diagnostics', 'auxiliary_attempted'), values))))
    return records


def expected_records(groups):
    expected = sum('::lapack::' in name and name.endswith(('_cases', '_parallel', '_subnormal', '_range'))
                       for family,names in groups.items() if family != 'svd' for name in names)
    svd_expected = sum('::lapack_svd_' in name and name.endswith((
            'dchkbd_15_to_19', 'diagonal_families', 'rotated_families', 'bidiagonal_family',
            'rank_one_at_recursion_splits', 'prefilled_outputs', 'layouts_and_reused_buffers', 'parallel_paths'))
            for name in groups['svd'])
    return expected + svd_expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('target/lapack-validation'))
    parser.add_argument('--limit-scale', type=float, default=1.0,
                        help='multiply every family time limit (slower hosts, e.g. the aarch64 lane)')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CARGO_PROFILE_DEV_DEBUG', '0')
    os.environ.setdefault('CARGO_INCREMENTAL', '0')
    envlog = output / 'environment.log'
    with envlog.open('w') as stream:
        for command in (['git', 'rev-parse', 'HEAD'], ['git', 'status', '--short'],
                        ['uname', '-a'], ['rustc', '-Vv']):
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    code, seconds = run_command(
        ['cargo', 'test', '-p', 't4a-faer', '--lib', '--locked', '--no-run', '--message-format=json'],
        output / 'build.log', 1200,
    )
    report = dict(build_exit_code=code, build_seconds=round(seconds, 3), families=[], cases=[])
    try:
        if code:
            raise ValueError(f'test build failed: exit {code}')
        binaries = []
        for line in (output / 'build.log').read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (item.get('reason') == 'compiler-artifact' and item.get('executable')
                    and item.get('target', {}).get('name') == 'faer'
                    and item.get('profile', {}).get('test')):
                binaries.append(item['executable'])
        if len(binaries) != 1:
            raise ValueError(f'expected one faer test binary, got {binaries}')
        binary = Path(binaries[0])
        listing = subprocess.run([str(binary), '--list', '--format', 'terse'],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
        (output / 'selection.log').write_text(listing)
        groups = selected_tests(listing)
        report['families'] = run_families(binary, groups, output, limit_scale=args.limit_scale)
        known = load_known(Path(__file__).resolve().parent.parent)
        report['known_failing'], report['unexpected'], report['stale'] = classify_failures(
            report['families'], groups, known, output)
        for row in report['families']:
            report['cases'].extend(case_records((output / row['log']).read_text()))
        report['execution_seconds'] = round(sum(row['seconds'] for row in report['families']), 3)
        # Every selected numeric group must emit a record, including failing groups.
        expected = expected_records(groups)
        if len(report['cases']) != expected:
            raise ValueError(f'case records missing: expected {expected}, got {len(report["cases"])}')
        if any(row['planned'] != row['attempted'] for row in report['cases']):
            raise ValueError('incomplete configured case loop')
        for row in report['families']:
            if row['exit_code'] != 0 and not row['complete']:
                raise ValueError(f"{row['family']}: the family did not complete ({row['executed']} of {row['selected']})")
    except (ValueError, subprocess.SubprocessError, OSError) as error:
        report['error'] = str(error)
        print(error, flush=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    unexpected = report.get('unexpected') or []
    stale = report.get('stale') or []
    known = report.get('known_failing') or []
    if known:
        print(f'known failures: {len(known)} (issues ' +
              ', '.join(f"#{issue}" for issue in sorted({entry["issue"] for entry in known})) + ')', flush=True)
    for entry in unexpected:
        print(f'UNEXPECTED FAILURE {entry["family"]}: {entry["test"]}', flush=True)
    for entry in stale:
        print(f'KNOWN FAILURE NOW PASSES (remove it from scripts/{KNOWN_FILE}): '
              f'#{entry["issue"]} {entry["test"]}', flush=True)
    # A known failure keeps the run green; a new failure, a listed test that now passes, an
    # incomplete family or a broken case loop does not. A family that failed without a listed
    # failing test (a crash, a timeout, a harness error) has no known explanation and fails too.
    known_families = {entry['family'] for entry in known}
    unexplained = [row['family'] for row in report['families']
                   if row['exit_code'] != 0 and row['family'] not in known_families]
    for family in unexplained:
        print(f'FAILED WITHOUT A LISTED TEST {family}: see its log', flush=True)
    report['unexplained'] = unexplained
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(bool(report.get('error')) or bool(unexpected) or bool(stale) or bool(unexplained))


if __name__ == '__main__':
    raise SystemExit(main())
