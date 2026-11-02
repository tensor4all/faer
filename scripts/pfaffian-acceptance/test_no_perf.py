import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('acceptance', Path(__file__).with_name('run.py'))
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


class NoPerfTests(unittest.TestCase):
    def test_unavailable_requires_exact_opt_in(self):
        for flag in (None, '0', 'true'):
            for executable in (None, '/usr/bin/perf'):
                with self.subTest(flag=flag, executable=executable), tempfile.TemporaryDirectory() as d, \
                     patch.dict(os.environ, {} if flag is None else {'T4A_ALLOW_NO_PERF': flag}, clear=True), \
                     patch.object(acceptance.shutil, 'which', return_value=executable), \
                     patch.object(acceptance.subprocess, 'run') as run:
                    run.return_value.returncode = 255
                    with self.assertRaisesRegex(RuntimeError, 'perf unavailable'):
                        acceptance.perf_preflight(Path(d))

    def test_opt_in_records_missing_and_denied_perf(self):
        for executable, rc in ((None, None), ('/usr/bin/perf', 255)):
            with self.subTest(executable=executable), tempfile.TemporaryDirectory() as d, \
                 patch.dict(os.environ, {'T4A_ALLOW_NO_PERF': '1'}, clear=True), \
                 patch.object(acceptance.shutil, 'which', return_value=executable), \
                 patch.object(acceptance.subprocess, 'run') as run:
                run.return_value.returncode = rc
                status = acceptance.perf_preflight(Path(d))
                self.assertTrue(status.startswith('perf unavailable (perf_event_paranoid='))
                self.assertTrue((Path(d) / 'perf-preflight.log').exists())
                self.assertEqual(run.called, executable is not None)
                with patch.object(acceptance.subprocess, 'run', side_effect=AssertionError('unexpected profiling')):
                    acceptance.finish(Path(d), status, 'unused', {}, {})
                report = (Path(d) / 'REPORT.md').read_text()
                self.assertIn('## Profile\nnot measured\n', report)
                self.assertIn('## Bandwidth\nnot measured\n', report)
                self.assertFalse(list(Path(d).glob('profile-*')))

    def test_no_perf_runs_full_timing_flow_with_fake_commands(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bench = root / 'benchmark'
            guard = bench / 'benchmarks/scripts/pinned.sh'
            guard.parent.mkdir(parents=True)
            guard.touch()
            (guard.parent / 'idle_cpus.py').touch()
            out = root / 'result'
            (root / 'libopenblas.so').touch()
            original_read = Path.read_text

            def read_text(path, *args, **kwargs):
                name = str(path)
                if name == '/proc/cpuinfo':
                    return 'AMD Ryzen AI 9 HX 470'
                if name.endswith('/cache/index3/shared_cpu_list'):
                    return '4-11,16-23'
                if name.endswith('/topology/core_id'):
                    return name.split('/cpu/cpu')[1].split('/')[0]
                if name.endswith('/cpufreq/boost'):
                    return '0'
                if name == '/proc/sys/kernel/perf_event_paranoid':
                    return '4'
                return original_read(path, *args, **kwargs)

            def command(args, **kwargs):
                args = [str(x) for x in args]
                if args[0] == 'tar':
                    p = Path(args[-1]) / 'faer/src/linalg/pfaffian/mod.rs'
                    p.parent.mkdir(parents=True)
                    p.write_text('const PANEL: usize = 24;')
                elif args[0] == 'cargo':
                    p = Path(args[-1]).parent / 'target/release/pfaffian-faer-probe'
                    p.parent.mkdir(parents=True)
                    p.touch()
                elif args[0] == 'gfortran' and '-o' in args:
                    Path(args[-1]).touch()

            class Archive:
                stdout = io.BytesIO(b'')
                def wait(self):
                    return 0

            def output(args, **kwargs):
                args = [str(x) for x in args]
                if args[0] == 'taskset':
                    return json.dumps({'pf_re': 2.5, 'pf_im': 0.0})
                if args[0] == 'ldd':
                    return f'libopenblas.so => {root}/libopenblas.so (0x0)'
                if args[0] in ('gfortran', 'cargo', 'rustc'):
                    return 'mock version'
                if 'rev-parse' in args:
                    return acceptance.PFA if 'pfapack' in args[2] else 'a' * 40
                return ''

            def run(args, **kwargs):
                if args[0] == 'perf':
                    return type('Result', (), {'returncode': 1})()
                row = {'pf_re': 2.5, 'pf_im': 0.0, 'median_seconds': 0.5,
                       'samples_seconds': [0.5] * 10, 'batch_iterations': [1] * 10}
                return type('Result', (), {'returncode': 0, 'stdout': json.dumps(row)})()

            # Fake source list is needed by the Fortran build step, not executed here.
            pfadir = out / 'pfapack/original_source/fortran'
            def fake_glob(path, pattern):
                if str(path) == str(pfadir):
                    return iter([pfadir / 'fake.f'])
                return iter([])

            with patch.object(acceptance, 'BENCH', bench), \
                 patch.object(acceptance.sys, 'argv', ['run.py', str(out)]), \
                 patch.dict(os.environ, {'T4A_ALLOW_NO_PERF': '1'}, clear=True), \
                 patch.object(acceptance.os, 'sched_getaffinity', return_value={0, 1, 2, 3}), \
                 patch.object(Path, 'read_text', read_text), \
                 patch.object(Path, 'glob', fake_glob), \
                 patch.object(acceptance, 'command', side_effect=command), \
                 patch.object(acceptance, 'output', side_effect=output), \
                 patch.object(acceptance, 'digest', return_value='fake-sha'), \
                 patch.object(acceptance.shutil, 'which', return_value='/usr/bin/perf'), \
                 patch.object(acceptance.subprocess, 'Popen', return_value=Archive()), \
                 patch.object(acceptance.subprocess, 'run', side_effect=run):
                acceptance.main()
            rows = json.loads((out / 'summary.json').read_text())
            self.assertEqual(len(rows), 24)
            self.assertFalse(any(r['formal_thresholds_met'] for r in rows if r['n'] >= 256))
            self.assertEqual(len((out / 'samples.csv').read_text().splitlines()), 193)
            self.assertIn('perf unavailable (perf_event_paranoid=4)', (out / 'manifest.json').read_text())
            self.assertIn('## Profile\nnot measured\n', (out / 'REPORT.md').read_text())
            self.assertIn('## Bandwidth\nnot measured\n', (out / 'REPORT.md').read_text())

    def test_available_still_profiles_and_failure_stops(self):
        for env in ({}, {'T4A_ALLOW_NO_PERF': '1'}):
            with self.subTest(env=env), tempfile.TemporaryDirectory() as d, \
                 patch.dict(os.environ, env, clear=True), \
                 patch.object(acceptance.shutil, 'which', return_value='/usr/bin/perf'), \
                 patch.object(acceptance.subprocess, 'run') as run:
                run.return_value.returncode = 0
                self.assertEqual(acceptance.perf_preflight(Path(d)), 'perf available')
                run.return_value.returncode = 75
                run.return_value.stdout = ''
                with self.assertRaisesRegex(RuntimeError, 'failed or guard rejected'):
                    acceptance.finish(Path(d), 'perf available', 'guard', {'base': 'base'}, {})
                self.assertFalse((Path(d) / 'REPORT.md').exists())


if __name__ == '__main__':
    unittest.main()
