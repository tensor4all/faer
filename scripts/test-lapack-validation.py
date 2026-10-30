#!/usr/bin/env python3
"""Small offline checks for selection, continuation and status accounting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location('validation', Path(__file__).with_name('run-lapack-validation.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class Runner(unittest.TestCase):
    def test_classification_and_empty_selection(self):
        self.assertEqual(v.classify('linalg::evd::lapack::f64::lapack_schur_cases'), 'schur')
        self.assertEqual(v.classify('linalg::evd::lapack::f64::lapack_schur_reorder_cases'), 'schur_reorder')
        self.assertEqual(v.classify('linalg::qr::lapack::c64::lapack_qr_reflector_checkers'), 'qr')
        self.assertEqual(v.classify('linalg::svd::tests::lapack_svd_f32::lapack::diagonal_families'), 'svd')
        self.assertEqual(v.classify('linalg::svd::bidiag_svd::tests::test_josef'), 'svd_large')
        self.assertIsNone(v.classify('linalg::svd::bidiag_svd::tests::test_deflation43'))
        # a self-adjoint accuracy test beside the case suite (tensor4all/faer#100): classified, not a case record
        name = 'linalg::evd::lapack::c32::lapack_self_adjoint_evd_qr_spectrum_accuracy'
        self.assertEqual(v.classify(name), 'self_adjoint_evd')
        self.assertEqual(v.expected_records({'svd': [], 'self_adjoint_evd': [name]}), 0)
        with self.assertRaisesRegex(ValueError, 'empty'):
            v.selected_tests('')
        with self.assertRaisesRegex(ValueError, 'unclassified'):
            v.classify('linalg::evd::lapack::f64::unmapped')

    def test_continues_after_failure_timeout_and_incomplete_output(self):
        calls = []
        def fake(command, log, timeout):
            calls.append(command)
            family = log.stem
            text = 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;'
            if family == 'svd':
                text = 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out;'
            log.write_text('timeout' if family == 'lu' else text)
            return (101 if family == 'svd' else 124 if family == 'lu' else 0), 0.1
        groups = {family: [family + '_test'] for family in v.FAMILIES}
        with tempfile.TemporaryDirectory() as directory:
            rows = v.run_families('fake-binary', groups, Path(directory), fake)
        self.assertEqual(len(calls), len(v.FAMILIES))
        self.assertEqual(next(r for r in rows if r['family'] == 'svd')['exit_code'], 101)
        self.assertFalse(next(r for r in rows if r['family'] == 'lu')['complete'])
        self.assertEqual(rows[-1]['family'], 'svd_large')
        self.assertTrue(rows[-1]['complete'])
        self.assertTrue(all('--exact' in c and '--include-ignored' in c for c in calls))

    def test_svd_calibration_is_not_a_numeric_group(self):
        groups = {'svd': [
            'linalg::svd::tests::lapack_svd_f32::lapack::panics_and_errors_do_not_skip_later_cases',
            'linalg::svd::tests::lapack_svd_f32::lapack::diagonal_families'],
            'lu': ['linalg::lu::lapack::f32::lapack_lu_cases']}
        self.assertEqual(v.expected_records(groups), 2)

    def test_known_failures_keep_the_run_green_and_new_ones_do_not(self):
        known = {'linalg::svd::tests::x': 110}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'svd.log').write_text('failures:\n    linalg::svd::tests::x\n\ntest result: FAILED. 0 passed; 1 failed;\n')
            (path / 'lu.log').write_text('test result: ok. 1 passed; 0 failed;\n')
            rows = [dict(family='svd', log='svd.log'), dict(family='lu', log='lu.log')]
            groups = {'svd': ['linalg::svd::tests::x'], 'lu': ['linalg::lu::tests::y']}
            known_failing, unexpected, stale = v.classify_failures(rows, groups, known, path)
        self.assertEqual([entry['issue'] for entry in known_failing], [110])
        self.assertEqual((unexpected, stale), ([], []))

    def test_an_unlisted_failure_and_a_fixed_listed_one_are_reported(self):
        known = {'linalg::svd::tests::x': 110, 'linalg::lu::tests::y': 112}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'svd.log').write_text('failures:\n    linalg::svd::tests::new\n\ntest result: FAILED.\n')
            (path / 'lu.log').write_text('test result: ok. 1 passed; 0 failed;\n')
            rows = [dict(family='svd', log='svd.log'), dict(family='lu', log='lu.log')]
            groups = {'svd': ['linalg::svd::tests::new'], 'lu': ['linalg::lu::tests::y']}
            known_failing, unexpected, stale = v.classify_failures(rows, groups, known, path)
        self.assertEqual(unexpected, [dict(test='linalg::svd::tests::new', family='svd')])
        self.assertEqual([entry['test'] for entry in stale], ['linalg::lu::tests::y'])
        self.assertEqual(known_failing, [])

    def test_failed_tests_reads_the_summary_section_only(self):
        log = ('test linalg::a ... FAILED\n\nfailures:\n\n---- linalg::a stdout ----\n'
               'some text\n\nfailures:\n    linalg::a\n\ntest result: FAILED.\n')
        self.assertEqual(v.failed_tests(log), ['linalg::a'])

    def test_the_ledger_names_an_issue_for_every_test(self):
        # An empty ledger (no known failures on main) is the healthy state and
        # satisfies this vacuously; the load itself still checks the format.
        known = v.load_known(Path(__file__).resolve().parent.parent)
        self.assertTrue(all(isinstance(issue, int) for issue in known.values()))

    def load_ledger(self, entries):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'scripts').mkdir()
            (Path(root) / 'scripts' / v.KNOWN_FILE).write_text(json.dumps({'known': entries}))
            return v.load_known(root)

    def test_ledger_format_is_enforced_whatever_main_lists(self):
        # These cases do not depend on the repository ledger, which may be empty.
        self.assertEqual(self.load_ledger([]), {})
        self.assertEqual(self.load_ledger([{'issue': 7, 'tests': ['linalg::a', 'linalg::b']}]),
                         {'linalg::a': 7, 'linalg::b': 7})
        with self.assertRaisesRegex(ValueError, 'without an issue'):
            self.load_ledger([{'issue': None, 'tests': ['linalg::a']}])
        with self.assertRaisesRegex(ValueError, 'listed twice'):
            self.load_ledger([{'issue': 1, 'tests': ['linalg::a']}, {'issue': 2, 'tests': ['linalg::a']}])

    def test_case_counts(self):
        row, = v.case_records('LAPACK_CASES family=lu dtype=f32 planned=2 attempted=2 call_failures=1 diagnostics=1 auxiliary_attempted=4')
        self.assertEqual((row['planned'], row['attempted'], row['auxiliary_attempted']), (2, 2, 4))


if __name__ == '__main__':
    unittest.main()
