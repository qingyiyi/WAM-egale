"""Exact eight approved replacement cases after the normal panel closes."""
from pathlib import Path
import argparse
import sys
import traceback

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from baai_resume_v3.runner import LibrarySeparatedRunner
from tf_fullsuite_eval_v2.runner import Runner as OriginalRunner
from baai_approved_retry_v1.contract import install

protocol = install()
read, write, require = protocol.read, protocol.write, protocol.require
PERMISSIONS = protocol.PERMISSIONS


class ApprovedRetryRunner(LibrarySeparatedRunner):
    def spawn(self, argv, log, env, affinity, role):
        if role.startswith(('server-', 'client-')):
            env = dict(env)
            client = role.startswith('client-')
            prefix = Path(self.p['client_python' if client else 'va_python']).resolve().parents[1]
            other = Path(self.p['va_python' if client else 'client_python']).resolve().parents[1]
            minor = 'python3.10' if client else 'python3.12'
            env['LD_LIBRARY_PATH'] = ':'.join(map(str, [
                Path.home() / 'cxz/deps/glvnd-ubuntu24-1.7.0/usr/lib/x86_64-linux-gnu',
                Path(self.p['root']) / 'userlib', prefix / 'lib',
                prefix / 'lib' / minor / 'site-packages/torch/lib', other / 'lib',
                Path('/usr/local/cuda/compat/lib.real'), Path('/usr/local/cuda/compat/lib'),
                Path('/usr/local/nvidia/lib'), Path('/usr/local/nvidia/lib64')]))
            argv = [argv[0], '-B', OWN / 'entry.py', 'client' if client else 'server', *argv[3:]]
        return OriginalRunner.spawn(self, argv, log, env, affinity, role)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    try:
        p = protocol.load(job)
        require(not (job / 'RUN-STARTED.json').exists(), 'one replacement lifecycle; no automatic retry')
        runner = ApprovedRetryRunner(job, p)
        code = runner.execute()
        write(job / 'APPROVED-RETRY-RESULT.json', dict(
            status='COMPLETE_RECORDED_500_CASE_PANEL_NO_QUALITY_CERTIFICATE' if code == 0
                   else 'PARTIAL_OR_FAILED_APPROVED_COMPLETION',
            native_returncode=code,
            approved_case_ids=p['authorized_retry_case_ids'],
            complete_replacement_cases=[cid for cid, result in runner.results.items()
                                        if result['status'] == 'COMPLETE_RECORDED_CASE'],
            original_outcomes_overwritten=False, new_independent_initial_states=False,
            all_owned_groups_empty=read(job / 'FINAL-OWNED-CLEANUP.json')['all_owned_groups_empty'],
            permissions=PERMISSIONS))
        return code
    except BaseException as exc:
        if not (job / 'APPROVED-RETRY-ENTRY-FAILED.json').exists():
            write(job / 'APPROVED-RETRY-ENTRY-FAILED.json', dict(
                error=repr(exc), traceback=traceback.format_exc(), permissions=PERMISSIONS))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
