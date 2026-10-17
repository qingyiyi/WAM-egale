"""Run the original dense full T_POOL after explicit permission for rows 0/1.

Only the process/resource wrapper is new. The original ordered server, client,
collector, complete episode loop and all numerical assertions stay unchanged.
"""
from pathlib import Path
import argparse
import sys
import traceback

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_suite_pipeline_v1 import runner as original
from tf_suite_pipeline_v1.protocol import load, read, write, require, PERMISSIONS


class AuthorizedRunner(original.Runner):
    def guard(self, stage):
        super().guard(stage)
        live = self.job / 'server-worker0/LIVE-COUNTER.json'
        current = read(live)['target_forward_attempts'] if live.exists() else 0
        require(current + self.p['prior_target_forward_debit'] <= 801000,
                'all old and new Object T_POOL attempts count toward original cap')
        if hasattr(self, 'storage_bytes'):
            require(self.storage_bytes + self.p['prior_evidence_bytes'] <= 5 * 2**30,
                    'old and new Object evidence count toward original five GiB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    try:
        p = load(job)
        require(p['phase'] == 'T_POOL' and p['suite'] == 'libero_object'
                and p['explicit_authorized_retry_row_ids'] == [0, 1]
                and p['job_forward_attempt_cap'] + p['prior_target_forward_debit'] == 801000,
                'only explicitly authorized Object full pool completion')
        require(len(p['rows']) == 200 and [r['row_id'] for r in p['rows']] == list(range(200)),
                'original dense server/controller row sequence required')
        debit = 0
        for item in p['closed_prior_attempt_sources']:
            old = Path(item['job']).resolve(strict=True)
            require(read(old / 'FINAL-OWNED-CLEANUP.json')['all_owned_groups_empty']
                    and read(old / 'GUARD-FINAL.json')['status'] == 'CLEAN',
                    'every prior Object phase and guard must be closed')
            live = read(old / 'server-worker0/LIVE-COUNTER.json')
            require(live['active_target_forwards'] == 0
                    and live['target_forward_attempts'] == item['actual_attempts'],
                    'closed original exact attempt debit differs')
            debit += live['target_forward_attempts']
            for terminal in (old / 'episodes').glob('case-*/*/EPISODE-TERMINAL.json'):
                require(read(terminal).get('complete_episode') is not True,
                        'a prior complete outcome cannot be rerun in this namespace')
        require(debit == p['prior_target_forward_debit'] == 296,
                'carry all old actual forwards without resetting budget')
        require(p['no_training_until_complete_original_FIT_CAL'] is True,
                'partial collection is not a training trigger')
        return AuthorizedRunner(job, p).execute()
    except BaseException as exc:
        if not (job / 'RUNNER-ENTRY-FAILED.json').exists():
            write(job / 'RUNNER-ENTRY-FAILED.json', dict(error=repr(exc),
                traceback=traceback.format_exc(), permissions=PERMISSIONS))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
