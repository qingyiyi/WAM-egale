"""Reuse only sealed completed rows, then run every never-started full episode."""
from pathlib import Path
import argparse
import copy
import sys

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from tf_suite_pipeline_v1.runner import Runner as BaseRunner
from tf_suite_pipeline_v1.protocol import load, read, write, require, PERMISSIONS
from baai_suite_pipeline_v1.runtime import environment, PIPELINE


class Runner(BaseRunner):
    def spawn(self, argv, log, env, affinity, role):
        argv = list(argv)
        if role.startswith(('server-', 'client-')):
            kind = 'server' if role.startswith('server-') else 'client'
            require(Path(argv[2]).resolve() == PIPELINE / (kind + '_entry.py'), 'fixed suite worker source')
            argv = [argv[0], argv[1], OWN / 'entry.py', kind, *argv[3:]]
            env = environment(self.p, env, interpreter=argv[0])
        return super().spawn(argv, log, env, affinity, role)

    def run_row(self, spec):
        if spec['row_id'] not in self.p.get('migration_readonly_completed_row_ids', []):
            return super().run_row(spec)
        source = read(Path(self.p['migration_readonly_row_results']) / f'ROW-{spec["row_id"]:04d}-RESULT.json')
        require(source['complete_episode'] is True and source['case_id'] == spec['case_id']
                and source['row_id'] == spec['row_id'] and source['arm'] == spec['arm'] == 'C'
                and source['permissions'] == PERMISSIONS, 'only complete immutable CURRENT rows can be reused')
        for name in ('EPISODE-TERMINAL.json', 'COUNTER.json', 'FINISH_READY.json', 'AUDIT.json', 'COLLECTION-AUDIT.json'):
            record = read(Path(spec['out']) / name)
            require(record['case_id'] == spec['case_id'] and record['row_id'] == spec['row_id']
                    and record['arm'] == 'C' and record['permissions'] == PERMISSIONS, 'sealed migration receipt identity')
        result = copy.deepcopy(source)
        result.update(status='READONLY_MIGRATED_COMPLETE_CURRENT_EPISODE', no_new_episode=True,
                      new_target_forwards=0, source_out=str(Path(spec['out'])),
                      migration_original_source=source['source_out'])
        self.results[spec['row_id']] = result
        write(self.job / f'ROW-{spec["row_id"]:04d}-RESULT.json', result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    job = parser.parse_args().job.resolve(strict=True)
    p = load(job)
    require(p['suite'] == 'libero_object' and not (job / 'RUN-STARTED.json').exists(), 'one fresh actual Object phase')
    readonly = p.get('migration_readonly_completed_row_ids', [])
    require(not readonly or (p['phase'] == 'CURRENT_DATA' and readonly == list(range(16))),
            'only the actually sealed original16 may be resumed readonly')
    code = Runner(job, p).execute()
    if readonly:
        terminal = read(job / 'SUPERVISOR-RESULT.json')
        write(job / 'MIGRATION-RESUMPTION-RESULT.json', dict(
            schema='object-complete-row-resumption/v1', readonly_completed_C_rows=len(readonly),
            new_complete_episodes=terminal['actual_new_complete_episodes'],
            union_complete_rows=terminal['complete_rows'], old_rows_not_rerun=True,
            legacy_reused_complete_target_rows_field_is_generic_count_of_readonly_rows=True,
            legacy_field_does_not_mean_these_C_rows_are_Target=True,
            phase_status=terminal['status'], permissions=PERMISSIONS))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
