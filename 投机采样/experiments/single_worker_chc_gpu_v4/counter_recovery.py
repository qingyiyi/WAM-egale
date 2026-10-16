"""Read-only owned-counter recovery; no CUDA, signalling, refunds or resume."""
from counter_journal import recover, JournalError


def recover_owned(path, identity, *, expected_owner=None, parent_last_ack=None):
    row=recover(path,identity)
    owner=row.get('writer')
    if owner is not None and expected_owner is not None and owner!=expected_owner:
        raise JournalError('journal writer differs from pinned worker identity')
    row['parent_last_ack_budget']=parent_last_ack
    row['parent_ack_is_not_final_counter']=True
    row['whole_stage_success_inferred']=False
    row['raw_journal_modified']=False
    return row


def budget_from_recovery(parent, recovered):
    row=recovered.get('ledger')
    if row is not None:
        return {**row,'counter_source':'verified_worker_journal_prefix',
                'counter_snapshot_status':recovered['status'],
                'counts_are_budget_intents_and_host_returns':True,
                'CUDA_device_completion_proven':False}
    # Missing/rejected records never inherit the parent's last acknowledged zero.
    return {**parent,'target_attempted':None,'target_completed':None,
            'stage_attempted':None,'target_failed':None,'target_inflight':None,
            'counter_source':'UNKNOWN_worker_counter',
            'counter_snapshot_status':recovered['status'],
            'CUDA_device_completion_proven':False}
