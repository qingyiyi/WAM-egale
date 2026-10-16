"""Fresh four-episode output scope: finite bytes plus runtime free-space floor."""
def entry(free,p):
    initial=p['disk_initial_min_bytes']
    floor=p['disk_runtime_floor_bytes']
    output=p['new_output_bytes_cap']
    if min(initial,floor,output)<=0 or floor+output>=initial:
        raise ValueError('finite output and free-space reserve required')
    if free<initial:raise RuntimeError('initial free disk below bounded E2E reserve')

def runtime(free,used,p):
    if free<p['disk_runtime_floor_bytes']:raise RuntimeError('disk below runtime floor')
    if used>p['new_output_bytes_cap']:raise RuntimeError('new job output byte cap')
