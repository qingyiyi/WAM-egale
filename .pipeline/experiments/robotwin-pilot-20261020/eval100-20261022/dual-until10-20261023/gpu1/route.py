"""Only returned policy calls advance the per-episode SST scheduler."""
class Router:
    def __init__(self, arm):
        assert arm in ('T', 'C', 'P')
        self.arm, self.returned = arm, 0
    def next(self):
        return 'T' if self.arm == 'T' or (self.arm == 'P' and self.returned % 3 == 2) else 'S'
    def commit_returned(self):
        self.returned += 1
