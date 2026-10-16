"""Target forwards are outside this direct original-torch comparison."""
STAGES = {}


class Ledger:
    def before_forward(self):
        raise RuntimeError('zero target forward cap')
