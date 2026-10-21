def __init__(self, pose, joints, all_joints):
    super(CuroboPlanner, self).__init__()
    self.motion_gen_batch = Motion()
    self._batch_warmup_pending = True
