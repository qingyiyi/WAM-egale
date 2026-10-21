def __init__(self, robot_origion_pose, active_joints_name, all_joints, yml_path=None):
    super(CuroboPlanner, self).__init__()
    ta.setup_logging('CRITICAL')
    logger.setup_logger(level='error', logger_name="'curobo")
    if yml_path != None:
        self.yml_path = yml_path
    else:
        raise ValueError('[Planner.py]: CuroboPlanner yml_path is None!')
    self.robot_origion_pose = robot_origion_pose
    self.active_joints_name = active_joints_name
    self.all_joints = all_joints
    with open(self.yml_path, 'r') as f:
        yml_data = yaml.safe_load(f)
    self.frame_bias = yml_data['planner']['frame_bias']
    if True:
        world_config = {'cuboid': {'table': {'dims': [0.7, 2, 0.04], 'pose': [self.robot_origion_pose.p[1], 0.0, 0.74 - self.robot_origion_pose.p[2], 1, 0, 0, 0.0]}}}
    motion_gen_config = MotionGenConfig.load_from_robot_config(self.yml_path, world_config, interpolation_dt=1 / 250, num_trajopt_seeds=1)
    self.motion_gen = MotionGen(motion_gen_config)
    self.motion_gen.warmup()
    motion_gen_config = MotionGenConfig.load_from_robot_config(self.yml_path, world_config, interpolation_dt=1 / 250, num_trajopt_seeds=1, num_graph_seeds=1)
    self.motion_gen_batch = MotionGen(motion_gen_config)
    self._batch_warmup_pending = True
