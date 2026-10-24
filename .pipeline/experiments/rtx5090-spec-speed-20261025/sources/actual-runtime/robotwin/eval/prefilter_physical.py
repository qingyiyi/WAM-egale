import numpy as np

def physical(env):
    scene = env.scene
    actors = []
    articulations = []
    for a in scene.get_all_actors():
        pose = a.get_pose()
        row = dict(name=a.get_name(), p=np.asarray(pose.p).tolist(), q=np.asarray(pose.q).tolist())
        for name in ['get_velocity', 'get_angular_velocity']:
            row[name] = np.asarray(getattr(a, name)()).tolist() if hasattr(a, name) else 'UNKNOWN_API_UNAVAILABLE'
        actors.append(row)
    for a in scene.get_all_articulations():
        pose = a.get_pose()
        articulations.append(dict(name=a.get_name(), p=np.asarray(pose.p).tolist(), q=np.asarray(pose.q).tolist(), qpos=np.asarray(a.get_qpos()).tolist(), qvel=np.asarray(a.get_qvel()).tolist()))
    return dict(actors=actors, articulations=articulations)
