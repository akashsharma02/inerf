import math
import numpy as np
import torch
from lieutils import SE3

render_poses = []
poses = []
poses.append(np.eye(4))
for _, _ in enumerate(poses):
    # Choose between three axis randomly
    rand_axis, rand_rot, rand_trans = np.random.choice([0, 1, 2]), np.random.uniform(-40, 40), np.random.uniform(-0.2, 0.2, 3)
    rotation_axis = np.zeros(6)
    rotation_axis[rand_axis], rotation_axis[3:] = rand_rot, rand_trans
    render_poses.append(SE3.Exp(torch.from_numpy(rotation_axis)))



render_poses = []
rad = (math.radians(-40.0), math.radians(40.0))
for _, _ in enumerate(poses):
    # Choose between three axis randomly
    rand_axis, rand_rot, rand_trans = np.random.choice([0, 1, 2]), np.random.uniform(rad[0], rad[1]), np.random.uniform(-0.2, 0.2, 3)
    rotation_axis = np.zeros(6)
    rotation_axis[rand_axis], rotation_axis[3:] = rand_rot, rand_trans
    render_poses.append(SE3.Exp(torch.from_numpy(rotation_axis)))

render_poses = torch.stack(render_poses)
print(render_poses)

print(np.linspace(-180, 180, 40+1)[:-1])
