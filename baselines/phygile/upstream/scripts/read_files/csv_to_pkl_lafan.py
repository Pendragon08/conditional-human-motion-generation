import pandas as pd
import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import joblib

# 假设CSV文件名为"data.csv"，并且每行的格式如你所述
csv_file = "/Users/bjc/Desktop/RL_robotics/data/LAFAN1_Retargeting_Dataset/g1/dance1_subject2.csv"

# 读取CSV文件
data = pd.read_csv(csv_file, header=None)

# 假设每行数据的结构是固定的，第一部分是root_joint，后面是各个关节的值
# 提取root_joint的数据，假设前7列是root_joint(XYZQXQYQZQW)
root_trans = data.iloc[:, :3].values  # XYZ
root_rot = data.iloc[:, 3:7].values  # QXQYQZQW

# 提取其他关节的数据
dof_pos = data.iloc[:, 7:].values  # 假设从第8列开始是各个关节的值

device = torch.device("cpu")

g1_rotation_axis = torch.tensor([[
    [0, 1, 0], # left_hip_pitch_joint
    [1, 0, 0], # left_hip_roll_joint
    [0, 0, 1], # left_hip_yaw_joint

    [0, 1, 0], # left_knee_joint
    [0, 1, 0], # left_ankle_pitch_joint
    [1, 0, 0], # left_ankle_roll_joint

    [0, 1, 0], # right_hip_pitch_joint
    [1, 0, 0], # right_hip_roll_joint
    [0, 0, 1], # right_hip_yaw_joint

    [0, 1, 0], # right_knee_joint
    [0, 1, 0], # right_ankle_pitch_joint
    [1, 0, 0], # right_ankle_roll_joint

    [0, 0, 1], # 12 waist_yaw_joint
    [1, 0, 0], # 13 waist_roll_joint fix
    [0, 1, 0], # 14 waist_pitch_joint fix弯腰

    [0, 1, 0], # left_shoulder_pitch_joint
    [1, 0, 0], # left_shoulder_roll_joint
    [0, 0, 1], # left_shoulder_yaw_joint

    [0, 1, 0], # left_elbow_joint
    [1, 0, 0], # 19 left_wrist_roll_joint
    [0, 1, 0], # 20 left_wrist_pitch_joint
    [0, 0, 1], # 21 left_wrist_yaw_joint

    [0, 1, 0], # right_shoulder_pitch_joint
    [1, 0, 0], # right_shoulder_roll_joint
    [0, 0, 1], # right_shoulder_yaw_joint

    [0, 1, 0], # right_elbow_joint
    [1, 0, 0], # right_wrist_roll_joint
    [0, 1, 0], # right_wrist_pitch_joint
    [0, 0, 1]  # right_wrist_yaw_joint
]]).to(device)
    
root_trans_tensor = torch.tensor(root_trans, dtype=torch.float32).to(device)
root_rot_tensor = torch.tensor(root_rot, dtype=torch.float32).to(device)
dof_pos_tensor = torch.tensor(dof_pos, dtype=torch.float32).to(device)

# 假设N是数据的行数
print(root_trans_tensor.shape)
N = root_trans_tensor.shape[0]
dof_pos_tensor = dof_pos_tensor.view(1, N, 29, 1)

#should roll 加
# dof_pos_tensor[:,:, [16],:] += 0.25
# dof_pos_tensor[:,:, [23],:] -= 0.25

root_vec = torch.from_numpy(sRot.from_quat(root_rot_tensor).as_rotvec()).to(device)
# 构造pose_aa_g1_new
# 假设g1_rotation_axis是一个已知的张量，这里用随机值代替
# import ipdb;ipdb.set_trace()
pose_aa_g1_new = torch.cat([root_vec[None, :, None], g1_rotation_axis * dof_pos_tensor, torch.zeros((1, N, 3, 3)).to(device)], axis=2)

#to 23 dof
# 获取第二维中除了这些索引的所有索引
indices_to_remove = [19, 20, 21, 26, 27, 28]
all_indices = np.arange(29)
indices_to_keep = np.setdiff1d(all_indices, indices_to_remove)
all_indices_aa = np.arange(33)
indices_to_keep_aa = np.setdiff1d(all_indices_aa, [x + 1 for x in indices_to_remove] )
print(f"indices_to_keep_aa:{indices_to_keep_aa}")
dof_pos_new = dof_pos_tensor[:,:, indices_to_keep,:]
pose_aa_g1_new = pose_aa_g1_new[:,:, indices_to_keep_aa,:]

selected_start = 168 #150
selected_end = 610 #turn 870
# 构造data_dump
data_dump = {}
data_key = csv_file.split('/')[-1].replace('.csv', '')
data_dump[data_key] = {
    "root_trans_offset": root_trans_tensor.cpu().detach().numpy()[selected_start:selected_end, ...].astype(np.float32),
    "pose_aa": pose_aa_g1_new.squeeze().cpu().detach().numpy()[selected_start:selected_end, ...].astype(np.float32),
    "dof": dof_pos_new.squeeze().detach().cpu().numpy()[selected_start:selected_end, ...].astype(np.float32),
    "root_rot": root_rot_tensor.detach().cpu().numpy()[selected_start:selected_end, ...].astype(np.float32),
    "fps": 30  # 将 fps 显式地改为 float 类型
}
print(f"data_dump[data_key][root_trans_offset]:{data_dump[data_key]['root_trans_offset'].shape}")
print(data_dump[data_key]['root_trans_offset'].dtype)
print(f"data_dump[data_key][pose_aa]:{data_dump[data_key]['pose_aa'].shape}")
print(data_dump[data_key]['pose_aa'].dtype)
print(f"data_dump[data_key][dof]:{data_dump[data_key]['dof'].shape}")
print(data_dump[data_key]['dof'].dtype)
print(f"data_dump[data_key][root_rot]:{data_dump[data_key]['root_rot'].shape}")
print(data_dump[data_key]['root_rot'].dtype)


joblib.dump(data_dump, f"{data_key}_lafan_23dof.pkl")

print(f"数据已保存为{data_key}_lafan_23dof.pkl")