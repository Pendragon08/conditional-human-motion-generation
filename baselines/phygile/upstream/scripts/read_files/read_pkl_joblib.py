import pickle
import numpy as np
from scipy.sparse import csc_matrix
import joblib
def print_value_details(value, depth=1):
    """打印值的类型、形状和长度"""
    if isinstance(value, (np.ndarray, list, tuple)):
        print(f"{'  ' * depth}Type: {type(value)}, Shape: {np.shape(value)}, Length: {len(value)}")
    elif isinstance(value, csc_matrix):
        dense_array = value.toarray()  # 转换为稠密数组
        print(f"{'  ' * depth}Type: csc_matrix, Shape: {np.shape(value)}, Length: {len(dense_array)}")
    elif isinstance(value, str):
        print(f"{'  ' * depth}Type: 字符串, Length: {len(value)}")
        print(f"{'  ' * depth}Value: {value[:50]}...")  # 只打印前50个字符
    elif isinstance(value, dict):
        print(f"{'  ' * depth}Type: 字典, Length: {len(value)}")
        print(f"{'  ' * depth}Keys: {list(value.keys())[:5]}")  # 打印前5个 key
        print_dict_details(value, depth + 1)  # 递归打印嵌套字典
    else:
        print(f"{'  ' * depth}Type: {type(value)}, Value: {value}")
        
        
def print_dict_details(d, depth=1):
    """递归打印字典的类型、长度、和一些键的详情"""
    if isinstance(d, dict):
        print(f"{'  ' * depth}Type: 字典, Length: {len(d)}")
        print(f"{'  ' * depth}Keys: {list(d.keys())}")  # 打印前5个 key
        # 递归打印嵌套字典
        for key, value in d.items():
            print(f"{'  ' * depth}Key: {key}")
            print_value_details(value, depth + 1)  # 打印值的类型和详情
            
# 指定.pkl文件的路径
# pkl_file_path = r"/home/PJLAB/baojiacheng/code/g1/H2O/data/h1/test.pkl"
# pkl_file_path = r"/home/PJLAB/baojiacheng/code/g1/H2O/data/g1/test.pkl"
# pkl_file_path = r"/ssd/humanoid/human2humanoid/data/g1/single_walk_contact.pkl"
# pkl_file_path = '/ssd/humanoid/human2humanoid/data/g1/liftbox_hand_23dof.pkl'
pkl_file_path = '/ssd/humanoid/dataset/AMASS_retarget/CMU/0837_35_30_poses_phc.pkl'
# pkl_file_path = "/ssd/bjc/PBHC/example/motion_data/Bruce_Lee_pose.pkl"
# 读取并打印pkl文件
with open(pkl_file_path, 'rb') as file:
    # data = pickle.load(file)  # 加载.pkl文件
    data = joblib.load(file)
    print("数据加载成功！")
    # 打印 shapedirs 内容

    # 如果数据是字典，逐个打印 key 和值的类型/shape/长度
    if isinstance(data, dict):
        print("数据的 Keys 和详情如下：")
        print(data.keys())
        for key, value in data.items():
            # print(f"\nKey: {key}")
            if isinstance(value, (np.ndarray, list, tuple)):
                # 处理数组、列表、元组
                try:
                    print(f"Shape: {np.shape(value)}")
                    print(f"Length: {len(value)}")
                except Exception:
                    print("无法获取 Shape 或 Length")
            elif isinstance(value, csc_matrix):
                dense_array = value.toarray()  # 转换为稠密数组
                print(f"Type: csc_matrix, Shape: {np.shape(value)}")
                print(f"Length: {len(dense_array)}")
            elif isinstance(value, str):
                # 处理字符串
                print(f"Type: 字符串, Length: {len(value)}")
                print(f"Value: {value[:50]}...")  # 只打印前50个字符
            elif isinstance(data, dict):
                print("数据的 Keys 和详情如下：")
                print(len(data.items()))
                for i, (key, value) in enumerate(data.items()):
                    # print(f"\nKey: {key}")
                    # print_value_details(value, depth=2)  # 打印字典项的详细信息
                    # break
                    # print(value["root_trans_offset"][0])
                    print(value["smpl_joints"].shape)
                    print(value["root_rot"].shape)
                    print(value["contact_mask"].shape)
                    #打印特定信息：
                    # if value['flag_bigloss']:
                    #     print(f"\nKey: {key}，index: {i}")
                    #     print(f"length: {value['dof'].shape[0]},flag: {value['flag_bigloss']}")
            else:
                # 处理其他类型
                print(f"Type: {type(value)}, Value: {value}")
            break
    else:
        print(f"数据不是字典类型，而是 {type(data)}。内容如下：")
        print(data)
        # print(data['0-INF_JumpingJack_S3_01_poses']['pose_aa'][0])


    
'''
每个key里的value都是一个字典
Key: 0-CMU_02_01_poses
    Type: 字典, Length: 6
    Keys: ['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'fps']
      Type: 字典, Length: 6
      Keys: ['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'fps', 'flag_bigloss']
      Key: root_trans_offset
        Type: <class 'numpy.ndarray'>, Shape: (86, 3), Length: 86
      Key: pose_aa
        Type: <class 'numpy.ndarray'>, Shape: (86, 32, 3), Length: 86
      Key: dof
        Type: <class 'numpy.ndarray'>, Shape: (86, 29), Length: 86
      Key: root_rot
        Type: <class 'numpy.ndarray'>, Shape: (86, 4), Length: 86
      Key: fps
        Type: <class 'int'>, Value: 30
      Key: flag_bigloss
        Type: <class 'bool'>, Value: False
'''