# python scripts/read_files/read_npz.py
import numpy as np

# 指定.npz文件的路径
npz_file_path = '/ssd/humanoid/dataset/mocap_ball_walk_kick/mocap_ball_walk_kick_1_122.npz'

try:
    # 读取.npz文件
    data = np.load(npz_file_path)

    print("数据加载成功！")

    # 打印数据中的所有键
    print("数据中的键如下：")
    for key in data.files:
        print(key)

    # 逐个打印每个键对应的值的信息
    print("\n每个键对应的值的详情如下:")
    for key in data.files:
        value = data[key]
        print(f"\nKey: {key}")
        if isinstance(value, (np.ndarray, list, tuple)):
            # 处理数组、列表、元组
            try:
                print(f"Shape: {np.shape(value)}")
                print(f"Length: {len(value)}")
            except Exception:
                print(value)
                print("无法获取 Shape 或 Length")
            # 检查是否存在非有限值（NaN/Inf）
            try:
                arr = np.asarray(value)
                finite_mask = np.isfinite(arr)
                if not finite_mask.all():
                    bad_count = (~finite_mask).sum()
                    print(f"[WARN] {key} contains {bad_count} non-finite values (NaN/Inf)")
                    # 可选：打印前若干个问题位置和值
                    bad_indices = np.argwhere(~finite_mask)
                    preview = bad_indices[:10]
                    print(f"[WARN] First {len(preview)} bad indices: {preview}")
            except Exception as e:
                print(f"[WARN] Failed to check finiteness for key {key}: {e}")
        elif isinstance(value, str):
            # 处理字符串
            print(f"Type: 字符串, Length: {len(value)}")
            print(f"Value: {value[:50]}...")  # 只打印前50个字符
        elif isinstance(value, dict):
            # 处理嵌套字典
            print(f"Type: 字典, Length: {len(value)}")
            print(f"Keys: {list(value.keys())[:5]}")  # 打印前5个 key
        else:
            # 处理其他类型
            print(f"Type: {type(value)}, Value: {value}")
    # print(data['timestamps_us'][:100])

except FileNotFoundError:
    print(f"错误：文件 '{npz_file_path}' 未找到。")
except Exception as e:
    print(f"发生错误：{e}")