import numpy as np
import math

def euler_to_quaternion(roll, pitch, yaw, degrees=False):
    """
    将XYZ欧拉角转换为WXYZ四元数
    
    参数:
        roll (float): 绕X轴旋转角度
        pitch (float): 绕Y轴旋转角度
        yaw (float): 绕Z轴旋转角度
        degrees (bool): 如果为True，输入角度为度数；否则为弧度
    
    返回:
        tuple: (w, x, y, z) 四元数
    """
    # 如果输入是度数，转换为弧度
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    
    # 计算半角
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    # 计算四元数 (WXYZ格式)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (w, x, y, z)


def euler_to_quaternion_numpy(roll, pitch, yaw, degrees=False):
    """
    使用NumPy将XYZ欧拉角转换为WXYZ四元数
    
    参数:
        roll: 绕X轴旋转角度 (可以是数组)
        pitch: 绕Y轴旋转角度 (可以是数组)
        yaw: 绕Z轴旋转角度 (可以是数组)
        degrees (bool): 如果为True，输入角度为度数；否则为弧度
    
    返回:
        numpy.ndarray: shape为(..., 4)的数组，最后一维为[w, x, y, z]
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.stack([w, x, y, z], axis=-1)


# 示例用法
if __name__ == "__main__":
    # 示例1: 单个角度转换（度数）
    roll, pitch, yaw = -90, 0, 0
    quat = euler_to_quaternion(roll, pitch, yaw, degrees=True)
    print(f"欧拉角 (度): Roll={roll}, Pitch={pitch}, Yaw={yaw}")
    print(f"四元数 (WXYZ): W={quat[0]:.4f}, X={quat[1]:.4f}, Y={quat[2]:.4f}, Z={quat[3]:.4f}")
    print(f"如果角度不对，好好反思一下需不需要角度取反")
    
    # # 示例2: 单个角度转换（弧度）
    # roll_rad = math.pi / 4
    # pitch_rad = math.pi / 6
    # yaw_rad = math.pi / 3
    # quat2 = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad, degrees=False)
    # print(f"欧拉角 (弧度): Roll={roll_rad:.4f}, Pitch={pitch_rad:.4f}, Yaw={yaw_rad:.4f}")
    # print(f"四元数 (WXYZ): W={quat2[0]:.4f}, X={quat2[1]:.4f}, Y={quat2[2]:.4f}, Z={quat2[3]:.4f}")
    # print()
