


from unitree_dds_wrapper.idl import unitree_hg, unitree_go
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription
from unitree_dds_wrapper.utils.crc import crc32
import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=200)
import struct
import copy
import threading
import time


lowstate_topic = "rt/lowstate"
odom_topic = "rt/lf/odommodestate"


#define TOPIC_SPORT_STATE "rt/odommodestate"//high frequency
#define TOPIC_SPORT_LF_STATE "rt/lf/odommodestate"//low frequency

# unitree_dds_wrapper/python/unitree_dds_wrapper/utils/joystick.py
lowstate_subscriber = Subscription(unitree_hg.msg.dds_.LowState_, lowstate_topic)
odom_subscriber = Subscription(unitree_go.msg.dds_.SportModeState_, odom_topic)


def load_mujoco():
    import mujoco
    import mujoco.viewer
    humanoid_xml = "./src/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = 1/30  # 或根据你的数据频率设置
    
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    viewer.cam.lookat[:] = np.array([0,0,0.7])
    viewer.cam.distance = 3.0        
    viewer.cam.azimuth = 180         
    viewer.cam.elevation = -30                      # 负值表示从上往下看viewer
    return (mujoco,mj_model,mj_data,viewer)
    
def step_mujoco(mjc,q,pos,quat):
    
    mujoco,mj_model,mj_data,viewer = mjc
    mj_data.qpos[:3]=pos 
    mj_data.qpos[3:7] = quat
    mj_data.qpos[7:] = q
    mujoco.mj_forward(mj_model, mj_data)
    viewer.sync()
    ...

def decode_motor_state(motor_state):
    # motor_state is a list of MotorState_ objects
    q = []
    dq = []
    Len = 35
    for i in range(Len):
        q.append(motor_state[i].q)
        dq.append(motor_state[i].dq)
    return np.array(q, dtype=np.float32), np.array(dq, dtype=np.float32)

def decode_odom_state(odom_state):
    # Pelvis-based 
    
    
    # odom_state is a OdomState_ object
    pos = odom_state.position
    vel = odom_state.velocity
    return np.array(pos, dtype=np.float32), np.array(vel, dtype=np.float32)


mjc = load_mujoco()
while True:
    if lowstate_subscriber.msg is not None and odom_subscriber.msg is not None:
        q,dq = decode_motor_state(lowstate_subscriber.msg.motor_state)
        imu_state = lowstate_subscriber.msg.imu_state
        
        quat = imu_state.quaternion # WXYZ
        omega = imu_state.gyroscope
        
        # print(f"quat={np.array(quat)[[1,2,3,0]]}")  # WXYZ -> XYZW
        # print(f"omega={np.array(omega,dtype=np.float32)}")

        pos, vel = decode_odom_state(odom_subscriber.msg)

        # print(f"pos={pos}, vel={vel}")
        step_mujoco(mjc,q[:29],pos,quat)

    else:
        print("No msg")
        
        
    # time.sleep(0.003)
    time.sleep(0.0003)

