"""旋转倒立摆：中心电机先摆起被动摆杆，再维持竖直。"""
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(__file__).resolve().parent / "models" / "cartpole.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    arm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "motor_yaw")
    pendulum = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "passive_hinge")
    motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "center_motor")
    if min(arm, pendulum, motor) < 0:
        raise ValueError("模型缺少中心关节、摆杆关节或中心电机")
    arm_pos, arm_vel = model.jnt_qposadr[arm], model.jnt_dofadr[arm]
    pend_pos, pend_vel = model.jnt_qposadr[pendulum], model.jnt_dofadr[pendulum]
    mujoco.mj_forward(model, data)
    balancing = False
    last_print = -1.0
    print("启动：中心电机驱动水平臂，摆杆关节自由转动。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.6]
        viewer.cam.distance = 5.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        while viewer.is_running():
            started = time.perf_counter()
            theta = data.qpos[pend_pos]
            theta_speed = data.qvel[pend_vel]
            arm_angle = data.qpos[arm_pos]
            arm_speed = data.qvel[arm_vel]
            # 摆杆默认朝下，因此角度 pi 是朝上；将误差折算到 -pi 到 pi。
            error = math.atan2(math.sin(theta - math.pi), math.cos(theta - math.pi))

            if not balancing and abs(error) < 0.6805 and abs(theta_speed) < 7.618:
                balancing = True
                print("进入平衡阶段", flush=True)
            elif balancing and abs(error) > 0.8:
                balancing = False
                print("偏离过大，重新起摆", flush=True)

            if balancing:
                # 直立附近的状态反馈：同时考虑摆杆与水平臂的位置和速度。
                torque = -(-1.41421356 * arm_angle + 28.06775293 * error
                           - 2.77856917 * arm_speed + 5.77991279 * theta_speed)
            else:
                # 摆起阶段通过电机给系统注入能量；前 0.2 秒轻推以打破静止。
                energy = (0.5 * 0.5 * 0.65**2 * theta_speed**2
                          + 0.5 * 9.81 * 0.65 * (1 - math.cos(theta)))
                target_energy = 9.81 * 0.65
                torque = (-11.48596 * (target_energy - energy) * theta_speed * math.cos(theta)
                          - 0.5319 * arm_speed - 0.2383 * arm_angle)
                if data.time < 0.2:
                    torque += 1.0

            low, high = model.actuator_ctrlrange[motor]
            data.ctrl[motor] = max(low, min(high, torque))
            mujoco.mj_step(model, data)
            viewer.sync()
            if data.time - last_print >= 0.5:
                phase = "平衡" if balancing else "起摆"
                print(f"{phase}  时间={data.time:.1f}s  偏离竖直={math.degrees(error):.1f}°  "
                      f"电机力矩={data.ctrl[motor]:.2f} N·m", flush=True)
                last_print = data.time
                
                
                
            delay = model.opt.timestep - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
