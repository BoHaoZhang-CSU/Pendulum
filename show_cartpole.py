"""在 MuJoCo 窗口中显示水平转臂和被动摆杆。"""
import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).resolve().parent / "models" / "cartpole.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("模型已加载：中心电机驱动水平连杆，被动关节连接竖直摆杆。")
    print("关闭 MuJoCo 窗口即可结束程序。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 在窗口中显示世界坐标系的 XYZ 方向标记。
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        viewer.cam.lookat[:] = [0, 0, 1.6]  # 相机看向的世界坐标 x、y、z
        viewer.cam.distance = 5.5
        while viewer.is_running():
            started = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            delay = model.opt.timestep - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
