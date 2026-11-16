"""在 GPU 上创建多个旋转倒立摆仿真环境。"""

from pathlib import Path

import mujoco
import mujoco_warp as mjw
import warp as wp


MODEL_PATH = Path(__file__).resolve().parent / "models" / "cartpole.xml"

@wp.kernel
def set_motor_torque(
    ctrl: wp.array2d(dtype=wp.float32), # pyright: ignore[reportInvalidTypeForm]
    torque: float,
):
    # 当前正在处理第几个环境。
    env_id = wp.tid()

    # 每个环境的第 0 个电机都设置成相同的力矩。
    ctrl[env_id, 0] = torque


class CartPoleGpuEnv:
    """一批并行运行的旋转倒立摆。"""

    def __init__(self, n_envs=128):
        self.n_envs = n_envs

        # 初始化 Warp，并选择第 0 块 NVIDIA 显卡。
        wp.init()
        self.device = wp.get_device("cuda:0")

        # 先用普通 MuJoCo 读取 XML，得到模型结构。
        self.cpu_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

        # 在指定 GPU 上创建 Warp 模型和一批仿真状态。
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.make_data(
                self.cpu_model,
                nworld=self.n_envs,
            )

    def step_physics(self):
        """让所有环境各推进一个物理小步。"""
        with wp.ScopedDevice(self.device):
            mjw.step(self.model, self.data)

        # 等待 GPU 完成，方便当前学习和检查结果。
        wp.synchronize()
    def set_torque(self, torque: float):
        """把所有环境的中心电机都设置成相同的力矩。"""
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=set_motor_torque,
                dim=self.n_envs,
                inputs=[self.data.ctrl, torque],
                device=self.device,
            )
        

if __name__ == "__main__":
    env = CartPoleGpuEnv(n_envs=128)

    print("GPU 设备：", env.device)
    print("环境数量：", env.n_envs)
    print("qpos 形状：", env.data.qpos.shape)
    print("ctrl 形状：", env.data.ctrl.shape)

    # 给 128 个中心电机都施加 1.0 的控制力矩。
    env.set_torque(1.0)

    # 连续推进 100 个物理小步。
    for _ in range(100):
        env.step_physics()

    # 为了打印，把 GPU 上的数据复制到 CPU。
    qpos = env.data.qpos.numpy()
    print("前 3 个环境的关节角度：")
    print(qpos[:3])
