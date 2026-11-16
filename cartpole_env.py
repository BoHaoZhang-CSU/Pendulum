"""把 MuJoCo 旋转倒立摆封装成 Gymnasium 环境。"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "cartpole.xml"


class CartPoleEnv(gym.Env):
    """用中心电机控制旋转倒立摆。"""

    def __init__(self):
        super().__init__()

        # 1. 加载机器人结构，并创建仿真状态。
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        # 2. 找到两个关节和中心电机。
        arm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_yaw")
        pendulum_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "passive_hinge"
        )
        self.motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "center_motor"
        )

        if min(arm_id, pendulum_id, self.motor_id) < 0:
            raise ValueError("模型缺少关节或中心电机，请检查 XML 中的名称")

        # 3. 记录角度和角速度在数组里的下标。
        self.arm_pos = self.model.jnt_qposadr[arm_id]
        self.arm_vel = self.model.jnt_dofadr[arm_id]
        self.pendulum_pos = self.model.jnt_qposadr[pendulum_id]
        self.pendulum_vel = self.model.jnt_dofadr[pendulum_id]

        # 4. PPO 输出一个动作，范围是 -1 到 1。
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # 5. 告诉 PPO：每次观察到 6 个数字。
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        # PPO 每做一次决策，MuJoCo 连续运行 10 个物理小步。
        self.frame_skip = 10

        # 一个回合最多运行 10 秒仿真时间。
        self.max_episode_time = 10.0

    def _get_observation(self):
        """读取当前状态，整理成 PPO 能接收的一组数字。"""

        arm_angle = self.data.qpos[self.arm_pos]
        arm_speed = self.data.qvel[self.arm_vel]

        pendulum_angle = self.data.qpos[self.pendulum_pos]
        pendulum_speed = self.data.qvel[self.pendulum_vel]

        return np.array(
            [
                np.sin(arm_angle),
                np.cos(arm_angle),
                arm_speed,
                np.sin(pendulum_angle),
                np.cos(pendulum_angle),
                pendulum_speed,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        """开始新回合，把倒立摆恢复到下垂状态。"""

        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 给摆杆一点点随机初始偏角，避免每次完全一样。
        self.data.qpos[self.pendulum_pos] = self.np_random.uniform(-0.02, 0.02)

        # 根据新状态更新位置等派生数据。
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """执行 PPO 给出的动作，然后返回新的状态和奖励。"""

        # PPO 输出 -1～1，映射到 XML 中电机允许的力矩范围。
        action_value = float(np.clip(action[0], -1.0, 1.0))
        low, high = self.model.actuator_ctrlrange[self.motor_id]

        torque = low + (action_value + 1.0) * 0.5 * (high - low)
        self.data.ctrl[self.motor_id] = torque

        # 同一个力矩保持 10 个 MuJoCo 物理小步。
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()

        pendulum_angle = self.data.qpos[self.pendulum_pos]
        arm_speed = self.data.qvel[self.arm_vel]
        pendulum_speed = self.data.qvel[self.pendulum_vel]

        # 模型中摆杆向下时角度约为 0，向上时约为 π。
        # cos(angle) 向下时为 1，向上时为 -1。
        upright_reward = (1.0 - np.cos(pendulum_angle)) / 2.0

        # 适当惩罚高速旋转和过大的控制动作。
        speed_penalty = 0.001 * arm_speed**2
        action_penalty = 0.001 * action_value**2

        # 在直立附近，鼓励摆杆速度也接近零。
        balance_penalty = 0.001 * upright_reward * pendulum_speed**2

        reward = float(
            upright_reward - speed_penalty - action_penalty - balance_penalty
        )

        # 不因为摆杆落下就提前结束，给它时间继续尝试起摆。
        terminated = False

        # 到达回合时长后结束本回合。
        truncated = self.data.time >= self.max_episode_time

        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """此环境没有单独创建窗口，因此暂时不需要清理窗口资源。"""
        pass
