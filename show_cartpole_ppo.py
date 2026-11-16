"""加载训练好的 PPO 模型，在 MuJoCo 窗口中观察它控制倒立摆。"""
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

from cartpole_env import CartPoleEnv


# 改成要观看的模型文件；也可以填写 checkpoints 文件夹里的 .zip 文件。
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "cartpole_ppo"
    / "20261115_180034_abbc7c08"
    / "checkpoints"
    / "ppo_cartpole_500000_steps.zip"
)


def main():
    env = CartPoleEnv()
    try:
        model = PPO.load(MODEL_PATH, device="cpu")
        observation, _ = env.reset(seed=42)
        episode_reward = 0.0

        print(f"正在观看模型：{MODEL_PATH}")
        print("关闭 MuJoCo 窗口结束；每个回合结束后自动重新开始。")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 1.6]
            viewer.cam.distance = 5.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

            while viewer.is_running():
                started = time.perf_counter()

                # deterministic=True 使用策略的确定性动作，便于比较 checkpoint。
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                viewer.sync()

                if terminated or truncated:
                    print(f"回合结束，总奖励：{episode_reward:.2f}", flush=True)
                    observation, _ = env.reset()
                    episode_reward = 0.0

                # 一个环境步包含 frame_skip 个物理小步，按仿真时间控制播放速度。
                step_time = env.model.opt.timestep * env.frame_skip
                remaining = step_time - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()
