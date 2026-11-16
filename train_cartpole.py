"""使用 Stable-Baselines3 的 PPO 训练旋转倒立摆。"""

from pathlib import Path
from datetime import datetime
from uuid import uuid4

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from cartpole_env import CartPoleEnv

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cartpole_ppo"


def main():
    # 每次运行使用独立目录；随机后缀也避免同一时刻启动时重名。
    run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    env = CartPoleEnv()

    # 先检查环境的输入输出是否符合 Stable-Baselines3 的要求。
    check_env(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        tensorboard_log=str(run_dir / "tensorboard"),
        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],  # Actor 的两层隐藏层
                "vf": [64, 64],  # Critic 的两层隐藏层
            },
            "activation_fn": torch.nn.Tanh,
        },
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_cartpole",
    )
    # 先用这组步数做首次实验，效果需要实际观察。
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    model.save(run_dir / "ppo_cartpole")
    env.close()

    print(f"训练完成，模型保存在：{run_dir / 'ppo_cartpole.zip'}")


if __name__ == "__main__":
    main()
