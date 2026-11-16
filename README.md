# 旋转倒立摆

使用 MuJoCo 模拟旋转倒立摆，中心电机驱动水平臂，摆杆关节被动转动。

## 文件组织

- `models/cartpole.xml`：倒立摆模型。
- `cartpole_env.py`：CPU 训练环境。
- `train_cartpole.py`：Stable-Baselines3 PPO 训练入口。
- `show_cartpole_ppo.py`：加载训练模型并打开窗口，修改文件中的 `MODEL_PATH` 选择 checkpoint。
- `show_cartpole.py`：查看未经训练控制的模型。
- `run_cartpole_pid.py`：手工控制示例。
- `cartpole_env_gpu.py`：正在学习中的 GPU 并行仿真示例，不是完整的 PPO 环境。
- `outputs/cartpole_ppo`：已有模型、checkpoint 和 TensorBoard 日志。

## 运行

在 VS Code 中打开 `D:\workspace\python\Pendulum`，选择安装有依赖的 Python 解释器。例如原来已经配置好的 `D:\workspace\python\pico-bot\.venv-mjx\Scripts\python.exe`。它可以暂时复用；如需完全独立，创建本项目的 `.venv` 并安装 `requirements.txt` 中的依赖。

```powershell
python show_cartpole_ppo.py
python train_cartpole.py
python -m tensorboard.main --logdir outputs/cartpole_ppo
```

GPU 示例还需兼容本机 CUDA 的 `warp-lang` 和 `mujoco-warp`；GPU 训练没有在此示例中实现。若要使用 GPU 版 PyTorch，请按 PyTorch 官方说明安装适合本机 CUDA 的版本。
