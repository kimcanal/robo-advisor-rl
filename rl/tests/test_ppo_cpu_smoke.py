"""CPU smoke: PPO.learn must start without a GPU.

Colab CPU + the PyPI CUDA wheel dies with exit 139 inside PPO() (no traceback).
Training here is intentionally tiny and pinned to device=cpu.
"""
from pathlib import Path

from stable_baselines3 import PPO

from rl.env.portfolio_env import PortfolioEnv
from rl.pipeline import prepare_env_inputs
from rl.vecnorm import make_train_vecnorm

ROOT = Path(__file__).resolve().parents[2]


def test_requirements_pin_cpu_torch_not_pypi_cuda_wheel():
    text = (ROOT / "rl" / "requirements.txt").read_text()
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in text
    assert "torch==2.14.0+cpu" in text
    # Bare PyPI pin resolves to the CUDA 13 wheel on Linux.
    assert "torch==2.14.0\n" not in text


def test_ppo_learn_starts_on_cpu():
    _, returns, rsi_df, macd_df = prepare_env_inputs(
        ["A0", "A1", "A2"],
        start="2019-01-01",
        end="2020-06-01",
        use_dummy=True,
        verbose=False,
    )

    def env_fn():
        return PortfolioEnv(returns, rsi_df, macd_df, window=20, reward_type="simple")

    venv = make_train_vecnorm(env_fn)
    model = PPO(
        "MlpPolicy",
        venv,
        verbose=0,
        seed=0,
        device="cpu",
        n_steps=32,
        batch_size=32,
        n_epochs=1,
    )
    # device="cpu" must not consult CUDA; Colab also needs `pip uninstall triton` (torch/_dynamo SIGSEGV).
    assert model.device.type == "cpu"
    model.learn(total_timesteps=32)
    assert model.num_timesteps >= 32
