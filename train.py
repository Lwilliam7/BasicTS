from basicts.launcher import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DLinear import DLinear, DLinearConfig


def main():
    # Step 1: Model config
    model_config = DLinearConfig(
        input_len=336,
        output_len=336,
    )

    # Step 2: Task config
    cfg = BasicTSForecastingConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="ETTh1",
        batch_size=32,
        learning_rate=1e-3,
        max_epochs=10,
        gpus=None,  # CPU. If you have CUDA working later, set "0"
    )

    # Step 3: Launch training
    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
