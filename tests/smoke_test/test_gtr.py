# pylint: disable=wrong-import-position

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../src/"))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.models.GTR import GTR, GTRConfig


def test_gtr_smoke_test():
    output_len = 64
    input_len = 64
    gtr_config = GTRConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=7,
        cycle_len=24,
        period_len=24,
        hidden_size=128,
        dropout=0.0,
    )

    BasicTSLauncher.launch_training(
        BasicTSForecastingConfig(
            model=GTR,
            dataset_name="ETTh1_mini",
            model_config=gtr_config,
            gpus=None,
            num_epochs=5,
            input_len=input_len,
            output_len=output_len,
            use_timestamps=True,
        )
    )


if __name__ == "__main__":
    test_gtr_smoke_test()
