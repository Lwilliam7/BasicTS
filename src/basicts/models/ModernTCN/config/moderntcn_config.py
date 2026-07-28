from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class ModernTCNConfig(BasicTSModelConfig):
    """Config class for ModernTCN forecasting."""

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of input/output features."})
    hidden_size: int = field(default=64, metadata={"help": "Hidden channel size."})
    num_layers: int = field(default=3, metadata={"help": "Number of temporal convolution blocks."})
    kernel_size: int = field(default=7, metadata={"help": "Depthwise temporal convolution kernel size."})
    expansion: int = field(default=2, metadata={"help": "Expansion factor for pointwise feed-forward layers."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to apply RevIN around the forecaster."})
    affine: bool = field(default=False, metadata={"help": "Whether RevIN uses affine parameters."})
    subtract_last: bool = field(default=False, metadata={"help": "Whether RevIN subtracts the last value."})
