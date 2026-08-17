from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class GTRConfig(BasicTSModelConfig):
    """Config class for Global Temporal Retrieval forecasting."""

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of variables/features."})
    cycle_len: int = field(default=24, metadata={"help": "Length of the global temporal cycle."})
    period_len: int = field(default=24, metadata={"help": "Dominant local period used by the retriever convolution."})
    hidden_size: int = field(default=512, metadata={"help": "Hidden size of the MLP forecasting backbone."})
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate in the forecasting head."})
    gtr_dropout: float = field(default=0.1, metadata={"help": "Dropout rate inside the GTR module."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use reversible instance normalization."})
    individual: bool = field(default=False, metadata={"help": "Whether to use channel-specific retriever convolutions."})
    aggregate_channels: bool = field(default=False, metadata={"help": "Whether to aggregate retrieved queries across channels."})
    timestamp_feature_index: int = field(default=0, metadata={"help": "Timestamp feature used to derive cycle indices."})
    use_timestamp_cycle_index: bool = field(default=True, metadata={"help": "Whether to derive cycle indices from timestamps."})
