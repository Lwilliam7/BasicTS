import torch
from torch import nn

from basicts.modules.norm import RevIN

from ..config.moderntcn_config import ModernTCNConfig


class ModernTCNBlock(nn.Module):
    """Depthwise temporal convolution block with pointwise channel mixing."""

    def __init__(self, hidden_size: int, kernel_size: int, expansion: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ModernTCN kernel_size must be odd to preserve sequence length")
        expanded_size = hidden_size * expansion
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,
        )
        self.norm = nn.BatchNorm1d(hidden_size)
        self.ffn = nn.Sequential(
            nn.Conv1d(hidden_size, expanded_size, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(expanded_size, hidden_size, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.depthwise(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        return hidden_states + residual


class ModernTCN(nn.Module):
    """
    Modern temporal convolution backbone for long-term forecasting.

    Inputs use BasicTS forecasting shape [batch_size, input_len, num_features].
    The backbone returns hidden states with shape [batch_size, hidden_size, input_len].
    """

    def __init__(self, config: ModernTCNConfig):
        super().__init__()
        self.input_projection = nn.Conv1d(config.num_features, config.hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(
                    hidden_size=config.hidden_size,
                    kernel_size=config.kernel_size,
                    expansion=config.expansion,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.BatchNorm1d(config.hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_projection(inputs.transpose(1, 2))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output_norm(hidden_states)


class ModernTCNForForecasting(nn.Module):
    """ModernTCN forecaster returning [batch_size, output_len, num_features]."""

    def __init__(self, config: ModernTCNConfig):
        super().__init__()
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.backbone = ModernTCN(config)
        self.temporal_head = nn.Linear(config.input_len, config.output_len)
        self.feature_head = nn.Conv1d(config.hidden_size, config.num_features, kernel_size=1)
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features,
                affine=config.affine,
                subtract_last=config.subtract_last,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_revin:
            inputs = self.revin(inputs, "norm")
        hidden_states = self.backbone(inputs)
        hidden_states = self.temporal_head(hidden_states)
        prediction = self.feature_head(hidden_states).transpose(1, 2)
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        return prediction
