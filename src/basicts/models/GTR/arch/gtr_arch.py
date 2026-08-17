from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from ..config.gtr_config import GTRConfig


class GlobalTemporalRetriever(nn.Module):
    """Global Temporal Retriever block from GTR.

    The block maps a retrieved cycle segment, stacks it with the local input,
    then applies a 2D convolution across local/global rows and time.
    """

    def __init__(
        self,
        input_len: int,
        num_features: int,
        period_len: int = 24,
        individual: bool = False,
        aggregate_channels: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.num_features = num_features
        self.period_len = period_len
        self.individual = individual
        self.aggregate_channels = aggregate_channels

        self.query_projection = nn.Linear(input_len, input_len)
        kernel_width = 1 + 2 * (period_len // 2)
        padding_width = period_len // 2
        if individual:
            self.channel_convs = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(2, kernel_width),
                        stride=1,
                        padding=(0, padding_width),
                        padding_mode="zeros",
                        bias=False,
                    )
                    for _ in range(num_features)
                ]
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(2, kernel_width),
                stride=1,
                padding=(0, padding_width),
                padding_mode="zeros",
                bias=False,
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor, retrieved_query: torch.Tensor) -> torch.Tensor:
        """Run GTR fusion.

        Args:
            inputs: Local input with shape [batch_size, num_features, input_len].
            retrieved_query: Retrieved global segment with the same shape as inputs.

        Returns:
            Tensor with shape [batch_size, num_features, input_len].
        """

        batch_size, num_features, input_len = inputs.shape
        global_query = self.query_projection(retrieved_query)

        if self.aggregate_channels:
            weight = F.softmax(global_query, dim=1)
            global_query = torch.sum(global_query * weight, dim=1, keepdim=True)
            global_query = global_query.repeat(1, num_features, 1)

        fused = torch.stack([inputs, global_query], dim=2)
        if self.individual:
            conv_outs = [
                conv(fused[:, i, :, :].unsqueeze(1))
                for i, conv in enumerate(self.channel_convs)
            ]
            conv_out = torch.cat(conv_outs, dim=1).squeeze(2)
        else:
            fused = fused.reshape(batch_size * num_features, 1, 2, input_len)
            conv_out = self.conv2d(fused).reshape(batch_size, num_features, input_len)

        return self.dropout(conv_out)


class GTRForForecasting(nn.Module):
    """
    Paper: Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval
    Link: https://openreview.net/forum?id=QUJBPSfyui
    Official Code: https://github.com/macovaseas/GTR
    Venue: ICLR 2026
    Task: Long-term and short-term time series forecasting
    """

    def __init__(self, config: GTRConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.cycle_len = config.cycle_len
        self.use_revin = config.use_revin
        self.timestamp_feature_index = config.timestamp_feature_index
        self.use_timestamp_cycle_index = config.use_timestamp_cycle_index

        self.global_cycle = nn.Parameter(torch.zeros(self.cycle_len, self.num_features))
        self.gtr = GlobalTemporalRetriever(
            input_len=config.input_len,
            num_features=config.num_features,
            period_len=config.period_len,
            individual=config.individual,
            aggregate_channels=config.aggregate_channels,
            dropout=config.gtr_dropout,
        )
        self.input_projection = nn.Linear(config.input_len, config.hidden_size)
        self.backbone = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
        )
        self.output_projection = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.output_len),
        )

    def _cycle_index_from_timestamps(self, inputs_timestamps: torch.Tensor) -> torch.Tensor:
        first_timestamp = inputs_timestamps[:, 0, self.timestamp_feature_index]
        if first_timestamp.dtype.is_floating_point:
            cycle_index = torch.round(first_timestamp * self.cycle_len).long()
        else:
            cycle_index = first_timestamp.long()
        return cycle_index.remainder(self.cycle_len)

    def _default_cycle_index(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(inputs.shape[0], device=inputs.device, dtype=torch.long)

    def _retrieve_global_segment(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor],
        cycle_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cycle_index is None:
            if self.use_timestamp_cycle_index and inputs_timestamps is not None:
                cycle_index = self._cycle_index_from_timestamps(inputs_timestamps)
            else:
                cycle_index = self._default_cycle_index(inputs)
        cycle_index = cycle_index.to(device=inputs.device, dtype=torch.long).view(-1)
        gather_index = (
            cycle_index.view(-1, 1)
            + torch.arange(self.input_len, device=inputs.device).view(1, -1)
        ).remainder(self.cycle_len)
        return self.global_cycle[gather_index].permute(0, 2, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
        cycle_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forecast future values.

        Args:
            inputs: Input tensor with shape [batch_size, input_len, num_features].
            inputs_timestamps: Optional timestamp features with shape
                [batch_size, input_len, num_timestamp_features].
            cycle_index: Optional absolute cycle index for the first input step.

        Returns:
            Prediction with shape [batch_size, output_len, num_features].
        """

        if self.use_revin:
            seq_mean = torch.mean(inputs, dim=1, keepdim=True)
            seq_var = torch.var(inputs, dim=1, keepdim=True) + 1e-5
            inputs = (inputs - seq_mean) / torch.sqrt(seq_var)

        channel_first = inputs.permute(0, 2, 1)
        retrieved_query = self._retrieve_global_segment(inputs, inputs_timestamps, cycle_index)
        global_information = self.gtr(channel_first, retrieved_query)

        projected = self.input_projection(channel_first + global_information)
        hidden = self.backbone(projected)
        prediction = self.output_projection(hidden + projected).permute(0, 2, 1)

        if self.use_revin:
            prediction = prediction * torch.sqrt(seq_var) + seq_mean

        return prediction


GTR = GTRForForecasting
