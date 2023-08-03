import torch
from torch import nn
from typing import Dict
from pytorch_forecasting.models import BaseModel


class FullyConnectedModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__()

        # input layer
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        return self.sequential(x)


# test that network works as intended
network = FullyConnectedModule(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)
x = torch.rand(20, 5)
network(x).shape



# class FullyConnectedModel(BaseModel):
#     def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, **kwargs):
#         # saves arguments in signature to `.hparams` attribute, mandatory call - do not skip this
#         self.save_hyperparameters()
#         # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
#         super().__init__(**kwargs)
#         self.network = FullyConnectedModule(
#             input_size=self.hparams.input_size,
#             output_size=self.hparams.output_size,
#             hidden_size=self.hparams.hidden_size,
#             n_hidden_layers=self.hparams.n_hidden_layers,
#         )

#     def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         # x is a batch generated based on the TimeSeriesDataset
#         network_input = x["encoder_cont"].squeeze(-1)
#         prediction = self.network(network_input)

#         # rescale predictions into target space
#         prediction = self.transform_output(prediction, target_scale=x["target_scale"])

#         # We need to return a dictionary that at least contains the prediction
#         # The parameter can be directly forwarded from the input.
#         # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
#         return self.to_network_output(prediction=prediction)