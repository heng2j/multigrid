from gymnasium import spaces
from ray.rllib.models.tf.complex_input_net import (
    ComplexInputNetwork as TFComplexInputNetwork
)
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.complex_input_net import (
    ComplexInputNetwork as TorchComplexInputNetwork
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()



class TFModel(TFModelV2):
    """
    Basic tensorflow model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs):
        """
        See ``TFModelV2.__init__()``.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = TFComplexInputNetwork(
            obs_space, action_space, num_outputs, model_config, name)
        self.forward = self.model.forward
        self.value_function = self.model.value_function


class TorchModel(TorchModelV2, nn.Module):
    """
    Basic torch model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs):
        """
        See ``TorchModelV2.__init__()``.
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.model = TorchComplexInputNetwork(
            obs_space, action_space, num_outputs, model_config, name)
        self.forward = self.model.forward
        self.value_function = self.model.value_function


class TorchLSTMModel(TorchModelV2, nn.Module):
    """
    Torch LSTM model to use with RLlib.

    Processes observations with a ``ComplexInputNetwork`` and then passes
    the output through an LSTM layer.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs):
        """
        See ``TorchModelV2.__init__()``.
        """
        nn.Module.__init__(self)
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        # Base
        self.base_model = TorchComplexInputNetwork(
            obs_space,
            action_space,
            None,
            model_config,
            f'{name}_base',
        )

        # LSTM
        self.lstm = nn.LSTM(
            self.base_model.post_fc_stack.num_outputs,
            model_config.get('lstm_cell_size', 256),
            batch_first=True,
        )

        # Action & Value
        self.action_model = nn.Linear(self.lstm.hidden_size, num_outputs)
        self.value_model = nn.Linear(self.lstm.hidden_size, 1)

        # Current LSTM output
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        # Base
        x, _ = self.base_model(input_dict, state, seq_lens)

        # LSTM
        x = add_time_dimension(
            x,
            seq_lens=seq_lens,
            framework='torch',
            time_major=False,
        )
        h, c = state[0].unsqueeze(0), state[1].unsqueeze(0)
        x, [h, c] = self.lstm(x, [h, c])

        # Out
        self._features = x.reshape(-1, self.lstm.hidden_size)
        logits = self.action_model(self._features)
        return logits, [h.squeeze(0), c.squeeze(0)]

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_model(self._features).flatten()

    def get_initial_state(self):
        return [torch.zeros(self.lstm.hidden_size), torch.zeros(self.lstm.hidden_size)]






from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np

class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)


        self.num_team_members = 1 # FIXME with custom policy spec model_config["custom_model_config"]["teams"]

        # Base of the model
        self.model = TorchComplexInputNetwork(
            obs_space, action_space, num_outputs, model_config, name)

        # Central VF maps (obs, team_obs, team_act) -> vf_pred
        # Calculate input size based on observation size, number of team members and action space
        obs_size = np.prod(obs_space.shape)
        act_size = action_space.n 
        input_size = obs_size * (self.num_team_members + 1) + act_size * self.num_team_members

        # input_size = 6 + 6 + 2  # my agent's obs + team member's obs + team member's actions
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, team_obs, team_actions):
        input_ = torch.cat(
            [
                obs,
                team_obs,
                torch.nn.functional.one_hot(team_actions.squeeze(1).long(), self.action_space.n).float(),

            ],
            1,
        )

        return torch.reshape(self.central_vf(input_), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used

