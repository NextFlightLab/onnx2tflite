import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from onnx2tflite.utils.op_registry import OPERATOR
from onnx2tflite.utils.definitions import Layout

LOG = logging.getLogger("recurrent_layers :")

_ACTIVATION_MAP = {
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "relu": tf.nn.relu,
}


def _get_activation(name, default_fn):
    if name is None:
        return default_fn
    act = _ACTIVATION_MAP.get(name.lower())
    if act is not None:
        return act
    return tf.keras.activations.get(name)


@OPERATOR.register_operator("GRU")
class TFGRU:
    def __init__(
        self, tensor_grap, node_weights, node_inputs, node_attribute, node_outputs, layout_dict, *args, **kwargs
    ) -> None:
        super().__init__()
        self.tensor_graph = tensor_grap
        self.node_weights = node_weights
        self.node_inputs = node_inputs
        self.initial_state_name = node_inputs[5] if len(node_inputs) > 5 and node_inputs[5] != "" else None

        self.hidden_size = int(node_attribute.get("hidden_size"))
        self.direction = node_attribute.get("direction", "forward")
        if self.direction not in ("forward", "reverse"):
            raise NotImplementedError(f"GRU direction {self.direction} is not supported yet.")
        self.linear_before_reset = bool(node_attribute.get("linear_before_reset", 0))

        weights = node_weights[node_inputs[1]]
        recurrent_weights = node_weights[node_inputs[2]]
        if weights.shape[0] != 1 or recurrent_weights.shape[0] != 1:
            raise NotImplementedError("Only single direction GRU is supported for now.")

        kernel = weights[0].transpose(1, 0).astype(np.float32)
        recurrent_kernel = recurrent_weights[0].transpose(1, 0).astype(np.float32)

        bias = None
        use_bias = False
        if len(node_inputs) > 3 and node_inputs[3] != "" and node_inputs[3] in node_weights:
            raw_bias = node_weights[node_inputs[3]]
            raw_bias = raw_bias[0] if raw_bias.ndim == 2 else raw_bias
            input_bias = raw_bias[: self.hidden_size * 3]
            recurrent_bias = raw_bias[self.hidden_size * 3 :]
            if self.linear_before_reset:
                bias = np.stack([input_bias, recurrent_bias], axis=0)
            else:
                bias = input_bias + recurrent_bias
            bias = bias.astype(np.float32)
            use_bias = True

        activations = node_attribute.get("activations", ["Sigmoid", "Tanh"])
        if not isinstance(activations, (list, tuple)) or len(activations) == 0:
            activations = ["Sigmoid", "Tanh"]
        gate_activation = _get_activation(activations[0] if len(activations) > 0 else None, tf.nn.sigmoid)
        candidate_activation = _get_activation(activations[1] if len(activations) > 1 else None, tf.nn.tanh)

        weights_list = [kernel, recurrent_kernel]
        if use_bias:
            weights_list.append(bias)

        layout_dict[node_outputs[0]] = Layout.Channel_None
        if len(node_outputs) > 1:
            layout_dict[node_outputs[1]] = Layout.Channel_None

        self.gru = keras.layers.GRU(
            units=self.hidden_size,
            activation=candidate_activation,
            recurrent_activation=gate_activation,
            return_sequences=True,
            return_state=True,
            go_backwards=self.direction == "reverse",
            time_major=True,
            reset_after=self.linear_before_reset,
            use_bias=use_bias,
            weights=weights_list,
        )

    def _prepare_initial_state(self, dtype):
        if self.initial_state_name is None:
            return None
        init_state = self.tensor_graph.get(self.initial_state_name)
        if init_state is None:
            init_state = self.node_weights.get(self.initial_state_name)
        if init_state is None:
            return None
        state = tf.cast(init_state, dtype)
        if len(state.shape) == 3:
            state = tf.transpose(state, perm=[1, 0, 2])[:, 0, :]
        return state

    def __call__(self, inputs):
        initial_state = self._prepare_initial_state(inputs.dtype)
        outputs = self.gru(inputs, initial_state=initial_state) if initial_state is not None else self.gru(inputs)
        sequence_out, final_state = outputs[0], outputs[1]
        sequence_out = tf.expand_dims(sequence_out, axis=1)
        final_state = tf.expand_dims(final_state, axis=0)
        return [sequence_out, final_state]
