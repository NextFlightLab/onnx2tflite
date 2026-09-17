import os
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx import TensorProto, helper, numpy_helper

from onnx2tflite import onnx_converter


class OperatorRegressionTest(unittest.TestCase):
    def test_gather_constant_indices_are_int32(self):
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        indices = numpy_helper.from_array(np.array(1, dtype=np.int64), "indices")
        gather = helper.make_node("Gather", ["x", "indices"], ["output"], axis=1)
        graph = helper.make_graph([gather], "gather_regression", [x_info], [output_info], [indices])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 8
        onnx.checker.check_model(model)

        x = np.arange(8, dtype=np.float32).reshape(1, 2, 4)

        with tempfile.TemporaryDirectory() as model_root:
            onnx_path = os.path.join(model_root, "gather.onnx")
            onnx.save(model, onnx_path)
            result = onnx_converter(
                onnx_model_path=onnx_path,
                need_simplify=False,
                output_path=model_root,
                target_formats=["tflite"],
            )

            interpreter = tf.lite.Interpreter(model_path=result["tflite"])
            interpreter.allocate_tensors()
            gather_op = next(op for op in interpreter._get_ops_details() if op["op_name"] == "GATHER")
            positions_index = gather_op["inputs"][1]
            tensor_details = {detail["index"]: detail for detail in interpreter.get_tensor_details()}
            self.assertEqual(tensor_details[positions_index]["dtype"], np.int32)

            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
            interpreter.invoke()
            actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

        np.testing.assert_array_equal(actual, x[:, 1, :])

    def test_multi_axis_squeeze_and_layer_normalization(self):
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 4])
        obs_info = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 2])
        output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 6])

        initializers = [
            numpy_helper.from_array(np.array([0, 1], dtype=np.int64), "axes"),
            numpy_helper.from_array(np.linspace(0.8, 1.3, 6, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.linspace(-0.2, 0.2, 6, dtype=np.float32), "bias"),
        ]
        nodes = [
            helper.make_node("Squeeze", ["x", "axes"], ["squeezed"]),
            helper.make_node("Concat", ["obs", "squeezed"], ["features"], axis=-1),
            helper.make_node(
                "LayerNormalization",
                ["features", "scale", "bias"],
                ["output"],
                axis=-1,
                epsilon=1e-5,
            ),
        ]
        graph = helper.make_graph(nodes, "operator_regression", [x_info, obs_info], [output_info], initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 8
        onnx.checker.check_model(model)

        x = np.array([[[[1.0, -2.0, 3.0, 0.5]]]], dtype=np.float32)
        obs = np.array([[0.25, -0.75]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as model_root:
            onnx_path = os.path.join(model_root, "operators.onnx")
            onnx.save(model, onnx_path)
            result = onnx_converter(
                onnx_model_path=onnx_path,
                need_simplify=False,
                output_path=model_root,
                target_formats=["tflite"],
            )

            expected = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x, "obs": obs})[0]
            interpreter = tf.lite.Interpreter(model_path=result["tflite"])
            interpreter.allocate_tensors()
            for detail in interpreter.get_input_details():
                value = x if len(detail["shape"]) == 4 else obs
                interpreter.set_tensor(detail["index"], value)
            interpreter.invoke()
            actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
