# General imports used throughout the tutorial
import tensorflow as tf
from IPython.display import SVG
import os

# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

# For Mini PCIe modules or Hailo-8R devices, use 'hailo8r'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

onnx_model_name = "model-small.onnx"
onnx_path = "MIDAS_model/model-small.onnx"


runner = ClientRunner()
hn, npz = runner.translate_onnx_model(
    onnx_path,
    onnx_model_name,
    start_node_names=["0"],
    end_node_names=["797"],
    net_input_shapes={"0": [1, 3, 256, 256]},
)

names = "Midas"
hailo_model_har_name = f"{names}_hailo_model_normalize.har"

# Load the model script to ClientRunner so it will be considered on optimization
# Midas has normalize layer in model
runner.optimize_full_precision()

runner.save_har(hailo_model_har_name)

