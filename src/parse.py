"""ONNX -> HAR (full precision).

The MiDaS ONNX is opset 10, whose Resize is always "asymmetric" (what onnxruntime computes).
The Hailo parser treats a missing coordinate_transformation_mode as the opset-11 default
"half_pixel", so the parsed decoder upsamples at different positions and the output drifts by
several pixels (ONNX vs HAR FP a1 ~83%). By default we convert to opset 11 with an explicit
"asymmetric" mode first, which the parser maps to resize_bilinear_pixels_mode=disabled.
"""
import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, version_converter

from hailo_sdk_client import ClientRunner


def fix_resize_mode(onnx_path, out_path, input_shape):
    """Write an opset-11 copy with explicit asymmetric Resize and check it matches the original."""
    model = onnx.load(onnx_path)
    fixed = version_converter.convert_version(model, 11)

    n_resize = 0
    for node in fixed.graph.node:
        if node.op_type == 'Resize':
            keep = [a for a in node.attribute if a.name != 'coordinate_transformation_mode']
            del node.attribute[:]
            node.attribute.extend(keep + [helper.make_attribute('coordinate_transformation_mode', 'asymmetric')])
            n_resize += 1

    onnx.checker.check_model(fixed)
    onnx.save(fixed, out_path)

    x = np.random.RandomState(0).rand(*input_shape).astype(np.float32)
    input_name = model.graph.input[0].name
    ref = ort.InferenceSession(onnx_path).run(None, {input_name: x})[0]
    out = ort.InferenceSession(out_path).run(None, {input_name: x})[0]
    max_diff = np.abs(ref - out).max()
    print(f"Resize fix: {n_resize} nodes -> asymmetric, max |original - fixed| = {max_diff:.3e}, saved {out_path}")
    assert max_diff < 1e-3, "converted model differs from the original"


def read_io(onnx_path):
    """Graph input/output tensor names and static input shapes from the ONNX file."""
    graph = onnx.load(onnx_path).graph
    initializers = {i.name for i in graph.initializer}  # older exporters list weights as inputs too
    inputs = [i for i in graph.input if i.name not in initializers]

    # None marks a dynamic dimension
    input_shapes = {i.name: [d.dim_value if d.HasField('dim_value') else None for d in i.type.tensor_type.shape.dim]
                    for i in inputs}
    return [i.name for i in inputs], [o.name for o in graph.output], input_shapes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='MIDAS_model/model-small.onnx')
    parser.add_argument('--har-out', default='Midas_hailo_model_normalize.har')
    parser.add_argument('--no-fix-resize', action='store_true', help='parse the original ONNX as-is (reproduces the half_pixel mismatch)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Tensor names survive the opset conversion, so read them from the original file
    input_names, output_names, input_shapes = read_io(args.onnx)
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError(f"expected a single-input/single-output ONNX, got inputs {input_names} / outputs {output_names}")
    start_node, end_node = input_names[0], output_names[0]
    input_shape = input_shapes[start_node]
    if None in input_shape:
        raise ValueError(f"input '{start_node}' shape is dynamic ({input_shape}); export the ONNX with a static shape")
    print(f"start_node={start_node} end_node={end_node} input_shape={input_shape}")

    onnx_path = args.onnx
    if not args.no_fix_resize:
        onnx_path = os.path.splitext(args.har_out)[0] + '_opset11_asym.onnx'
        fix_resize_mode(args.onnx, onnx_path, input_shape)

    runner = ClientRunner()
    runner.translate_onnx_model(
        onnx_path,
        'model-small.onnx',  # net name -> HAR layer prefix "model-small_onnx/", keep stable for model scripts
        start_node_names=[start_node],
        end_node_names=[end_node],
        net_input_shapes={start_node: input_shape},
    )

    # Normalization (Sub/Div) is already inside the ONNX graph, so no model script is needed here
    runner.optimize_full_precision()
    runner.save_har(args.har_out)
    print(f"Saved {args.har_out}")


if __name__ == '__main__':
    main()
