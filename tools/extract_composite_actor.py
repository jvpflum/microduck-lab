#!/usr/bin/env python3
"""Extract one standard 61D/14D MLP actor embedded in a DuckLab composite ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


ACTOR_TENSORS = (
    "obs_normalizer._mean",
    "onnx::Div_24",
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "mlp.6.weight",
    "mlp.6.bias",
)


def extract(source: Path, output: Path, prefix: str) -> None:
    model = onnx.load(source)
    initializers = {item.name: item for item in model.graph.initializer}
    missing = [f"{prefix}{name}" for name in ACTOR_TENSORS if f"{prefix}{name}" not in initializers]
    if missing:
        raise SystemExit(f"Composite is missing actor tensors for prefix {prefix!r}: {missing}")

    def name(suffix: str) -> str:
        return suffix

    def source_name(suffix: str) -> str:
        return f"{prefix}{suffix}"

    obs = "obs"
    normalized = "normalized_obs"
    nodes = [
        helper.make_node("Sub", [obs, name("obs_normalizer._mean")], ["centered"]),
        helper.make_node("Div", ["centered", name("onnx::Div_24")], [normalized]),
    ]
    current = normalized
    for index, layer in enumerate((0, 2, 4, 6)):
        output_name = "actions" if layer == 6 else f"hidden_{layer}"
        nodes.append(
            helper.make_node(
                "Gemm",
                [current, name(f"mlp.{layer}.weight"), name(f"mlp.{layer}.bias")],
                [output_name],
                name=f"ExtractedMLP{layer}",
                transB=1,
            )
        )
        if layer != 6:
            activated = f"elu_{layer}"
            nodes.append(helper.make_node("Elu", [output_name], [activated]))
            current = activated

    extracted = helper.make_model(
        helper.make_graph(
            nodes,
            "ducklab_extracted_actor",
            [helper.make_tensor_value_info(obs, TensorProto.FLOAT, [1, 61])],
            [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
            initializer=[
                onnx.helper.make_tensor(
                    suffix,
                    initializers[source_name(suffix)].data_type,
                    initializers[source_name(suffix)].dims,
                    initializers[source_name(suffix)].raw_data,
                    raw=True,
                )
                for suffix in ACTOR_TENSORS
            ],
        ),
        opset_imports=list(model.opset_import),
        producer_name="ducklab-composite-actor-extractor",
        ir_version=model.ir_version,
    )
    helper.set_model_props(
        extracted,
        {
            "source_composite": str(source.resolve()),
            "source_initializer_prefix": prefix,
            "ducklab_extracted_standard_actor": "true",
        },
    )
    onnx.checker.check_model(extracted)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(extracted, output)
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("composite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    extract(args.composite, args.output, args.prefix)


if __name__ == "__main__":
    main()
