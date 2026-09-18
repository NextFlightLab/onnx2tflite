#!/usr/bin/env python3

"""Install a TFLite model and its operator resolver into mc_nn_control."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
import tempfile


MODULE_RELATIVE_PATH = Path("src/modules/mc_nn_control")
RESOLVER_RELATIVE_PATH = Path(
    "src/lib/tensorflow_lite_micro/tflite_micro/"
    "tensorflow/lite/micro/micro_mutable_op_resolver.h"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update PX4 mc_nn_control model data and registered TFLite Micro operators."
    )
    parser.add_argument("--px4-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-info", type=Path, required=True)
    return parser.parse_args()


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_model_info(path: Path, model: bytes) -> list[str]:
    text = path.read_text(encoding="utf-8")
    digest_match = re.search(r"^SHA256:\s*([0-9a-fA-F]{64})\s*$", text, re.MULTILINE)
    if digest_match is None:
        raise RuntimeError(f"SHA256 entry is missing from {path}")

    actual_digest = hashlib.sha256(model).hexdigest()
    if digest_match.group(1).lower() != actual_digest:
        raise RuntimeError("model_info.txt does not describe the supplied TFLite model")

    operators_match = re.search(r"^Operators:\s*$\n(?P<body>(?:\s+-\s+[^\n]+\n?)+)", text, re.MULTILINE)
    if operators_match is None:
        raise RuntimeError(f"operator list is missing from {path}")

    operators = re.findall(r"^\s+-\s+([^\s]+)\s*$", operators_match.group("body"), re.MULTILINE)
    operators = sorted(set(operators))
    if not operators:
        raise RuntimeError("the TFLite model has no operators")
    return operators


def resolver_methods(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    methods = set(re.findall(r"\bTfLiteStatus\s+(Add[A-Za-z0-9_]+)\s*\(", text))
    methods.difference_update({"AddBuiltin", "AddCustom"})

    by_normalized_name: dict[str, str] = {}
    for method in sorted(methods):
        key = normalized_name(method[3:])
        if key in by_normalized_name:
            raise RuntimeError(f"ambiguous resolver methods: {by_normalized_name[key]} and {method}")
        by_normalized_name[key] = method
    return by_normalized_name


def map_operator_methods(operators: list[str], resolver_header: Path) -> list[str]:
    methods = resolver_methods(resolver_header)
    registrations: list[str] = []
    unsupported: list[str] = []

    for operator in operators:
        method = methods.get(normalized_name(operator))
        if method is None:
            unsupported.append(operator)
        else:
            registrations.append(method)

    if unsupported:
        names = ", ".join(unsupported)
        raise RuntimeError(f"PX4 TFLite Micro has no automatic registration method for: {names}")
    return registrations


def generate_control_net_cpp(model: bytes) -> str:
    rows = []
    for offset in range(0, len(model), 12):
        values = ", ".join(f"0x{value:02x}" for value in model[offset : offset + 12])
        rows.append(f"  {values},")

    array_body = "\n".join(rows)
    return (
        '#include <cstdint>\n'
        '#include "control_net.hpp"\n\n'
        'alignas(16) const unsigned char control_net_tflite[] = {\n'
        f"{array_body}\n"
        '};\n'
    )


def generate_control_net_hpp(model_size: int) -> str:
    return (
        '#pragma once\n\n'
        '#include <cstdint>\n\n'
        f'constexpr unsigned int control_net_tflite_size = {model_size};\n'
        'extern const unsigned char control_net_tflite[];\n'
    )


def update_register_ops(source: str, operators: list[str], registrations: list[str]) -> str:
    registration_lines = "\n".join(
        f"\tTF_LITE_ENSURE_STATUS(op_resolver.{method}());" for method in registrations
    )
    replacement = (
        '// Generated from the deployed TFLite model by export_deploy.sh.\n'
        f'using NNControlOpResolver = tflite::MicroMutableOpResolver<{len(operators)}>;\n\n'
        'TfLiteStatus RegisterOps(NNControlOpResolver &op_resolver)\n'
        '{\n'
        f'{registration_lines}\n'
        '\treturn kTfLiteOk;\n'
        '}'
    )
    pattern = re.compile(
        r"(?:^//[^\n]*\n)*^using\s+NNControlOpResolver\s*=\s*"
        r"tflite::MicroMutableOpResolver<\d+>;\s*\n\s*"
        r"TfLiteStatus\s+RegisterOps\(NNControlOpResolver\s*&op_resolver\)\s*\n"
        r"\{.*?^\}",
        re.MULTILINE | re.DOTALL,
    )
    updated, count = pattern.subn(replacement, source)
    if count != 1:
        raise RuntimeError(f"expected one RegisterOps block in mc_nn_control.cpp, found {count}")
    return updated


def prepare_atomic_write(path: Path, content: str) -> tuple[Path, Path] | None:
    current = path.read_text(encoding="utf-8")
    if current == content:
        return None

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, stat.S_IMODE(path.stat().st_mode))
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        descriptor = -1
        with stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path, path


def main() -> int:
    args = parse_args()
    px4_root = args.px4_root.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    model_info_path = args.model_info.resolve(strict=True)
    module_path = px4_root / MODULE_RELATIVE_PATH
    resolver_header = px4_root / RESOLVER_RELATIVE_PATH
    control_cpp = module_path / "control_net.cpp"
    control_hpp = module_path / "control_net.hpp"
    controller_cpp = module_path / "mc_nn_control.cpp"

    for required_path in (resolver_header, control_cpp, control_hpp, controller_cpp):
        if not required_path.is_file():
            raise RuntimeError(f"required PX4 file not found: {required_path}")

    model = model_path.read_bytes()
    if len(model) < 8 or model[4:8] != b"TFL3":
        raise RuntimeError(f"not a valid TFLite FlatBuffer: {model_path}")

    operators = parse_model_info(model_info_path, model)
    registrations = map_operator_methods(operators, resolver_header)
    controller_source = controller_cpp.read_text(encoding="utf-8")

    generated_files = {
        control_cpp: generate_control_net_cpp(model),
        control_hpp: generate_control_net_hpp(len(model)),
        controller_cpp: update_register_ops(controller_source, operators, registrations),
    }

    pending_writes: list[tuple[Path, Path]] = []
    try:
        for target, content in generated_files.items():
            prepared = prepare_atomic_write(target, content)
            if prepared is not None:
                pending_writes.append(prepared)

        for temporary_path, target in pending_writes:
            os.replace(temporary_path, target)
    finally:
        for temporary_path, _target in pending_writes:
            temporary_path.unlink(missing_ok=True)

    changed_paths = {target for _temporary, target in pending_writes}
    for target in generated_files:
        state = "updated" if target in changed_paths else "unchanged"
        print(f"  {state}: {target}")

    print(f"  model bytes: {len(model)}")
    print(f"  operator types: {len(operators)}")
    for operator, method in zip(operators, registrations):
        print(f"    {operator} -> {method}()")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
