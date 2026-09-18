#!/usr/bin/env bash

set -Eeuo pipefail

# Required settings. Edit the defaults or override them with environment variables.
WEIGHTS="${WEIGHTS:-/home/robot/diff_rl/logs/diff_rl/shac/2026-09-17_21-59-20/exported/policy.onnx}"
OUTPATH="${OUTPATH:-/home/robot/diff_rl/logs/diff_rl/shac/2026-09-17_21-59-20/exported}"

# Optional settings.
# Leave DEPLOY_NAME empty to use the ONNX filename. For example, policy.onnx
# produces policy.tflite and the C symbols policy_tflite/policy_tflite_len.
DEPLOY_NAME="${DEPLOY_NAME:-}"
# Conda environment that contains this repository's conversion dependencies.
CONDA_ENV="${CONDA_ENV:-onnx2tflite}"
# Set to 0 only when the target is not PX4/TFLite Micro. The strict check rejects
# INT64 tensors and Flex operators that the embedded runtime cannot execute.
STRICT_TFLM_CHECK="${STRICT_TFLM_CHECK:-1}"
# Set to 0 to skip the detailed TensorFlow Lite graph report.
WRITE_ANALYZER_REPORT="${WRITE_ANALYZER_REPORT:-1}"
# Set to 1 to install the exported model and operator registrations into PX4.
UPDATE_PX4="${UPDATE_PX4:-1}"
# PX4 repository root. Set UPDATE_PX4=0 to export artifacts without changing PX4.
PX4_ROOT="${PX4_ROOT:-/home/robot/PX4-Autopilot}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR=""
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" && "${WORK_DIR}" == "${OUTPATH}"/.onnx2tflite-export.* ]]; then
        rm -rf -- "${WORK_DIR}"
    fi
}

trap cleanup EXIT
trap 'printf "Error: export failed at line %s.\n" "$LINENO" >&2' ERR


command -v realpath >/dev/null 2>&1 || die "realpath is required"
command -v xxd >/dev/null 2>&1 || die "xxd is required (install the xxd package)"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
[[ "${STRICT_TFLM_CHECK}" == 0 || "${STRICT_TFLM_CHECK}" == 1 ]] || die "STRICT_TFLM_CHECK must be 0 or 1"
[[ "${WRITE_ANALYZER_REPORT}" == 0 || "${WRITE_ANALYZER_REPORT}" == 1 ]] || die "WRITE_ANALYZER_REPORT must be 0 or 1"
[[ "${UPDATE_PX4}" == 0 || "${UPDATE_PX4}" == 1 ]] || die "UPDATE_PX4 must be 0 or 1"

WEIGHTS="$(realpath -e -- "${WEIGHTS}")"
mkdir -p -- "${OUTPATH}"
OUTPATH="$(realpath -e -- "${OUTPATH}")"

if [[ "${UPDATE_PX4}" == 1 ]]; then
    PX4_ROOT="$(realpath -e -- "${PX4_ROOT}")"
    [[ -d "${PX4_ROOT}/src/modules/mc_nn_control" ]] || \
        die "mc_nn_control module not found under PX4_ROOT: ${PX4_ROOT}"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]] && command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1; then
    PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
elif [[ -x "${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python" ]]; then
    PYTHON_CMD=("${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python")

elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python" ]]; then
    PYTHON_CMD=("${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python")
else
    die "activate the ${CONDA_ENV} Conda environment before running this script"
fi

if ! PYTHONNOUSERSITE=1 "${PYTHON_CMD[@]}" -c 'import onnx2tflite, tensorflow' >/dev/null 2>&1; then
    die "the selected Python environment cannot import onnx2tflite and tensorflow"
fi

MODEL_FILE="$(basename -- "${WEIGHTS}")"
SOURCE_STEM="${MODEL_FILE%%.*}"
MODEL_STEM="${DEPLOY_NAME:-${SOURCE_STEM}}"
[[ "${MODEL_STEM}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || \
    die "DEPLOY_NAME must be a valid C identifier (letters, digits, and underscores)"
TFLITE_FILE="${MODEL_STEM}.tflite"
WORK_DIR="$(mktemp -d "${OUTPATH}/.onnx2tflite-export.XXXXXX")"
TFLITE_PATH="${WORK_DIR}/${TFLITE_FILE}"

printf 'Converting %s\n' "${WEIGHTS}"
(
    cd -- "${SCRIPT_DIR}"
    PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=2 "${PYTHON_CMD[@]}" -m onnx2tflite \
        --weights "${WEIGHTS}" \
        --outpath "${WORK_DIR}" \
        --formats tflite
)

CONVERTED_TFLITE="${WORK_DIR}/${SOURCE_STEM}.tflite"
[[ -s "${CONVERTED_TFLITE}" ]] || die "converter did not create ${SOURCE_STEM}.tflite"
if [[ "${CONVERTED_TFLITE}" != "${TFLITE_PATH}" ]]; then
    mv -- "${CONVERTED_TFLITE}" "${TFLITE_PATH}"
fi
[[ -s "${TFLITE_PATH}" ]] || die "converter did not create ${TFLITE_FILE}"

printf 'Validating %s\n' "${TFLITE_FILE}"
MODEL_INFO="$(
    PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=2 "${PYTHON_CMD[@]}" - \
        "${TFLITE_PATH}" "${STRICT_TFLM_CHECK}" <<'PY'
import hashlib
import pathlib
import sys

import numpy as np
import tensorflow as tf

model_path = pathlib.Path(sys.argv[1])
strict_tflm_check = sys.argv[2] == "1"
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
int64_tensors = [item["name"] for item in tensor_details if item["dtype"] == np.int64]
if strict_tflm_check and int64_tensors:
    names = ", ".join(int64_tensors)
    raise RuntimeError(f"INT64 tensors are not supported by the PX4 TFLite Micro target: {names}")

ops = sorted({
    item["op_name"]
    for item in interpreter._get_ops_details()
    if item["op_name"] != "DELEGATE"
})
flex_ops = [name for name in ops if name.startswith("Flex")]
if strict_tflm_check and flex_ops:
    raise RuntimeError(f"Select TF operators are not supported by TFLite Micro: {', '.join(flex_ops)}")

def tensor_line(item):
    shape = "x".join(str(value) for value in item["shape"])
    return f"  - {item['name']}: shape={shape}, dtype={item['dtype'].__name__}"

digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
print(f"Model: {model_path.name}")
print(f"Size: {model_path.stat().st_size} bytes")
print(f"SHA256: {digest}")
print("Inputs:")
for detail in interpreter.get_input_details():
    print(tensor_line(detail))
print("Outputs:")
for detail in interpreter.get_output_details():
    print(tensor_line(detail))
print("Operators:")
for op in ops:
    print(f"  - {op}")
PY
)"
printf '%s\n' "${MODEL_INFO}" | tee "${WORK_DIR}/model_info.txt"

if [[ "${WRITE_ANALYZER_REPORT}" == 1 ]]; then
    printf 'Writing TensorFlow Lite analyzer report\n'
    PYTHONNOUSERSITE=1 TF_CPP_MIN_LOG_LEVEL=2 "${PYTHON_CMD[@]}" - "${TFLITE_PATH}" \
        > "${WORK_DIR}/model_analyzer.txt" <<'PY'
import sys

import tensorflow as tf

with open(sys.argv[1], "rb") as model_file:
    tf.lite.experimental.Analyzer.analyze(model_content=model_file.read())
PY
fi

(
    cd -- "${WORK_DIR}"
    xxd -i "${TFLITE_FILE}" > model_data.cc
)
[[ -s "${WORK_DIR}/model_data.cc" ]] || die "xxd did not create model_data.cc"

mv -f -- "${TFLITE_PATH}" "${OUTPATH}/${TFLITE_FILE}"
mv -f -- "${WORK_DIR}/model_data.cc" "${OUTPATH}/model_data.cc"
mv -f -- "${WORK_DIR}/model_info.txt" "${OUTPATH}/model_info.txt"
if [[ "${WRITE_ANALYZER_REPORT}" == 1 ]]; then
    mv -f -- "${WORK_DIR}/model_analyzer.txt" "${OUTPATH}/model_analyzer.txt"
fi

if [[ "${UPDATE_PX4}" == 1 ]]; then
    printf 'Updating PX4 mc_nn_control sources\n'
    PYTHONNOUSERSITE=1 "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/tools/update_px4_model.py" \
        --px4-root "${PX4_ROOT}" \
        --model "${OUTPATH}/${TFLITE_FILE}" \
        --model-info "${OUTPATH}/model_info.txt"
fi

printf '\nExport completed:\n'
printf '  TFLite:    %s\n' "${OUTPATH}/${TFLITE_FILE}"
printf '  C++ data:  %s\n' "${OUTPATH}/model_data.cc"
printf '  Model info: %s\n' "${OUTPATH}/model_info.txt"
if [[ "${WRITE_ANALYZER_REPORT}" == 1 ]]; then
    printf '  Analyzer:  %s\n' "${OUTPATH}/model_analyzer.txt"
fi
if [[ "${UPDATE_PX4}" == 1 ]]; then
    printf '  PX4 module: %s\n' "${PX4_ROOT}/src/modules/mc_nn_control"
    printf '  Next step: rebuild and restart PX4\n'
fi
printf '  SHA256:    %s\n' "$(sha256sum "${OUTPATH}/${TFLITE_FILE}" | cut -d ' ' -f 1)"
