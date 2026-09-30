#!/usr/bin/env bash
# MiDaS v2.1 small -> Hailo INT8: parse -> quantize -> evaluate
#
# 사용법: [환경변수=값 ...] ./run_pipeline.sh [stage ...]      (도움말: ./run_pipeline.sh -h)
#
# Stages (순서대로 실행, 생략하면 parse quantize evaluate 전체)
#   parse      src/parse.py     ONNX -> HAR(FP). Resize를 asymmetric으로 보정한 뒤 파싱
#   quantize   src/quantize.py  RUN_DIR의 파싱 HAR -> 양자화 HAR (calibration: DA-2K 전체)
#   evaluate   src/evaluate.py  ONNX FP32 출력을 기준으로 HAR 각 단계(native/fp_optimized/quantized) 비교
#
# 예시
#   ./run_pipeline.sh                                  # 전체 실행 (새 runs/<timestamp>/)
#   ./run_pipeline.sh parse evaluate                   # 파싱 손실만 확인 (양자화 없이 빠름)
#   NUM_IMAGES=100 ./run_pipeline.sh parse evaluate    # 100장만으로 빠르게 확인
#   FIX_RESIZE=0 ./run_pipeline.sh parse evaluate      # Resize 보정 없이 파싱 (기존 a1 ~83% 재현)
#   RUN_DIR=runs/20260929_173500 ./run_pipeline.sh quantize evaluate
#                                                      # 기존 run의 파싱 HAR로 양자화부터 다시
#   ALLS=my_script.alls ./run_pipeline.sh              # 직접 만든 model script로 양자화
#   HAR=runs/xxx/Midas_quantize_model.har ./run_pipeline.sh evaluate
#                                                      # 이미 있는 HAR만 평가
#   HAR=... CONTEXTS="fp_optimized quantized" ./run_pipeline.sh evaluate
#                                                      # 원하는 단계만 비교
#
# 환경변수
#   DATA_ROOT     DA-2K/, MIDAS_model/ 가 있는 폴더 (기본: 이 repo)
#   RUN_DIR       결과 폴더 (기본: runs/<timestamp>). 기존 폴더를 주면 그 안의 HAR를 이어서 사용
#   PY            hailo_sdk_client + onnxruntime 있는 python (기본: hailo-dataflow conda env)
#   FIX_RESIZE    1 = Resize asymmetric 보정 후 파싱 (기본), 0 = 원본 ONNX 그대로 파싱
#   ALLS          양자화 model script 파일 (기본: src/quantize.py 의 DEFAULT_ALLS)
#   HAR           평가할 HAR (기본: RUN_DIR에 양자화 HAR 있으면 그것, 없으면 파싱 HAR)
#   CONTEXTS      평가할 HAR 단계 (기본: 파싱 HAR = "native fp_optimized", 양자화 HAR = 셋 다)
#   NUM_IMAGES    평가에 앞에서부터 N장만 사용 (기본: 전체 1033장). 정렬 순서라 한 카테고리에 몰릴 수 있음
#   CUDA_VISIBLE_DEVICES  사용할 GPU (기본: 0)
#
# 결과 (RUN_DIR/)
#   Midas_hailo_model_normalize.har            파싱 HAR (FP)
#   Midas_hailo_model_normalize_opset11_asym.onnx  Resize 보정한 ONNX (FIX_RESIZE=1일 때)
#   Midas_quantize_model.har                   양자화 HAR
#   eval/<기준>_vs_<대상>.csv                    이미지별 지표 (abs_rel, sq_rel, rms, log_rms, a1-a3, rel<0.1%)
#   logs/<stage>.log                           단계별 전체 로그. 평가 요약 표는 logs/evaluate.log 끝부분
#
# 평가 표 읽는 법 (기준 -> 대상)
#   onnx -> native               파싱 손실 (ONNX -> HAR)
#   native -> fp_optimized       FP 최적화(equalization 등) 손실
#   fp_optimized -> quantized    순수 양자화 손실
#   onnx -> quantized            전체 손실
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
    exit 0
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$(cd "${DATA_ROOT:-$REPO}" && pwd)"
RUN_DIR="${RUN_DIR:-$REPO/runs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR/logs"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/hailo-dataflow/bin/python}"
IMAGES_ROOT="$DATA_ROOT/DA-2K/DA-2K/images"
ONNX="$DATA_ROOT/MIDAS_model/model-small.onnx"
PARSED_HAR="$RUN_DIR/Midas_hailo_model_normalize.har"
QUANT_HAR="$RUN_DIR/Midas_quantize_model.har"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# run <stage> <script> [args...]: run from RUN_DIR so SDK logs land there, tee to logs/<stage>.log
run() {
    local stage="$1"; shift
    echo "=== [$stage] $(date '+%F %T')"
    (cd "$RUN_DIR" && "$PY" "$REPO/src/$1" "${@:2}") 2>&1 | tee "$RUN_DIR/logs/$stage.log"
    echo "=== [$stage] done $(date '+%F %T')"
}

stage_parse() {
    local fix=()
    [[ "${FIX_RESIZE:-1}" == 0 ]] && fix=(--no-fix-resize)
    run parse parse.py --onnx "$ONNX" --har-out "$PARSED_HAR" "${fix[@]}"
}

stage_quantize() {
    run quantize quantize.py --har-in "$PARSED_HAR" --har-out "$QUANT_HAR" \
        --images-root "$IMAGES_ROOT" ${ALLS:+--alls "$ALLS"}
}

stage_evaluate() {
    local har="${HAR:-}" contexts="${CONTEXTS:-}"
    if [[ -z "$har" ]]; then
        if [[ -f "$QUANT_HAR" ]]; then har="$QUANT_HAR"; else har="$PARSED_HAR"; fi
    fi
    if [[ -z "$contexts" ]]; then
        # a parsed-only HAR has no quantized weights
        if [[ "$har" == "$PARSED_HAR" ]]; then contexts="native fp_optimized"; else contexts="native fp_optimized quantized"; fi
    fi
    run evaluate evaluate.py --onnx "$ONNX" --har "$har" --contexts $contexts \
        --images-root "$IMAGES_ROOT" --out-dir "$RUN_DIR/eval" ${NUM_IMAGES:+--num-images "$NUM_IMAGES"}
}

stages=("$@")
[[ ${#stages[@]} -eq 0 ]] && stages=(parse quantize evaluate)

echo "DATA_ROOT=$DATA_ROOT"; echo "RUN_DIR=$RUN_DIR"; echo "stages: ${stages[*]}"
for s in "${stages[@]}"; do
    case "$s" in
        parse|quantize|evaluate) "stage_$s" ;;
        *) echo "unknown stage: $s (parse | quantize | evaluate)"; exit 1 ;;
    esac
done
