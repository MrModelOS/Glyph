#!/usr/bin/env bash
# Build & run the NeuMoE/Static-MoE/Dense study on the MX450 (sm_75).
#
# Usage:  ./run.sh [steps] [eval_every] [seq_len] [n_train]
#   steps      total training steps per config (default 500)
#   eval_every validation interval (default 50)
#   seq_len    padded corpus length (default 128)
#   n_train    training sequences (default 2048)
set -euo pipefail

STEPS=${1:-500}
EVAL=${2:-50}
SLEN=${3:-128}
NTRAIN=${4:-2048}

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="$REPO/build/neumoe"
mkdir -p "$WORK"

echo "== corpus generation =="
python3 "$HERE/gen_corpus.py" "$SLEN" "$NTRAIN"
mkdir -p "$WORK/data"
mv -f train.bin val_prose.bin val_code.bin val_qa.bin "$WORK/data/"

for net in neumoe static dense; do
    echo "== $net =="
    "$REPO/build/nsc" "$HERE/$net.ns" --cuda --runtime > "$WORK/${net}_driver.cu"
    nvcc -O2 -std=c++11 -arch=sm_75 -I "$REPO/include" \
        "$WORK/${net}_driver.cu" "$HERE/host.cpp" -o "$WORK/${net}_driver"
    mkdir -p "$WORK/out/$net"
    "$WORK/${net}_driver" "$WORK/data" "$STEPS" "$WORK/out/$net" "$EVAL"
done

echo
echo "== summary =="
for net in neumoe static dense; do
    echo "--- $net ---"
    tail -1 "$WORK/out/$net/metrics.csv"
done