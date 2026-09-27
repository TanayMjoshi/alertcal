#!/usr/bin/env bash
# Download Server Machine Dataset files from NetManAIOps/OmniAnomaly (GitHub, branch master).
# Layout verified 16 Sep 2026: ServerMachineDataset/{train,test,test_label}/machine-X-Y.txt
#   train      38 comma-separated metrics per row, benign only
#   test       same columns, contains anomalies
#   test_label one 0/1 label per row of test
# Usage: scripts/get_smd.sh [machine ...]     default: machine-1-1 machine-1-2
#        SMD_DIR=/elsewhere scripts/get_smd.sh   to change the destination (default data/smd)
set -euo pipefail

BASE="https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${SMD_DIR:-$ROOT/data/smd}"

if [ "$#" -gt 0 ]; then MACHINES=("$@"); else MACHINES=(machine-1-1 machine-1-2); fi

for m in "${MACHINES[@]}"; do
  for split in train test test_label; do
    mkdir -p "$DEST/$split"
    f="$DEST/$split/$m.txt"
    if [ -s "$f" ]; then
      echo "have  $split/$m.txt"
      continue
    fi
    echo "fetch $split/$m.txt"
    curl -fsSL --retry 3 --retry-delay 2 -o "$f.part" "$BASE/$split/$m.txt"
    mv "$f.part" "$f"
  done
done

echo
for m in "${MACHINES[@]}"; do
  rows() { awk 'END { print NR }' "$1"; }   # counts a last line without trailing newline too
  tr_n=$(rows "$DEST/train/$m.txt"); te_n=$(rows "$DEST/test/$m.txt"); la_n=$(rows "$DEST/test_label/$m.txt")
  printf "%-12s train %6s rows | test %6s rows | labels %6s rows\n" "$m" "$tr_n" "$te_n" "$la_n"
  if [ "$te_n" -ne "$la_n" ]; then
    echo "row count mismatch between test and test_label for $m" >&2
    exit 1
  fi
done
echo "SMD files in $DEST"
