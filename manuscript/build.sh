#!/usr/bin/env bash
# Build the manuscript (draft and Frontiers layouts) and the Supplementary Material from ../outputs,
# and run the checks on each. Needs Python with the packages in ../requirements.txt, Node.js with the
# "docx" package, and pandoc. Set PYTHON if your interpreter is not called python3.
set -euo pipefail
PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="$ROOT/outputs"
EX="$OUT/exhibits_v12"
export NODE_PATH="${NODE_PATH:-$(npm root -g)}"

cd "$HERE/main"
mkdir -p facts build build_frontiers
"$PY" v12_facts.py "$OUT" ./facts > facts/facts_log.txt 2>&1 || { tail -20 facts/facts_log.txt; exit 1; }
tail -1 facts/facts_log.txt
for layout in draft frontiers; do
  dir=build; [ "$layout" = frontiers ] && dir=build_frontiers
  "$PY" build_v12.py --layout "$layout" --template v12_template.md --facts facts/v12_facts.json \
    --refs v12_references.md --tables "$EX" \
    --fig1 "$EX/Figure1_HeartRate.png" --fig2 "$OUT/figures/Figure3_PSD_Curves.png" \
    --fig3 "$EX/Figure3_Resolution.png" --fig4 "$OUT/figures/Figure8_CrossValidation.png" \
    --out "$dir" > "$dir/build_log.txt" 2>&1 || { tail -20 "$dir/build_log.txt"; exit 1; }
done
"$PY" check_v12.py build/manuscript_v12_draft.docx v12_template.md facts/v12_facts.json v12_references.md \
  build/v12_supplement_numbers.csv "$EX" | tail -3
"$PY" check_v12.py build_frontiers/manuscript_v12_frontiers.docx v12_template.md facts/v12_facts.json \
  v12_references.md build_frontiers/v12_supplement_numbers.csv "$EX" | tail -3
cmp -s build/v12_rendered.md build_frontiers/v12_rendered.md && echo "the two layouts carry identical text" \
  || { echo "LAYOUT TEXT DIFFERS"; exit 1; }

cd "$HERE/supplement"
mkdir -p build
"$PY" build_s6.py --project "$ROOT" --main ../main --template s6_template.md \
  --refs-main ../main/v12_references.md --refs-extra s6_references_extra.md \
  --figures "$OUT/figures" --out build > build/build_log.txt 2>&1 || { tail -20 build/build_log.txt; exit 1; }
"$PY" check_s6.py build/Supplementary_Material_v6.docx s6_template.md build ../main/build "$ROOT" | tail -3

echo "Manuscript: manuscript/main/build/manuscript_v12_draft.docx and build_frontiers/manuscript_v12_frontiers.docx"
echo "Supplement: manuscript/supplement/build/Supplementary_Material_v6.docx"
