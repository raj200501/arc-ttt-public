#!/usr/bin/env bash
# B.7-r6 one-command resume for a cancelled checkpointed arm kernel.
#
# Usage: scripts/resume_arm_kernel.sh <suffix>        e.g. s1a, s3kb
#
# Pulls the cancelled kernel's output (artifacts + novel_ckpt_* files),
# banks any completed artifacts, versions the cord-validation dataset so
# the checkpoint files seed the relaunch, and re-pushes the same kernel.
# The kernel then restores the adapter bit-identically and skips journaled
# docs (entry-side logic, ENTERPRISE_EVAL_SPEC.md B.7-r6).
set -euo pipefail

SUFFIX="${1:?usage: resume_arm_kernel.sh <suffix like s1a>}"
KERNEL="rajskashikar/arc-ttt-novel-schema-${SUFFIX}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PULL_DIR="$(mktemp -d /tmp/pull_resume_XXXX)"
DS_DIR="$(mktemp -d /tmp/dsstage_resume_XXXX)"

echo "== pulling ${KERNEL}"
kaggle kernels output "${KERNEL}" -p "${PULL_DIR}"
ls "${PULL_DIR}"

echo "== banking completed artifacts (if any)"
python3 "${ROOT}/scripts/bank_novel_schema.py" \
  --pull-dir "${PULL_DIR}" --experiments "${ROOT}/experiments" --date 2026-08-12 || {
    echo "BANKER REFUSED — resolve by the preregistered first-terminal-wins rule before rerunning"; exit 2; }

if ls "${PULL_DIR}"/novel_ckpt_* >/dev/null 2>&1; then
  echo "== versioning dataset with checkpoint files"
  kaggle datasets download rajskashikar/cord-validation --unzip -p "${DS_DIR}"
  kaggle datasets metadata rajskashikar/cord-validation -p "${DS_DIR}"
  python3 - "$DS_DIR" <<'EOF'
import json, sys
p = f"{sys.argv[1]}/dataset-metadata.json"
d = json.load(open(p)); d.setdefault("id", "rajskashikar/cord-validation")
json.dump(d, open(p, "w"), indent=1)
EOF
  cp "${PULL_DIR}"/novel_ckpt_* "${DS_DIR}/"
  # newly banked artifacts also seed, so completed arms skip entirely
  cp "${ROOT}"/experiments/novel_schema_0.5b_k*_2026-08-12.json "${DS_DIR}/" 2>/dev/null || true
  (cd "${DS_DIR}" && kaggle datasets version -p . -m "resume seed for ${SUFFIX}: checkpoints + banked arms")
  echo "waiting 90s for dataset version to process"; sleep 90
else
  echo "== no checkpoint files in output (kernel died before adapt?) — plain relaunch"
fi

echo "== re-pushing ${KERNEL}"
(cd "${ROOT}/kaggle" && kaggle kernels push -p "novel-schema-${SUFFIX}") || {
  echo "push failed (slot cap?) — delete+repush:";
  yes | kaggle kernels delete "${KERNEL}";
  (cd "${ROOT}/kaggle" && kaggle kernels push -p "novel-schema-${SUFFIX}"); }
echo "== resumed ${KERNEL}"
