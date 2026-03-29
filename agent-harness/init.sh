#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITY_VERSION="$(sed -n 's/^m_EditorVersion: //p' "$ROOT_DIR/ProjectSettings/ProjectVersion.txt" | head -n 1)"
SCENE_PATH="Assets/OpenRDW/Scenes/OpenRDW Scene.unity"
RESULTS_DIR="Experiment Results"
FEATURES_FILE="$ROOT_DIR/agent-harness/feature_list.json"
PROGRESS_FILE="$ROOT_DIR/agent-harness/progress.md"

find_unity_editor() {
  local candidates=(
    "/Applications/Unity/Hub/Editor/${UNITY_VERSION}/Unity.app/Contents/MacOS/Unity"
    "/Applications/Unity/Unity.app/Contents/MacOS/Unity"
    "$HOME/Applications/Unity/Hub/Editor/${UNITY_VERSION}/Unity.app/Contents/MacOS/Unity"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

print_header() {
  cat <<EOF
OpenRdw_vbdi long-running agent harness
======================================
root: $ROOT_DIR
unity_version: ${UNITY_VERSION:-unknown}
scene: $SCENE_PATH
results_dir: $RESULTS_DIR
features_file: $FEATURES_FILE
progress_file: $PROGRESS_FILE
EOF
}

print_git_status() {
  if git -C "$ROOT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    echo
    echo "[git]"
    git -C "$ROOT_DIR" status --short
  else
    echo
    echo "[git]"
    echo "No git repository detected at project root."
    echo "Recommended bootstrap command:"
    echo "  git init && git add agent-harness ProjectSettings Packages Assets/OpenRDW && git commit -m 'chore: bootstrap agent harness'"
  fi
}

print_unity_commands() {
  echo
  echo "[unity]"
  if UNITY_EDITOR="$(find_unity_editor)"; then
    echo "Unity editor found:"
    echo "  $UNITY_EDITOR"
    echo "Suggested batchmode smoke command:"
    echo "  \"$UNITY_EDITOR\" -batchmode -projectPath \"$ROOT_DIR\" -quit -logFile \"$ROOT_DIR/Logs/agent-smoke.log\""
  else
    echo "Unity editor not found in standard macOS locations."
    echo "Set UNITY_EDITOR manually before running batchmode checks."
    echo "Expected version: ${UNITY_VERSION:-unknown}"
  fi

  echo
  echo "Suggested manual validation target:"
  echo "  Open scene: $SCENE_PATH"
}

print_agent_checklist() {
  cat <<'EOF'

[session checklist]
1. Read agent-harness/progress.md
2. Read agent-harness/feature_list.json
3. Pick exactly one top-priority item with status "todo" or "blocked"
4. Run the smallest validation that proves the project still boots
5. Implement only that one item
6. Re-run validation and record evidence in progress.md
7. Update feature_list.json status fields before ending the session
EOF
}

print_header
print_git_status
print_unity_commands
print_agent_checklist
