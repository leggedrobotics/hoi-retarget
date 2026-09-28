#!/usr/bin/env bash
# Retarget an OMOMO corpus, in parallel, on one machine. Each clip's object is read from
# its file name and its output dir is the file stem, so other sources go one clip at a time.
#
#     find -L data/InterMimic/OMOMO_new -name '*.pt' > outputs/clips.txt
#     hoi_retarget/tools/batch_retarget.sh outputs/clips.txt outputs/omomo [workers] [hoi-retarget flags...]
#
# Extra flags go to every solve, e.g. `--robot unitree_h2`. Each solve needs ~2 GB of
# RAM, so workers default to 4. The unit of work is one clip, so this maps onto any
# scheduler: replace `parallel` with `sbatch --array`. Idempotent -- a clip that
# already has a contact_window.pkl is skipped, so an interrupted run resumes.
set -euo pipefail

USAGE="usage: batch_retarget.sh <clips.txt> <out_dir> [workers] [hoi-retarget flags...]"
CLIP_LIST="${1:?$USAGE}"
OUT_DIR="${2:?$USAGE}"
WORKERS="${3:-4}"
shift $(( $# < 3 ? $# : 3 ))
export HR_EXTRA="$*"

command -v parallel >/dev/null || { echo "need GNU parallel (conda install -c conda-forge parallel)" >&2; exit 1; }

# One clip path per line; blank lines and # comments ignored.
mapfile -t CLIPS < <(grep -vE '^\s*(#|$)' "$CLIP_LIST" || true)
[[ ${#CLIPS[@]} -gt 0 ]] || { echo "no clips in $CLIP_LIST" >&2; exit 1; }
echo "clips    : ${#CLIPS[@]}   from: $CLIP_LIST"
echo "workers  : $WORKERS   -> $OUT_DIR   ${HR_EXTRA:+flags: $HR_EXTRA}"

retarget_one() {
  local clip="$1" out_dir="$2"
  local stem; stem=$(basename "${clip%.pt}")
  if [[ -f "$out_dir/$stem/contact_window.pkl" ]]; then
    echo "skip  $stem"; return 0
  fi
  # shellcheck disable=SC2086  # HR_EXTRA is a flag list
  hoi-retarget --input_file "$clip" --out_dir "$out_dir/$stem" \
               --mode contact --no-record_video --headless $HR_EXTRA \
    && echo "done  $stem" \
    || { echo "FAIL  $stem" >&2; return 1; }
}
export -f retarget_one

# --halt never: one unsolvable clip should not abandon the rest.
rc=0
printf '%s\n' "${CLIPS[@]}" \
  | parallel --halt never -j "$WORKERS" retarget_one {} "$OUT_DIR" || rc=$?

solved=$(find "$OUT_DIR" -name contact_window.pkl | wc -l)
echo
echo "solved ${solved}/${#CLIPS[@]}"
exit "$rc"
