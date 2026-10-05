#!/usr/bin/env bash
# Chạy thực nghiệm colabmode theo LƯỢT: mỗi lượt = 1 target, chạy đồng thời 4 pair.
# Mỗi pair = 2 biến thể chạy song song, cùng target, cùng shared_seeds_queue (riêng cho từng kịch bản).
# 4 pair x 5 target = 20 kịch bản = 5 lượt; lượt sau chỉ bắt đầu khi lượt trước xong hết.
#
#   ./run_experiments.sh            chạy (nên dùng: nohup setsid ./run_experiments.sh > master.log 2>&1 &)
#   ./run_experiments.sh stop       dừng an toàn toàn bộ
#   ./run_experiments.sh status     xem trạng thái
#
# Chạy thử nhanh 1 target:
#   TARGET_LIST="idna_fuzz" DURATION=10m ./run_experiments.sh

set -u

############ CẤU HÌNH (đều ghi đè được bằng biến môi trường) ############
SRC="${SRC:-$(pwd)}"
RUNS="${RUNS:-$SRC/runs}"
SEED_SRC="${SEED_SRC:-$SRC/shared_seeds_queue}"
PYTHON="${PYTHON:-python3}"
DURATION="${DURATION:-6h}"
CORE_OFFSET="${CORE_OFFSET:-1}"       # core bắt đầu ghim (core 0 để lại cho hệ điều hành)
CPUS_PER_PROC="${CPUS_PER_PROC:-1}"   # số core cho MỖI tiến trình fuzz
REPEATS="${REPEATS:-1}"
LINK_TARGETS="${LINK_TARGETS:-0}"
KILL_GRACE="${KILL_GRACE:-300}"       # sau khi hết giờ + SIGINT, chờ bấy nhiêu giây rồi SIGKILL
STOP_INT_WAIT="${STOP_INT_WAIT:-30}"

# 4 pair chạy đồng thời trong mỗi lượt (ghi đè được, phân tách bằng khoảng trắng)
PAIR_LIST="${PAIR_LIST:-colabmode_pythonfuzz+colabmode_pythonfuzz colabmode_pythonfuzz+colabmode_grpo colabmode_grpo+colabmode_grpo}"
TARGET_LIST="${TARGET_LIST:-dateutil_fuzz}"
read -r -a PAIRS <<< "$PAIR_LIST"
read -r -a TARGETS <<< "$TARGET_LIST"
#########################################################################

SRC="$(readlink -f "$SRC")"
mkdir -p "$RUNS"; RUNS="$(readlink -f "$RUNS")"
SEED_SRC="$(readlink -f "$SEED_SRC")"
PYPATH_DIR="$RUNS/.pypath"        # chứa symlink targets -> $SRC/targets (chỉ đọc)
STOP_FILE="$RUNS/.stop"
PID_DIR="$RUNS/.pids"
MASTER_PID_FILE="$RUNS/.master.pid"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- lệnh phụ: stop / status ----
case "${1:-run}" in
  stop)
    if [ -f "$MASTER_PID_FILE" ] && kill -0 "$(cat "$MASTER_PID_FILE")" 2>/dev/null; then
      kill -TERM "$(cat "$MASTER_PID_FILE")"
      echo "Đã gửi SIGTERM tới script chính (pid $(cat "$MASTER_PID_FILE")). Xem master.log để theo dõi."
    else
      echo "Không có script chính đang chạy."
    fi
    exit 0 ;;
  status)
    echo "Script chính: $( [ -f "$MASTER_PID_FILE" ] && kill -0 "$(cat "$MASTER_PID_FILE")" 2>/dev/null && echo "đang chạy (pid $(cat "$MASTER_PID_FILE"))" || echo "không chạy" )"
    echo "Đã xong: $(ls "$RUNS"/*/.done 2>/dev/null | wc -l)"
    echo "Đang chạy:"
    for f in "$PID_DIR"/*.pgid; do
      [ -e "$f" ] || continue
      pg="$(cat "$f")"
      kill -0 "-$pg" 2>/dev/null && echo "  $(basename "$f" .pgid)  (pgid $pg)"
    done
    exit 0 ;;
  run) ;;
  *) echo "Dùng: $0 [run|stop|status]"; exit 2 ;;
esac

# ---- chỉ cho phép 1 bản script chạy trên cùng thư mục RUNS ----
exec 9>"$RUNS/.lock"
if ! flock -n 9; then
  echo "Đã có một script khác đang chạy với RUNS=$RUNS. Dừng nó trước: $0 stop"; exit 1
fi
echo $$ > "$MASTER_PID_FILE"

# ---- kiểm tra đầu vào ----
for p in "${PAIRS[@]}"; do
  for v in "${p%%+*}" "${p##*+}"; do
    [ -d "$SRC/$v" ] || { echo "Thiếu thư mục $SRC/$v"; exit 1; }
    [ -f "$SRC/$v/main.py" ] || { echo "Thiếu $SRC/$v/main.py"; exit 1; }
  done
done
for t in "${TARGETS[@]}"; do
  [ -f "$SRC/targets/$t.py" ] || { echo "Thiếu target $SRC/targets/$t.py"; exit 1; }
done

# PYTHON -> đường dẫn tuyệt đối (không dùng readlink -f để không phá venv)
_py="$(command -v "$PYTHON")" || { echo "Không tìm thấy PYTHON='$PYTHON'"; exit 1; }
case "$_py" in /*) PYTHON="$_py" ;; *) PYTHON="$PWD/$_py" ;; esac

mkdir -p "$PYPATH_DIR"; ln -sfn "$SRC/targets" "$PYPATH_DIR/targets"

# Preflight: cảnh báo đường dẫn cứng/tương đối đáng ngờ trong mã nguồn (không chặn chạy)
preflight() {
  local v
  for v in $(printf '%s\n' "${PAIRS[@]}" | tr '+' '\n' | sort -u); do
    grep -rnE --include='*.py' '/content/|/drive/|/home/|/Users/|[A-Za-z]:\\\\|\.\./' "$SRC/$v" 2>/dev/null \
      | sed "s|^|[preflight] |" | head -15
  done
}
preflight

SLOTS=${#PAIRS[@]}                         # mỗi pair 1 slot, chạy đồng thời trong cùng lượt
njobs=$(( SLOTS * ${#TARGETS[@]} * REPEATS ))
need_cores=$(( CORE_OFFSET + SLOTS * 2 * CPUS_PER_PROC ))
ncpu="${NCPU:-$(nproc)}"
if [ "$need_cores" -gt "$ncpu" ]; then
  echo "Cần $need_cores core (offset $CORE_OFFSET + $SLOTS pair x 2 x $CPUS_PER_PROC) nhưng máy chỉ có $ncpu. Giảm CPUS_PER_PROC."; exit 1
fi

rm -f "$STOP_FILE"; rm -rf "$PID_DIR"; mkdir -p "$PID_DIR"

# ---- dừng an toàn ----
# Mỗi tiến trình fuzz chạy trong session/process group riêng (setsid), pgid lưu ở $PID_DIR.
# Dừng theo pgid nên không bao giờ đụng tới tiến trình ngoài script.
signal_all() {  # $1 = tên tín hiệu
  local f pg
  for f in "$PID_DIR"/*.pgid; do
    [ -e "$f" ] || continue
    pg="$(cat "$f" 2>/dev/null)"; [ -n "$pg" ] && kill -"$1" -- "-$pg" 2>/dev/null
  done
}
any_alive() {
  local f pg
  for f in "$PID_DIR"/*.pgid; do
    [ -e "$f" ] || continue
    pg="$(cat "$f" 2>/dev/null)"; [ -n "$pg" ] && kill -0 -- "-$pg" 2>/dev/null && return 0
  done
  return 1
}
CLEANING=0
cleanup() {
  [ "$CLEANING" = 1 ] && return; CLEANING=1
  trap '' INT TERM HUP
  touch "$STOP_FILE"                       # chặn mọi lượt/job mới
  log "Nhận tín hiệu dừng: gửi SIGINT (cho main.py ghi kết quả cuối)..."
  signal_all INT
  local i
  for ((i=0; i<STOP_INT_WAIT; i++)); do any_alive || break; sleep 1; done
  if any_alive; then log "Còn tiến trình, gửi SIGTERM..."; signal_all TERM; sleep 5; fi
  if any_alive; then log "Còn tiến trình, gửi SIGKILL..."; signal_all KILL; fi
  kill $(jobs -p) 2>/dev/null              # các subshell của job
  rm -f "$MASTER_PID_FILE"
  log "Đã dừng."
  exit 130
}
trap cleanup INT TERM HUP
trap 'rm -f "$MASTER_PID_FILE"' EXIT

run_job() {
  local slot="$1" pair="$2" target="$3" rep="$4"
  local vA="${pair%%+*}" vB="${pair##*+}"
  # Tên thư mục con trong runs/<id>/. Nếu 2 biến thể giống nhau (vd pythonfuzz+pythonfuzz)
  # thì thêm hậu tố _1/_2 để không trùng thư mục.
  local dA="$vA" dB="$vB"
  if [ "$vA" = "$vB" ]; then dA="${vA}_1"; dB="${vB}_2"; fi
  local id="${pair}__${target}__r${rep}"
  local rd="$RUNS/$id"
  # Dải core: mỗi tiến trình CPUS_PER_PROC core liên tiếp, không chồng lấn giữa các slot
  local w="$CPUS_PER_PROC"
  local base=$(( CORE_OFFSET + slot * 2 * w ))
  local coreA="${base}-$(( base + w - 1 ))"
  local coreB="$(( base + w ))-$(( base + 2*w - 1 ))"

  [ -e "$STOP_FILE" ] && return
  if [ -f "$rd/.done" ]; then log "[slot $slot] SKIP $id (đã xong)"; return; fi

  rm -rf "$rd"
  mkdir -p "$rd/shared_seeds_queue"
  cp -a "$SRC/$vA" "$rd/$dA"
  cp -a "$SRC/$vB" "$rd/$dB"
  find "$rd" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
  [ "$LINK_TARGETS" = "1" ] && ln -s "$SRC/targets" "$rd/targets"
  if [ -d "$SEED_SRC" ] && [ -n "$(ls -A "$SEED_SRC" 2>/dev/null)" ]; then
    cp -a "$SEED_SRC"/. "$rd/shared_seeds_queue"/
  fi

  local targ="$target"
  local shared="$rd/shared_seeds_queue"          # tuyệt đối (rd bắt nguồn từ RUNS đã readlink -f)
  local seeds0; seeds0=$(find "$shared" -type f 2>/dev/null | wc -l)

  log "[slot $slot] START $id  (cores $coreA | $coreB; seed ban đầu: $seeds0)"
  date +%s > "$rd/start_time"

  local pids=() names=() variant core pidf
  local n
  for n in 1 2; do
    if [ "$n" = 1 ]; then variant="$dA"; core="$coreA"; else variant="$dB"; core="$coreB"; fi
    pidf="$PID_DIR/${id}.${variant}.pgid"
    [ -e "$STOP_FILE" ] && break
    mkdir -p "$rd/$variant/.tmp"
    # setsid: session + process group riêng -> kill theo pgid, không lọt tiến trình con (multiprocessing).
    # Python reset SIGINT về mặc định trước khi exec: tiến trình nền của bash mặc định bị ignore SIGINT,
    # cờ này kế thừa qua exec và làm main.py không nhận được KeyboardInterrupt.
    PIDF="$pidf" WD="$rd/$variant" CORE="$core" DUR="$DURATION" GRACE="$KILL_GRACE" THREADS="$w" \
    PY="$PYTHON" TARG="$targ" SHARED="$shared" SRC_DIR="$SRC" RUN_DIR="$rd" \
    MAIN="$rd/$variant/main.py" PYPATH="$PYPATH_DIR" \
    setsid -w bash -c '
      echo $$ > "$PIDF"
      cd "$WD" || exit 99
      # RUN_DIR đứng đầu: nếu code import theo tên gói (colabmode_x) thì lấy bản copy, không phải bản gốc.
      # Không đưa $SRC vào path để tránh vô tình import/ghi vào thư mục gốc.
      export PYTHONPATH="$RUN_DIR:$PYPATH:$SRC_DIR/targets:${PYTHONPATH:-}"
      export TMPDIR="$WD/.tmp"
      export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
      export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
      exec taskset -c "$CORE" timeout --foreground -s INT -k "$GRACE" "$DUR" \
        "$PY" -c "import os,signal,sys;signal.signal(signal.SIGINT,signal.SIG_DFL);os.execvp(sys.argv[1],sys.argv[1:])" \
        "$PY" -u "$MAIN" "$TARG" "$SHARED"
    ' > "$rd/$variant/run.log" 2>&1 &
    pids+=($!); names+=("$variant")
    sleep 2
  done

  local rcs=() i rc
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    rcs+=("$rc")
    rm -f "$PID_DIR/${id}.${names[$i]}.pgid"
  done
  date +%s > "$rd/end_time"
  : > "$rd/exit_codes"
  for i in "${!names[@]}"; do echo "${names[$i]}=${rcs[$i]}" >> "$rd/exit_codes"; done

  if [ -e "$STOP_FILE" ]; then log "[slot $slot] STOPPED $id"; return; fi
  touch "$rd/.done"
  local seeds1; seeds1=$(find "$shared" -type f 2>/dev/null | wc -l)
  log "[slot $slot] DONE  $id (rc: ${rcs[*]};  124 = hết giờ bình thường; seed: $seeds0 -> $seeds1)"
  [ "$seeds1" -le "$seeds0" ] && log "[slot $slot] CẢNH BÁO $id: shared_seeds_queue không có file mới!"
}

# ---- chạy theo lượt: mỗi lượt = 1 target x tất cả pair đồng thời ----
log "Tổng $njobs kịch bản = ${#TARGETS[@]} target x $SLOTS pair x $REPEATS lần; mỗi lượt $SLOTS kịch bản đồng thời, mỗi kịch bản $DURATION"
log "Core: mỗi tiến trình $CPUS_PER_PROC core, dùng core $CORE_OFFSET..$(( need_cores - 1 )); kết quả: $RUNS"

round=0; total_rounds=$(( ${#TARGETS[@]} * REPEATS ))
for ((rep=1; rep<=REPEATS; rep++)); do
  for t in "${TARGETS[@]}"; do
    [ -e "$STOP_FILE" ] && break 2
    round=$(( round + 1 ))
    log "===== LƯỢT $round/$total_rounds: target=$t (lần lặp $rep) ====="
    s=0
    for p in "${PAIRS[@]}"; do
      run_job "$s" "$p" "$t" "$rep" &
      s=$(( s + 1 ))
    done
    wait                                   # chờ hết cả 4 pair của lượt này
  done
done

log "HOÀN TẤT toàn bộ."
