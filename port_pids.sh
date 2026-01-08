#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash port_pids.sh              # default ports
#   bash port_pids.sh 5055 5355    # custom ports
#
# No ss/netstat/lsof needed. Uses /proc.

PORTS=("$@")
if [ ${#PORTS[@]} -eq 0 ]; then
  PORTS=(5055 5305 5355 5405 5505)
fi

echo "[Info] target ports: ${PORTS[*]}"

# helper: port -> 4-hex uppercase (e.g. 5055 -> 13BF)
to_hex_port() {
  printf "%04X" "$1"
}

# collect target hex ports
HEX_PORTS=()
for p in "${PORTS[@]}"; do
  HEX_PORTS+=("$(to_hex_port "$p")")
done

# Build a quick grep pattern like ":13BF|:14EB|..."
PAT=""
for hp in "${HEX_PORTS[@]}"; do
  if [ -z "$PAT" ]; then
    PAT=":${hp}"
  else
    PAT="${PAT}|:${hp}"
  fi
done

# 1) Find LISTEN sockets for those ports in /proc/net/tcp and tcp6
# state LISTEN = 0A
# columns: sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode ...
# inode is 10th field (index may differ, but usually field 10)
LISTEN_LINES=$(
  { cat /proc/net/tcp 2>/dev/null || true; cat /proc/net/tcp6 2>/dev/null || true; } \
  | awk 'NR>1 && $4=="0A" {print}' \
  | egrep -i "${PAT}" || true
)

if [ -z "${LISTEN_LINES}" ]; then
  echo "[Warn] no LISTEN sockets found for target ports in /proc/net/tcp*"
  exit 0
fi

echo "[Info] LISTEN sockets (raw):"
echo "${LISTEN_LINES}"
echo

# 2) Map inode -> port(hex)
# local_address is field 2, inode is field 10
# local_address looks like IPHEX:PORTHEX
declare -A INODE2PORTHEX
while read -r line; do
  [ -z "$line" ] && continue
  local_addr=$(echo "$line" | awk '{print $2}')
  inode=$(echo "$line" | awk '{print $10}')
  port_hex=${local_addr##*:}
  INODE2PORTHEX["$inode"]="$port_hex"
done <<< "${LISTEN_LINES}"

# 3) Scan /proc/*/fd symlinks for socket:[inode]
echo "[Result] port -> pid/cmd (LISTEN)"
for inode in "${!INODE2PORTHEX[@]}"; do
  port_hex="${INODE2PORTHEX[$inode]}"
  # hex -> dec
  port_dec=$((16#$port_hex))

  # find PIDs holding this inode
  # readlink will print socket:[12345]
  for fd in /proc/[0-9]*/fd/*; do
    # skip unreadable
    link=$(readlink "$fd" 2>/dev/null || true)
    [ -z "$link" ] && continue
    case "$link" in
      socket:\["$inode"\])
        pid=$(echo "$fd" | cut -d/ -f3)
        cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | sed 's/ *$//')
        if [ -z "$cmd" ]; then
          cmd="(cmdline unreadable)"
        fi
        echo "port ${port_dec} (0x${port_hex}) -> pid ${pid} cmd: ${cmd}"
        ;;
    esac
  done
done
#!/usr/bin/env bash
set -euo pipefail

PORTS=("$@")
if [ ${#PORTS[@]} -eq 0 ]; then
  PORTS=(5055 5305 5355 5405 5505)
fi

echo "[Info] target ports: ${PORTS[*]}"

to_hex_port() { printf "%04X" "$1"; }

# porthex set 만들기
HEX_PORTS=()
for p in "${PORTS[@]}"; do HEX_PORTS+=("$(to_hex_port "$p")"); done

PAT=""
for hp in "${HEX_PORTS[@]}"; do
  PAT="${PAT:+$PAT|}:${hp}"
done

# LISTEN(0A)에서 inode 뽑기 (tcp/tcp6 둘 다)
LISTEN_LINES=$(
  { cat /proc/net/tcp 2>/dev/null || true; cat /proc/net/tcp6 2>/dev/null || true; } \
  | awk 'NR>1 && $4=="0A" {print}' \
  | egrep -i "${PAT}" || true
)

if [ -z "${LISTEN_LINES}" ]; then
  echo "[Warn] no LISTEN sockets found"
  exit 0
fi

echo "[Info] LISTEN sockets (raw):"
echo "${LISTEN_LINES}"
echo

declare -A INODE2PORTHEX
while read -r line; do
  [ -z "$line" ] && continue
  local_addr=$(awk '{print $2}' <<<"$line")
  inode=$(awk '{print $10}' <<<"$line")
  port_hex=${local_addr##*:}
  INODE2PORTHEX["$inode"]="$port_hex"
done <<< "${LISTEN_LINES}"

# 후보 PID를 좁히면 훨씬 빠르고, hidepid 환경에서도 "내 프로세스"는 보일 확률이 큼
# Unity/게임 실행파일명이 다르면 여기 grep 패턴만 바꿔줘
PIDS=$(
  ps -eo pid,args \
  | egrep -i 'immortal|unity|immortal_suffering|train\.py|torchrun' \
  | grep -v egrep \
  | awk '{print $1}' \
  | sort -n \
  | uniq
)

echo "[Info] scanning candidate pids:"
echo "$PIDS"
echo

found=0
echo "[Result] port -> pid/cmd (LISTEN)"
for pid in $PIDS; do
  fd_dir="/proc/${pid}/fd"
  [ -d "$fd_dir" ] || continue
  for fd in "$fd_dir"/*; do
    link=$(readlink "$fd" 2>/dev/null || true)
    [ -n "$link" ] || continue
    # socket:[12345] 형태만 잡기
    if [[ "$link" == socket:\[*\] ]]; then
      inode="${link#socket:[}"
      inode="${inode%]}"
      if [[ -n "${INODE2PORTHEX[$inode]+x}" ]]; then
        port_hex="${INODE2PORTHEX[$inode]}"
        port_dec=$((16#$port_hex))
        cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | sed 's/ *$//')
        [ -n "$cmd" ] || cmd="(cmdline unreadable)"
        echo "port ${port_dec} (0x${port_hex}) -> pid ${pid} cmd: ${cmd}"
        found=1
      fi
    fi
  done
done

if [ "$found" -eq 0 ]; then
  echo
  echo "[Warn] No PID matched."
  echo "  - If /proc is mounted with hidepid, try running as root (sudo) or limit to your known PIDs."
  echo "  - Or: the listeners might belong to a process not matching the grep filter above."
fi
