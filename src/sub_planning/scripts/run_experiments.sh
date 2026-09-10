#!/usr/bin/env bash
# ===========================================================================
# run_experiments.sh — 批量运行目标跟踪仿真对比实验
#
# 每次测试的完整生命周期：
#   1. 清理残留进程
#   2. 启动全新 SITL 环境（sim_vehicle.py ArduSub）
#   3. 等待 EKF 开始使用 GPS（MAVLink STATUSTEXT "EKF3 IMU0 is using GPS"，约 50s）
#   4. 启动 ROS 仿真（自动 GUIDED + ARM，metrics_logger 从 ARM 开始记录）
#   5. 运行 DURATION 秒
#   6. 杀掉所有进程（ROS + sim_vehicle SITL 环境）
#   7. 重复下一个方法/种子
#
# 用法：
#   ./scripts/run_experiments.sh                            # 默认：4方法 × 5种子 × 200s
#   ./scripts/run_experiments.sh -m proposed,b1_bearing_only -s 42 -d 120  # 自定义
#
# 选项：
#   -m  方法列表，逗号分隔（默认: proposed,b1_bearing_only,b2_ce_nmpc）
#   -s  随机种子列表，逗号分隔（默认: 42,43,44,45,46）
#   -d  每次仿真运行时长（秒）（默认: 200）
#   -g  GPS 等待超时（秒）（默认: 120）
#   -o  CSV 输出目录（默认: $HOME/sub_planning_metrics）
#   -w  运行间清理等待时间（秒）（默认: 8）
#   --cleanup  仅清理残留进程后退出（脚本中断后的手动清理用）
#   -h  显示此帮助
#
# 前提条件：
#   - 已 source devel/setup.bash 且 sim_vehicle.py 在 PATH 中
#   - 每个方法 × 种子组合生成一个 CSV: {output_dir}/{method}_{timestamp}.csv
#   - 日志: /tmp/sitl_{method}_{seed}.log 与 /tmp/roslaunch_{method}_{seed}.log
#   - SITL 工作目录: /tmp/sitl_{method}_{seed}/（mav.tlog、eeprom.bin 等
#     MAVProxy/SITL 副产物写在这里，不污染工作区根目录）
# ===========================================================================

set -uo pipefail

# 脚本所在目录（用于定位 wait_sitl_gps.py 等辅助工具）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 默认参数 ---
METHODS="proposed,b1_bearing_only,b2_ce_nmpc"
# SEEDS="50,51,52,53,54,55,56,57,58,59"
SEEDS="60,61,62,63,64,65,66,67,68,69"
DURATION=200
GPS_TIMEOUT=120
OUTPUT_DIR="$HOME/sub_planning_metrics"
WAIT_BETWEEN=8
SITL_CMD="sim_vehicle.py -v ArduSub -l 31.284730,121.211700,0,0 --console"

# --- 解析参数 ---
CLEANUP_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -m) METHODS="$2"; shift 2 ;;
        -s) SEEDS="$2"; shift 2 ;;
        -d) DURATION="$2"; shift 2 ;;
        -g) GPS_TIMEOUT="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        -w) WAIT_BETWEEN="$2"; shift 2 ;;
        --cleanup) CLEANUP_ONLY=true; shift ;;
        -h|--help)
            sed -n '1,25p' "$0" | sed 's/^#//; s/^ //'
            exit 0
            ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

# --- 检查环境 ---
if ! command -v roslaunch &>/dev/null; then
    echo "[ERROR] roslaunch 未找到，请先 source devel/setup.bash"
    exit 1
fi

if ! rospack find sub_planning &>/dev/null; then
    echo "[ERROR] sub_planning 包未找到，请先 catkin build"
    exit 1
fi

if ! command -v sim_vehicle.py &>/dev/null; then
    echo "[ERROR] sim_vehicle.py 未在 PATH 中，请确认 ArduPilot 环境"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/ardupilot_ws/bagfiles"

# SITL 二进制无窗口运行（不开 xterm），避免批量运行时 GUI 资源累积导致系统卡死
export SITL_RITW_TERMINAL="bash -c"

# ===========================================================================
#  工具函数
# ===========================================================================

kill_process_group() {
    # 终止整个进程组（setsid 启动的会话）
    local pgid="$1"
    local sig="${2:-TERM}"
    if [[ -n "$pgid" ]] && kill -0 -- "-$pgid" 2>/dev/null; then
        kill -"$sig" -- "-$pgid" 2>/dev/null || true
    fi
}

kill_all() {
    # 终止本次实验的所有进程：ROS 会话、SITL 会话，以及兜底清扫
    echo "[CLEANUP] killing all processes..."
    # ROS 组先发 SIGINT：roslaunch 优雅关闭节点，rosbag 正确落盘
    if [[ -n "$ROS_PGID" ]] && kill -0 -- "-$ROS_PGID" 2>/dev/null; then
        kill -INT -- "-$ROS_PGID" 2>/dev/null || true
        sleep 5
        kill_process_group "$ROS_PGID" TERM
        sleep 3
        kill_process_group "$ROS_PGID" KILL
    fi
    # SITL 组：先 TERM 后 KILL
    kill_process_group "$SITL_PGID" TERM
    sleep 3
    kill_process_group "$SITL_PGID" KILL
    # 兜底清扫（避免残留进程影响下一次运行；[x] 写法防止匹配到自身 shell）
    pkill -f "[r]oslaunch.*target_tracking_simulation" 2>/dev/null || true
    pkill -f "[r]osbag.*record" 2>/dev/null || true
    pkill -f "[r]osmaster" 2>/dev/null || true
    pkill -f "[m]avros_node" 2>/dev/null || true
    pkill -f "[m]avproxy" 2>/dev/null || true
    pkill -f "[s]im_vehicle.py" 2>/dev/null || true
    pkill -f "[s]itl/bin/ardusub" 2>/dev/null || true
    pkill -f "[A]rduSub" 2>/dev/null || true
    pkill -f "[s]cript -qec" 2>/dev/null || true
    pkill -f "[r]viz" 2>/dev/null || true
    sleep 2
}

wait_gps_ready() {
    # 等待 EKF 开始使用 GPS：通过 MAVLink STATUSTEXT 检测
    # "EKF3 IMU0 is using GPS"（控制台提示的原始消息，数据层可捕获）
    local timeout="$1"
    local helper="${SCRIPT_DIR}/utils/wait_sitl_gps.py"
    echo "[SITL] waiting for EKF to use GPS via MAVLink (timeout ${timeout}s)..."
    if python3 "$helper" --timeout "$timeout"; then
        return 0
    fi
    echo "[WARN] EKF-GPS not ready within ${timeout}s — 继续启动（ROS 侧会等待里程计就绪）"
    return 1
}

# ===========================================================================
#  中断处理：Ctrl+C / TERM 时自动清理 SITL 与 ROS 进程
# ===========================================================================

ROS_PGID=""
SITL_PGID=""

interrupt_handler() {
    echo ""
    echo "[INTERRUPT] 收到中断信号，正在清理 SITL 与 ROS 进程..."
    kill_all
    echo "[INTERRUPT] 清理完成，脚本退出"
    exit 130
}
trap interrupt_handler INT TERM

# ===========================================================================
#  仅清理模式：./run_experiments.sh --cleanup（手动清理残留用）
# ===========================================================================

if [[ "$CLEANUP_ONLY" == true ]]; then
    echo "[CLEANUP] 仅清理模式：终止所有残留的 SITL / ROS / rosbag 进程"
    kill_all
    echo "[CLEANUP] 完成"
    exit 0
fi

# ===========================================================================
#  主流程
# ===========================================================================

IFS=',' read -ra METHOD_LIST <<< "$METHODS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

TOTAL_RUNS=$((${#METHOD_LIST[@]} * ${#SEED_LIST[@]}))
CURRENT=0

echo "============================================"
echo "  批量仿真实验"
echo "  方法数: ${#METHOD_LIST[@]}  种子数: ${#SEED_LIST[@]}"
echo "  每次时长: ${DURATION}s  总次数: ${TOTAL_RUNS}"
echo "  SITL: ${SITL_CMD}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "============================================"

# 启动前彻底清理一次（避免上次中断的残留）
kill_all

for method in "${METHOD_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        CURRENT=$((CURRENT + 1))
        SITL_LOG="/tmp/sitl_${method}_${seed}.log"
        ROS_LOG="/tmp/roslaunch_${method}_${seed}.log"
        echo ""
        echo "=============================================="
        echo "[${CURRENT}/${TOTAL_RUNS}] method=${method} seed=${seed}"
        echo "=============================================="

        # ---------- 1. 清理上一轮残留 ----------
        kill_all

        # ---------- 2. 启动全新 SITL 环境 ----------
        echo "[SITL] 启动新仿真环境: ${SITL_CMD}"
        # script 提供伪终端（--console 需要 TTY），输出写入日志便于检测 GPS。
        # SITL 工作目录切到 /tmp：MAVProxy 与 SITL 会在 CWD 生成
        # mav.tlog/mav.parm/eeprom.bin，避免污染工作区根目录；
        # 按 method_seed 分目录，便于失败后排查（每轮覆盖同组合旧文件）
        SITL_WORKDIR="/tmp/sitl_${method}_${seed}"
        mkdir -p "${SITL_WORKDIR}"
        setsid script -qec "cd ${SITL_WORKDIR} && ${SITL_CMD}" "${SITL_LOG}" >/dev/null 2>&1 &
        SITL_PGID=$!
        echo "[SITL] 进程组 PGID=${SITL_PGID}, 日志: ${SITL_LOG}, 工作目录: ${SITL_WORKDIR}"

        # ---------- 3. 等待 GPS（MAVLink 数据检测，约 50s） ----------
        wait_gps_ready "$GPS_TIMEOUT"

        # ---------- 4. 启动 ROS 仿真 ----------
        echo "[ROS] 启动: method=${method} seed=${seed}"
        setsid roslaunch sub_planning target_tracking_simulation.launch \
            method:="${method}" \
            random_seed:="${seed}" \
            forward_sonar_window:=false \
            forward_sonar_debug_window:=false \
            show_rviz:=false \
            output_dir:="${OUTPUT_DIR}" \
            > "${ROS_LOG}" 2>&1 &
        ROS_PGID=$!
        echo "[ROS] 进程组 PGID=${ROS_PGID}, 日志: ${ROS_LOG}"

        # ---------- 5. 运行 DURATION 秒 ----------
        echo "[RUN] 运行 ${DURATION}s ..."
        sleep "$DURATION"

        # ---------- 6. 杀掉所有进程 ----------
        kill_all

        sleep "$WAIT_BETWEEN"
        echo "[INFO] 完成. CSV 位于 ${OUTPUT_DIR}/"
    done
done

# 最终清理
kill_all

echo ""
echo "============================================"
echo "  全部完成! ${TOTAL_RUNS} 次运行"
echo "  CSV 文件位于: ${OUTPUT_DIR}/"
echo "============================================"
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "(目录为空或不存在)"
