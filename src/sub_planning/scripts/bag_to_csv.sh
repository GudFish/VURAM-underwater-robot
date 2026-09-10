#!/usr/bin/env bash
# ===========================================================================
# bag_to_csv.sh — 从 rosbag 重新生成 metrics CSV（重启丢失 /tmp 后的恢复工具）
#
# 原理：回放 bag（--clock 发布仿真时钟），同时运行 metrics_logger 节点
#       （use_sim_time=true，Timer 跟随仿真时钟），完整重建原 CSV。
#       CSV 的 t_sec 与在线运行时一致（ARM 上升沿为 0）。
#
# 用法：
#   ./scripts/bag_to_csv.sh                              # 处理 ~/ardupilot_ws/bagfiles 下所有 bag
#   ./scripts/bag_to_csv.sh -i 某个.bag -o ~/metrics     # 单个 bag
#   ./scripts/bag_to_csv.sh -r 5                         # 5 倍速回放
#
# 选项：
#   -i  bag 文件或目录（默认 ~/ardupilot_ws/bagfiles）
#   -o  CSV 输出目录（默认 $HOME/sub_planning_metrics）
#   -r  回放倍速（默认 10；Timer 跟随仿真时钟，倍速不影响采样密度，
#       指标话题均为低带宽，10~20 倍速安全；过高可能丢消息）
#   -h  显示此帮助
#
# 注意：
#   - 方法名从 bag 文件名自动提取（_seed 前缀）
#   - /planner/recovery_status 未被旧 bag 记录，对应列为 NaN（不影响其余指标）
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BAG_INPUT="$HOME/ardupilot_ws/bagfiles"
OUTPUT_DIR="$HOME/sub_planning_metrics"
RATE=10

while [[ $# -gt 0 ]]; do
    case $1 in
        -i) BAG_INPUT="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        -r) RATE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,28p' "$0" | sed 's/^#//; s/^ //'
            exit 0
            ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

if ! command -v rosbag &>/dev/null; then
    echo "[ERROR] rosbag 未找到，请先 source devel/setup.bash"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- 收集 bag 列表 ---
if [[ -f "$BAG_INPUT" ]]; then
    BAGS=("$BAG_INPUT")
else
    mapfile -t BAGS < <(find "$BAG_INPUT" -maxdepth 1 -name '*.bag' | sort)
fi

if [[ ${#BAGS[@]} -eq 0 ]]; then
    echo "[ERROR] 未找到 bag 文件: $BAG_INPUT"
    exit 1
fi
echo "待处理 bag 数: ${#BAGS[@]}"

wait_for_master() {
    # 轮询等待 ROS master 就绪（首次启动可能较慢，也可能复用残留 master）
    local timeout="${1:-15}"
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if rosparam list >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

process_bag() {
    local bag="$1"
    local fname
    fname="$(basename "$bag" .bag)"
    local method="${fname%%_seed*}"
    # 从 bag 文件名提取种子（命名格式: {method}_seed{seed}_ts_...）
    local seed=""
    if [[ "$fname" =~ _seed([0-9]+)_ ]]; then
        seed="${BASH_REMATCH[1]}"
    fi

    echo ""
    echo "=== 处理: ${fname} (method=${method}, seed=${seed:-未知}) ==="

    # 1. roscore + 轮询等待 master 就绪
    roscore > /tmp/roscore_replay.log 2>&1 &
    local roscore_pid=$!
    if ! wait_for_master 15; then
        echo "[ERROR] ROS master 未在 15s 内就绪，跳过该 bag"
        echo "[ERROR] roscore 日志: /tmp/roscore_replay.log"
        tail -5 /tmp/roscore_replay.log
        kill -TERM "$roscore_pid" 2>/dev/null
        return 1
    fi

    # 2. 仿真时钟
    rosparam set /use_sim_time true

    # 3. metrics_logger（跟随仿真时钟，从 ARM 上升沿开始记录）
    rosrun sub_planning metrics_logger.py \
        _method:="$method" \
        _random_seed:="$seed" \
        _output_dir:="$OUTPUT_DIR" \
        > "/tmp/metrics_replay_${method}.log" 2>&1 &
    local logger_pid=$!
    sleep 2

    # 4. 回放 bag
    rosbag play --clock -r "$RATE" "$bag" > "/tmp/bag_play_${fname}.log" 2>&1

    # 5. 回放结束：SIGINT 优雅关闭 logger（触发 flush 落盘），再清理
    sleep 2
    kill -INT "$logger_pid" 2>/dev/null
    sleep 3
    kill -TERM "$logger_pid" 2>/dev/null
    kill -TERM "$roscore_pid" 2>/dev/null
    sleep 2
    kill -KILL "$logger_pid" "$roscore_pid" 2>/dev/null
    wait 2>/dev/null

    echo "[OK] ${fname} 完成，CSV 位于 ${OUTPUT_DIR}/"
}

for bag in "${BAGS[@]}"; do
    process_bag "$bag"
done

echo ""
echo "=== 全部完成 ==="
ls -la "$OUTPUT_DIR/"*.csv 2>/dev/null | tail -25

# 清理可能残留的 roscore（[r] 写法避免匹配自身 shell）
pkill -f "[r]osmaster" 2>/dev/null || true
echo "rosmaster 已清理"
