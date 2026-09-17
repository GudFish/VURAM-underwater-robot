#!/usr/bin/env bash
# ===========================================================================
# experiment_bag_to_csv.sh — 从实物实验 rosbag 重新生成 metrics CSV
#
# 参考 bag_to_csv.sh（仿真 bag 恢复工具），针对 bagfiles/experiment/ 下的
# 实物实验 bag：
#   - 递归查找 experiment/ 下所有子目录（shangwu/、xiawu/ 等）的 bag
#   - CSV 文件名与对应 bag 完全一致（仅扩展名 .csv），例如
#     proposed_real_ts_nvidia_desktop_20260914_101252.bag
#     → proposed_real_ts_nvidia_desktop_20260914_101252.csv
#   - 只回放 metrics 相关话题（--topics 过滤），图像/TF 不参与，
#     大体积 bag 也能高速提取
#   - 声呐几何参数与 target_tracking_real.launch 默认一致（量程 10m），
#     保证 target_in_sonar_fov 列与在线实验口径相同
#
# CSV 列与 ~/sub_planning_metrics/ 内仿真 CSV 完全一致（同一 metrics_logger 节点）：
#   t_sec, gt_x, gt_y, robot_x, robot_y, robot_psi, estimation_pos_error_m,
#   cov_trace_m2, gt_standoff_m, standoff_error_m, within_standoff_tolerance,
#   control_effort_sq, target_in_sonar_fov, solver_time_ms, solver_feasible,
#   recovery_active, recovery_mode, recovery_elapsed_s, recovery_count,
#   min_obstacle_clearance_m, near_collision_count, collision_count
#   （实物实验无仿真障碍物话题，后三列为 NaN，不影响其余指标）
#
# 用法：
#   ./scripts/experiment_bag_to_csv.sh                # 处理 experiment/ 下所有 bag
#   ./scripts/experiment_bag_to_csv.sh -i 某个.bag -o ~/metrics
#   ./scripts/experiment_bag_to_csv.sh -r 20          # 20 倍速回放
#
# 选项：
#   -i  bag 文件或目录（默认 ~/ardupilot_ws/bagfiles/experiment）
#   -o  CSV 输出目录（默认 $HOME/sub_planning_metrics）
#   -r  回放倍速（默认 10；只回放低带宽指标话题，可上调到 20）
#   -h  显示此帮助
#
# 注意：
#   - 需已 source devel/setup.bash（脚本内自动 source）
#   - CSV 与 bag 同名（仅扩展名不同），依赖 metrics_logger 的
#     ~csv_name 参数（默认命名不变，在线仿真不受影响）
# ===========================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HOME}/ardupilot_ws/devel/setup.bash"

BAG_INPUT="${HOME}/ardupilot_ws/bagfiles/experiment"
OUTPUT_DIR="${HOME}/ardupilot_ws/results/sub_planning_metrics_exp"
RATE=10

# --- 声呐几何参数：与 target_tracking_real.launch 默认一致 ---
# （若实验时用非默认参数启动，请同步修改这里）
SONAR_FOV_DEG=130.0
SONAR_MAX_RANGE=10.0
SONAR_MIN_RANGE=0.2
DESIRED_STANDOFF=3.0
STANDOFF_TOLERANCE=0.8

# --- metrics 提取所需话题（其余话题不回放，加速处理） ---
# 注意：本机 rosbag 的 --topics 是空格分隔的多参数列表（吞掉其后所有参数），
# 使用时必须放在 bag 路径之后、且不加引号展开
METRICS_TOPICS="/mavros/state /dynamic_target/pose /target_estimate/pose /target_estimate/belief /mavros/local_position/odom /target_velocity_xy /planner/solver_time /planner/recovery_status /dynamic_obstacles/markers"

while [[ $# -gt 0 ]]; do
    case $1 in
        -i) BAG_INPUT="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        -r) RATE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0" | sed 's/^#//; s/^ //'
            exit 0
            ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

if ! command -v rosbag &>/dev/null; then
    echo "[ERROR] rosbag 未找到，请检查 ROS 环境"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- 收集 bag 列表（递归：experiment/ 下按 shangwu/xiawu 分目录存放） ---
if [[ -f "$BAG_INPUT" ]]; then
    BAGS=("$BAG_INPUT")
else
    mapfile -t BAGS < <(find "$BAG_INPUT" -name '*.bag' | sort)
fi

if [[ ${#BAGS[@]} -eq 0 ]]; then
    echo "[ERROR] 未找到 bag 文件: $BAG_INPUT"
    exit 1
fi
echo "待处理 bag 数: ${#BAGS[@]}"

# ===========================================================================
#  进程管理
# ===========================================================================

ROSCORE_PID=""
LOGGER_PID=""

cleanup() {
    echo ""
    echo "[CLEANUP] 清理残留进程..."
    [[ -n "$LOGGER_PID" ]] && kill -INT "$LOGGER_PID" 2>/dev/null
    [[ -n "$ROSCORE_PID" ]] && kill -TERM "$ROSCORE_PID" 2>/dev/null
    pkill -f "[r]osmaster" 2>/dev/null || true
    pkill -f "[m]etrics_logger" 2>/dev/null || true
    pkill -f "[r]osbag.*play" 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

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
    local idx="$2"
    local total="$3"
    local fname
    fname="$(basename "$bag" .bag)"

    # 方法名仅用于进度显示（CSV 文件名与 bag 完全一致，见 _csv_name）
    local prefix="${fname%%_real*}"
    local method="${prefix}_real"

    echo ""
    echo "=== [${idx}/${total}] 处理: ${fname} (method=${method}) ==="

    # 1. roscore + 轮询等待 master 就绪
    roscore > /tmp/roscore_exp_replay.log 2>&1 &
    ROSCORE_PID=$!
    if ! wait_for_master 15; then
        echo "[ERROR] ROS master 未在 15s 内就绪，跳过该 bag（日志: /tmp/roscore_exp_replay.log）"
        kill -TERM "$ROSCORE_PID" 2>/dev/null
        ROSCORE_PID=""
        return 1
    fi

    # 2. 仿真时钟
    rosparam set /use_sim_time true

    # 3. metrics_logger（跟随仿真时钟，从 ARM 上升沿开始记录；
    #    声呐几何参数与实物实验 launch 默认一致）
    rosrun sub_planning metrics_logger.py \
        _method:="$method" \
        _random_seed:="-1" \
        _csv_name:="${fname}" \
        _output_dir:="$OUTPUT_DIR" \
        _sonar_fov_deg:="$SONAR_FOV_DEG" \
        _sonar_max_range:="$SONAR_MAX_RANGE" \
        _sonar_min_range:="$SONAR_MIN_RANGE" \
        _desired_standoff:="$DESIRED_STANDOFF" \
        _standoff_tolerance:="$STANDOFF_TOLERANCE" \
        > "/tmp/metrics_exp_${fname}.log" 2>&1 &
    LOGGER_PID=$!
    sleep 2

    # 4. 只回放指标话题（过滤图像/TF，大幅加速大 bag 处理）。
    #    注意：bag 路径放在 --topics 之前；${METRICS_TOPICS} 为空格分隔
    #    多参数列表，必须不加引号展开（本机 rosbag --topics 语义）
    rosbag play --clock -r "$RATE" "$bag" --topics ${METRICS_TOPICS} \
        > "/tmp/bag_exp_play_${fname}.log" 2>&1

    # 5. 回放结束：SIGINT 优雅关闭 logger（触发 flush 落盘），再清理
    sleep 2
    kill -INT "$LOGGER_PID" 2>/dev/null
    sleep 3
    kill -TERM "$LOGGER_PID" 2>/dev/null
    kill -TERM "$ROSCORE_PID" 2>/dev/null
    sleep 2
    kill -KILL "$LOGGER_PID" "$ROSCORE_PID" 2>/dev/null
    wait 2>/dev/null
    LOGGER_PID=""
    ROSCORE_PID=""

    echo "[OK] ${fname} 完成，CSV 位于 ${OUTPUT_DIR}/"
}

TOTAL=${#BAGS[@]}
IDX=0
for bag in "${BAGS[@]}"; do
    IDX=$((IDX + 1))
    process_bag "$bag" "$IDX" "$TOTAL"
done

echo ""
echo "=== 全部完成 ==="
ls -lt "$OUTPUT_DIR/"*_real_*.csv 2>/dev/null | head -25

# 清理可能残留的 roscore（[r] 写法避免匹配自身 shell）
pkill -f "[r]osmaster" 2>/dev/null || true
echo "rosmaster 已清理"
