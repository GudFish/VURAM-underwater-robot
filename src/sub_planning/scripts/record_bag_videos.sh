#!/usr/bin/env bash
# ===========================================================================
# record_bag_videos.sh — 依次回放多个 bag，同时录制全屏画面与声呐图像视频
#
# 每个 bag 的流程：
#   1. 检查 bag 是否存在、是否含声呐图像话题
#   2. 启动声呐 debug 显示窗口（直接显示 bag 内录制的 /forward_sonar/debug_image，
#      不重跑检测器，避免与 bag 数据流重复发布导致画面闪烁）
#   3. 启动声呐视频录制器（image_view video_recorder，从话题数据流无损录制）
#   4. 启动 ffmpeg 全屏录制
#   5. roslaunch rosbag_play.launch 回放（RViz + belief 可视化随 launch 自动打开）
#   6. 回放结束后停止所有录制器，进入下一个 bag
#
# 输出（保存在 ~/video/）：
#   {bag}_screen.mp4 – 全屏录屏（含 RViz、声呐窗口）
#   {bag}_sonar.avi  – 声呐 debug 图像视频（来自 bag 数据流，与窗口画面一致）
#
# 用法：
#   ./scripts/record_bag_videos.sh              # 回放默认的三个 seed55 bag
#   ./scripts/record_bag_videos.sh foo bar      # 指定其他 bag（不带 .bag 后缀）
#
# 注意：
#   - 需要图形桌面会话（X11），DISPLAY 已设置
#   - 录屏抓取的是屏幕像素，录制期间窗口不可被遮挡
#   - RATE 保持 1.0 时视频即真实速度
#   - 录完后可用 ffmpeg 裁剪片头片尾，例如：
#       ffmpeg -i x_screen.mp4 -ss 00:05 -to 03:15 -c copy x_trim.mp4
# ===========================================================================

set -uo pipefail

# ---------- 环境 ----------
source "${HOME}/ardupilot_ws/devel/setup.bash"

VIDEO_DIR="${HOME}/video"
mkdir -p "${VIDEO_DIR}"

# ---------- 可调参数 ----------
RATE=10.0                            # 回放倍速（建议 1.0，视频即真实速度）
SCREEN_FPS=30                       # 全屏录制帧率
SCREEN_CRF=20                       # 屏幕视频质量（越小画质越好，18~23 合理）
SONAR_TOPIC="/forward_sonar/debug_image"    # 声呐录制话题（可改为 /forward_sonar/image 录原始图）
# 声呐源 10Hz；倍速回放时帧到达速度 = 10×RATE，写入帧率需同步，否则视频快慢失真
SONAR_FPS=$(awk "BEGIN{printf \"%.1f\", 10.0*${RATE}}")

# ---------- 要回放的 bag 列表（不带 .bag 后缀） ----------
if [ $# -ge 1 ]; then
    BAGS=("$@")
else
    BAGS=(
        "proposed_seed55_ts_lhy_Vostro_3910_883358_409789244248077547"
        "b2_ce_nmpc_seed55_ts_lhy_Vostro_3910_885952_1726322180234215296"
        "b1_bearing_only_seed55_ts_lhy_Vostro_3910_884746_4258926308438369397"
    )
fi

# ---------- 屏幕分辨率 ----------
if [ -z "${DISPLAY:-}" ]; then
    echo "[ERROR] DISPLAY 未设置，无法录屏（需要图形桌面会话）"
    exit 1
fi
SCREEN_RES=$(xdpyinfo -display "${DISPLAY}" 2>/dev/null | grep -oP 'dimensions:\s+\K[0-9]+x[0-9]+' | head -1)
SCREEN_RES="${SCREEN_RES:-1920x1080}"
echo "[INFO] 屏幕分辨率: ${SCREEN_RES} (DISPLAY=${DISPLAY}), 输出目录: ${VIDEO_DIR}"

# ===========================================================================
#  进程管理与清理
# ===========================================================================

ROSMASTER_PID=""
ROSLAUNCH_PID=""
FFMPEG_PID=""
SONAR_REC_PID=""
SONAR_VIEW_PID=""

stop_pid() {
    # 优雅停止单个进程：SIGINT（ffmpeg/ROS 节点均会落盘退出），等待后兜底 TERM
    local pid="$1"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        kill -INT "${pid}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${pid}" 2>/dev/null || return 0
            sleep 1
        done
        kill -TERM "${pid}" 2>/dev/null || true
    fi
}

cleanup() {
    echo ""
    echo "[CLEANUP] 清理残留进程..."
    stop_pid "${ROSLAUNCH_PID}"
    stop_pid "${FFMPEG_PID}"
    stop_pid "${SONAR_REC_PID}"
    stop_pid "${SONAR_VIEW_PID}"
    # 兜底清扫（[x] 写法防止匹配到自身 shell）
    pkill -f "[f]fmpeg.*x11grab" 2>/dev/null || true
    pkill -f "[v]ideo_recorder" 2>/dev/null || true
    pkill -f "[f]orward_sonar_image_viewer" 2>/dev/null || true
    pkill -f "[r]oslaunch.*rosbag_play" 2>/dev/null || true
    pkill -f "[r]osbag.*play" 2>/dev/null || true
    if [ -n "${ROSMASTER_PID}" ] && kill -0 "${ROSMASTER_PID}" 2>/dev/null; then
        kill -TERM "${ROSMASTER_PID}" 2>/dev/null || true
    fi
    echo "[CLEANUP] 完成"
}
trap cleanup INT TERM

# ---------- 预清理上一次中断的残留 ----------
cleanup

# ===========================================================================
#  启动 roscore（已有 master 则复用，不主动关闭他人的 master）
# ===========================================================================

if ! rosparam list &>/dev/null; then
    roscore > /tmp/roscore_record.log 2>&1 &
    ROSMASTER_PID=$!
    echo "[INFO] 启动 roscore (PID=${ROSMASTER_PID})"
    for _ in $(seq 1 30); do
        rosparam list &>/dev/null && break
        sleep 1
    done
    if ! rosparam list &>/dev/null; then
        echo "[ERROR] roscore 启动失败，详见 /tmp/roscore_record.log"
        exit 1
    fi
else
    echo "[INFO] 复用已运行的 rosmaster"
fi

# ===========================================================================
#  主循环：逐个回放 bag 并录制
# ===========================================================================

for bag in "${BAGS[@]}"; do
    echo ""
    echo "=============================================="
    echo "[RECORD] bag=${bag}"
    echo "=============================================="

    BAG_FILE="${HOME}/ardupilot_ws/bagfiles/${bag}.bag"
    if [ ! -f "${BAG_FILE}" ]; then
        echo "[WARN] 找不到 ${BAG_FILE}，跳过"
        continue
    fi

    # ---------- 检查声呐话题是否存在 ----------
    HAS_SONAR=false
    if rosbag info "${BAG_FILE}" 2>/dev/null | grep -q "${SONAR_TOPIC}"; then
        HAS_SONAR=true
        echo "[INFO] bag 含 ${SONAR_TOPIC}，将录制声呐视频与显示窗口"
    else
        echo "[INFO] bag 不含 ${SONAR_TOPIC}，仅录制全屏"
    fi

    # ---------- 1. 声呐显示窗口 + 声呐视频录制（在回放开始前挂上订阅） ----------
    if [ "${HAS_SONAR}" = true ]; then
        rosrun fls_detector forward_sonar_image_viewer.py \
            _image_topic:=${SONAR_TOPIC} \
            _window_name:="Forward Sonar Detection Debug" \
            _colormap:=true _display_scale:=1.4 \
            > /tmp/sonar_viewer_${bag}.log 2>&1 &
        SONAR_VIEW_PID=$!
        echo "[INFO] 声呐显示窗口启动 (PID=${SONAR_VIEW_PID})"

        rosrun image_view video_recorder image:=${SONAR_TOPIC} \
            _filename:=${VIDEO_DIR}/${bag}_sonar.avi _fps:=${SONAR_FPS} \
            > /tmp/sonar_rec_${bag}.log 2>&1 &
        SONAR_REC_PID=$!
        echo "[INFO] 声呐视频录制启动 (PID=${SONAR_REC_PID}, ${SONAR_FPS} fps)"
    fi

    # ---------- 2. 全屏录制 ----------
    ffmpeg -y -f x11grab -video_size "${SCREEN_RES}" -framerate "${SCREEN_FPS}" \
        -i "${DISPLAY}.0" -c:v libx264 -preset veryfast -crf "${SCREEN_CRF}" \
        -loglevel warning \
        "${VIDEO_DIR}/${bag}_screen.mp4" > /tmp/ffmpeg_${bag}.log 2>&1 &
    FFMPEG_PID=$!
    echo "[INFO] ffmpeg 全屏录制启动 (PID=${FFMPEG_PID}, ${SCREEN_RES}@${SCREEN_FPS})"

    # ---------- 3. 回放 bag（RViz 与 belief 可视化随 launch 自动打开） ----------
    echo "[INFO] 启动回放: roslaunch sub_planning rosbag_play.launch bag:=${bag} rate:=${RATE}"
    roslaunch sub_planning rosbag_play.launch bag:=${bag} rate:=${RATE} \
        > /tmp/roslaunch_play_${bag}.log 2>&1 &
    ROSLAUNCH_PID=$!
    wait "${ROSLAUNCH_PID}"
    ROSLAUNCH_RC=$?
    ROSLAUNCH_PID=""
    if [ "${ROSLAUNCH_RC}" -ne 0 ]; then
        echo "[WARN] 回放异常退出 (rc=${ROSLAUNCH_RC})，详见 /tmp/roslaunch_play_${bag}.log"
    else
        echo "[INFO] 回放结束"
    fi

    # ---------- 4. 停止所有录制器 ----------
    stop_pid "${FFMPEG_PID}"
    FFMPEG_PID=""
    stop_pid "${SONAR_REC_PID}"
    SONAR_REC_PID=""
    stop_pid "${SONAR_VIEW_PID}"
    SONAR_VIEW_PID=""

    # ---------- 5. 输出结果 ----------
    ls -lh "${VIDEO_DIR}/${bag}_screen.mp4" 2>/dev/null || echo "[WARN] 屏幕视频缺失"
    ls -lh "${VIDEO_DIR}/${bag}_sonar.avi" 2>/dev/null
    sleep 3
done

# ===========================================================================
#  收尾
# ===========================================================================

if [ -n "${ROSMASTER_PID}" ] && kill -0 "${ROSMASTER_PID}" 2>/dev/null; then
    kill -TERM "${ROSMASTER_PID}" 2>/dev/null || true
    echo "[INFO] roscore 已关闭"
fi

echo ""
echo "=============================================="
echo "  全部完成! 视频文件:"
echo "=============================================="
ls -lh "${VIDEO_DIR}/" | grep -E 'screen|sonar' || true
echo ""
echo "提示: 可用以下命令裁剪片头片尾（示例：保留 5s 到 195s）"
echo "  ffmpeg -i ~/video/<bag>_screen.mp4 -ss 00:05 -to 03:15 -c copy ~/video/<bag>_screen_trim.mp4"
