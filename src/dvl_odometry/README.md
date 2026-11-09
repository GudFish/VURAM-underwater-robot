# Water Linked DVL A50：速度驱动与三种里程计

ROS1 Noetic / Python 3。默认连接 `192.168.1.95:16171`，每套 launch 仅建立一个 TCP 连接。

## 设备能力与频率

依据 [A50 TCP JSON 协议](https://docs.waterlinked.com/dvl/dvl-json-protocol/)：

- 对底速度 `velocity` 每次计算完成即输出，随离底高度自适应，约 **2–15 Hz**；配置中没有设置输出 Hz 的字段。
- `trigger_ping` 可用于声学同步（关闭自动声学后软件触发），受实际测量周期和最多 15 条排队限制，不保证任意指定 Hz。本包保留正常自动测量，不将 ROS 频率参数下发为硬件触发。
- 内置 `position_local` 预计 **5 Hz**。它的字段包含位置和姿态，没有线速度或角速度字段；直接里程计的线速度取独立速度报告。
- [A50 不输出原始 IMU](https://docs.waterlinked.com/dvl/overview/)，但内置里程计提供来自 IMU/AHRS 的姿态角。其航向相对启动/重置时刻，存在漂移，不是绝对罗盘航向。[内置航位推算说明](https://docs.waterlinked.com/dvl/dead-reckoning/)

`velocity_rate` 是 ROS 有效速度发布**上限**：0 为每个有效量测发布一次；大于 0 时限频。
例如设置 20 Hz、设备实际 2 Hz，仍只发布约 2 Hz，并给出诊断，不补样、不重复旧速度。
低频设置时应相应增大 `vel_timeout`，否则里程计会正确地将间隔过长的速度判为过期。

## 安装与启动

```bash
sudo apt-get install ros-noetic-robot-localization
cd /home/lhy/ardupilot_ws
catkin_make --only-pkg-with-deps dvl_odometry
source devel/setup.bash

# 默认：DVL 自身融合，20 Hz
roslaunch dvl_odometry dvl_odometry.launch
# 原生内置里程计，约 5 Hz
roslaunch dvl_odometry dvl_odometry.launch mode:=dvl_raw
# MAVROS IMU + DVL 速度 EKF，20 Hz；需另行启动 MAVROS
roslaunch dvl_odometry dvl_odometry.launch mode:=mavros_fused
```

| mode | 输入/算法 | 发布行为 |
|---|---|---|
| `dvl_raw` | 内置位置、姿态直接转换；线速度取新鲜 DVL 速度 | 每份有效 DR 发布一次，约 5 Hz；忽略 `odom_rate` |
| `dvl_fused`（默认） | 内置位置/姿态为锚点，按速度到达时刻分段外推 | `odom_rate>5` 时定频预测；`odom_rate<=5` 时仅对 DR 限频 |
| `mavros_fused` | `robot_localization` EKF，DVL 三轴速度 + MAVROS 姿态/角速度/加速度 | 固定 20 Hz；不融合相关的 DVL 内置位姿 |

DVL 自身融合的高频帧会短时保持最近有效速度，不能观测量测间隔内的真实加速。
20 Hz 发布率不等于 20 Hz 声学量测率。协方差按速度、姿态误差和过程噪声传播，同一速度样本的不确定度不因提高发布率而虚假缩小。

四个现有入口 `AUV_ctrl_custom.launch`、`AUV_ctrl_6dof.launch`、`AUV_ctrl_thruster.launch`、
`policy_goal_deploy.launch` 均支持 `dvl_mode`、`dvl_odom_rate`，例如：

```bash
roslaunch vuram_control AUV_ctrl_custom.launch odom_source:=dvl dvl_mode:=mavros_fused
# 仅增加 DVL 话题，保留原有 MAVROS TF：
roslaunch vuram_control AUV_ctrl_custom.launch use_dvl:=true dvl_mode:=dvl_raw
```

## 数据、坐标与时间

| 话题 | 类型 | 内容 |
|---|---|---|
| `/dvl/odom` | `nav_msgs/Odometry` | 所选模式的位姿、机体速度和协方差 |
| `/dvl/velocity` | `geometry_msgs/TwistWithCovarianceStamped` | **ROS FLU** 机体线速度，frame=`base_link` |
| `/dvl/velocity_raw` | 同上 | 原始设备 FRD 速度，frame=`dvl_link`，与转换后速度同步限频 |
| `/dvl/dead_reckoning` | `geometry_msgs/PoseWithCovarianceStamped` | 有效内置位姿，ROS 轴约定，约 5 Hz |
| `/dvl/velocity_valid`、`/dvl/dr_valid` | `std_msgs/Bool` | 有效性（含断连/超时），锁存；true 不是新量测 |
| `/dvl/altitude` | `sensor_msgs/Range` | 对底高度，m |
| `/dvl/fom` | `std_msgs/Float32` | 速度优度，m/s |
| `/dvl/connected`、`/dvl/resetting` | `std_msgs/Bool` | 连接/重置状态，锁存 |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 实际量测频率、发布上限、新鲜度、预测/惯性降级状态 |

**迁移注意：旧版 `/dvl/velocity` 是 FRD；现在该话题是 FLU，原有 FRD 使用者应改订阅 `/dvl/velocity_raw`。**
不发布无效/非有限速度；`velocity_water` 为对水速度，不作为对底量测送入估计器。
未知角速度、失效速度使用大方差，不能把零值理解为确定静止。

A50 FRD → ROS FLU：位置/速度 `(x,-y,-z)`，姿态 `(roll,-pitch,-yaw)`，协方差 `S C Sᵀ`，`S=diag(1,-1,-1)`。
安装航向偏置由 DVL 的 `mounting_rotation_offset` 对齐车体；主机不再施加该航向旋转，但 FRD→FLU 转换始终执行。
**默认 DVL 输出轴已对齐车体，DVL 测量原点与 base_link 重合**；未补偿非零安装杆臂或任意安装倾角，实际安装须校准或增加外参处理。
直接/DVL 融合的 `odom` 方向由 DVL 重置时刻定义；MAVROS 模式的 `odom` 原点为滤波启动位置，方向遵循 MAVROS ENU，不保证切换模式后沿用同一参考系。

DVL 消息使用主机接收时间，不假定设备时钟同步；设备 `time_of_validity` 保留在诊断中。
MAVROS 保留自身消息时间戳，不进行第二次 ENU/FLU 转换。真实系统应保证主机和 MAVROS 时间一致，网络/处理延时仍影响融合精度。

## 初始化、失锁、重置和 TF

- DVL 两种模式等待有效内置位姿；MAVROS 模式等待新鲜 IMU 和首次有效 DVL 速度，初始化前不发布全零里程计。
- DVL 融合在速度/姿态失效后停止外推，持有最后预测位置并增长不确定度；仍接受新内置位姿校正，不退回旧锚点。直接模式无新量测即无新输出。
- MAVROS 模式 DVL 失锁/断连后继续用 IMU 进行惯性推算，仍输出 20 Hz 并发出 WARN；位置误差会随惯性偏置、姿态误差和时间累积，没有绝对位置校正。恢复 DVL 后恢复速度校正，不清零位置。
- MAVROS IMU 也失效时暂停公共 odom 和 TF。恢复时将 EKF 重新置于最后公开位姿、增加位置不确定度，速度重新估计；缺失区间不补积分，内部隐藏预测不会被直接公开。
- `/mavros/imu/data` 必须包含有效姿态、角速度、加速度，frame 须为配置的 `robot_frame`。EKF 移除重力；输入协方差设有保守下限（姿态 0.03 rad、角速度 0.01 rad/s、加速度 0.1 m/s²）。航向是否受罗盘约束由 ArduSub AHRS/EKF 配置决定，本包不重复融合 `compass_hdg`。

| 服务 | 类型 | 行为 |
|---|---|---|
| `/dvl_odometry/reset` | `std_srvs/Trigger` | DVL 模式重置设备并清除预测缓存；MAVROS 模式清零滤波位置、保留 ENU 姿态 |
| `/dvl_driver/reset_dead_reckoning` | `std_srvs/Trigger` | 底层设备重置，不用于 MAVROS 滤波位置复位 |
| `/dvl_odometry/calibrate_gyro` | `std_srvs/Trigger` | 标定 A50 陀螺；设备须静止，最多等待 20 s |

DVL 模式首次连接默认 reset 一次；重连不自动 reset。ACK 后丢弃至少 0.1 s 报告以等待硬件清零，不立即发布虚构零位姿。
显式 reset 会改变局部参考系并使位置跳变；运行中的下游控制/滤波须相应处理。
TF 默认 `publish_tf=false`；打开后仅所选模式广播 `odom→base_link`，EKF 内部 TF 始终关闭。
使用 `odom_source:=dvl` 的上层入口会关闭原有 MAVROS 定位 TF，避免双父节点。

## 主要 launch 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `mode` | `dvl_fused` | 三种模式之一，非法值启动时报错 |
| `ip` / `port` | `192.168.1.95` / `16171` | TCP 地址 |
| `velocity_rate` | `0` | 有效速度发布上限 Hz；0 为原生事件发布 |
| `odom_rate` | `20` | DVL 融合输出 Hz；直接模式忽略，MAVROS 固定 20 |
| `vel_timeout` / `dr_timeout` / `imu_timeout` | `1.0` / `1.0` / `0.5` | 新鲜度秒数 |
| `imu_topic` | `/mavros/imu/data` | 飞控融合姿态及惯性数据 |
| `odom_topic` | `/dvl/odom` | 公共里程计话题 |
| `odom_frame` / `robot_frame` / `dvl_frame` | `odom` / `base_link` / `dvl_link` | 帧名称 |
| `publish_tf` / `publish_z` | `false` / `true` | TF、位置 z；publish_z=false 仅将公开位置 z 压零 |
| `reset_on_start` | `true` | 首次连接一次设备 reset |
| `reconnect_min_delay` / `reconnect_max_delay` | `1` / `30` | 单调时钟指数退避，秒 |
| `apply_config` | `false` | 下发声速和安装航向偏置 |
| `speed_of_sound` | `0` | 0 不下发；否则 1000–2000 m/s |
| `mounting_rotation_offset` | `0` | 0–360°，apply_config=true 时也可显式设回 0 |

节点私有参数还包括驱动 `reset_timeout`、`gyro_calib_timeout`、`velocity_fallback_std`、`dr_z_std`、`dr_rp_std`、`dr_yaw_std`，及 DVL 融合 `cov_growth_pos`；见实现中的默认值。
缺失/非法速度协方差以保守的 0.05 m/s 标准差代替，不解释为零误差。

## 模拟与回归

```bash
# 终端一；MAVROS 模式需 publish_imu=true。不得与真实 MAVROS 同话题同时运行。
roslaunch dvl_odometry dvl_simulator.launch publish_imu:=true
# 终端二
roslaunch dvl_odometry dvl_odometry.launch ip:=127.0.0.1 mode:=mavros_fused
rostopic hz /dvl/velocity
rostopic hz /dvl/odom

# 模拟器中的角度/角速度参数采用 rad、rad/s，支持 roll/pitch 和加速度。
# 默认 0.3 m/s、0.1 rad/s 转弯；静止倾斜测试：
roslaunch dvl_odometry dvl_simulator.launch publish_imu:=true speed:=0 yaw_rate:=0 roll:=0.2 pitch:=-0.3
```

TCP 与 IMU 模拟器使用同一个确定性轨迹和时间原点；机体速度与世界位置分开计算，IMU 含正确重力及转弯向心加速度。
保留无需 rospy 的独立 TCP 程序：`rosrun dvl_odometry dvl_simulator.py`。

通过 TCP 发送测试命令，支持 `sim_invalid`、`sim_pause_velocity`、`sim_pause_dr`，参数 `seconds` 默认 2。
`sim_dropout` 断开全部客户端；`sim_malformed` 注入坏 JSON/非法量测。可由第二条连接发命令，不改变运动轨迹：

```bash
printf '%s\n' '{"command":"sim_invalid","parameters":{"seconds":5}}' | nc -q 1 -w 1 127.0.0.1 16171
rosservice call /dvl_simulator/imu_enabled 'data: false'
rosservice call /dvl_simulator/imu_enabled 'data: true'
python3 -m pytest -q src/dvl_odometry/tests
```

回归会启动独立端口的 ROS master、模拟器和真实 EKF，不连接飞控。
真机验收还需在水中检查：实际速度/DR 频率、三轴速度符号、静止重力补偿、安装外参、失锁诊断、恢复底锁及 reset 后参考系变化；模拟测试不能代替水中精度测量。
