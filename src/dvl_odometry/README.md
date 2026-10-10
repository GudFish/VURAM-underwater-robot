# dvl_odometry — Water Linked DVL A50 驱动与定位导航

驱动 Water Linked [DVL A50](https://docs.waterlinked.com/dvl/overview/)（TCP JSON API，json_v3.3），
发布原始 DVL 数据话题与 20 Hz 里程计（`nav_msgs/Odometry`），并提供 reset 服务：
重新从原点建立坐标系、重新开始定位导航。

- 环境：ROS1 Noetic（catkin），Python 3
- 硬件连接：DVL 网口直连机载电脑，TCP server **端口 16171**；本机 DVL 静态 IP **192.168.1.95**
  （设备兜底 IP 为 192.168.194.95，支持 DHCP + mDNS `waterlinked-dvl`）

## 快速上手

真机（DVL 已上电、已连网）：

```bash
roslaunch dvl_odometry dvl_odometry.launch
```

无硬件联调（模拟器 + 驱动，另开两个终端）：

```bash
roslaunch dvl_odometry dvl_simulator.launch
roslaunch dvl_odometry dvl_odometry.launch ip:=127.0.0.1 publish_tf:=true
```

三条 AUV 直控链路已内置定位源切换（默认原行为不变）：

```bash
# DVL 数据只发布话题（/dvl/odom 等），TF 树仍由 mavros 的 map→base_link 提供
roslaunch vuram_control AUV_ctrl_custom.launch use_dvl:=true
# DVL 定位接管：odom→base_link TF 由 DVL 广播、mavros TF 关闭、RViz 用 odom 固定帧
roslaunch vuram_control AUV_ctrl_custom.launch odom_source:=dvl
```

## 话题

| 话题 | 类型 | 频率 | 说明 |
|---|---|---|---|
| `/dvl/odom` | nav_msgs/Odometry | 20 Hz（`~odom_rate`） | frame_id=`odom`，child=`base_link`；位姿 = DR 锚点 + 速度外推，twist = 转换到 ROS 系的速度 |
| `/dvl/velocity` | TwistWithCovarianceStamped | 2–15 Hz（随水深） | **原始 DVL 机体系**（z 向下，未转换），frame_id=`dvl_link` |
| `/dvl/altitude` | sensor_msgs/Range | 同上 | 对底高度（沿 DVL z 轴），0.05–50 m |
| `/dvl/fom` | std_msgs/Float32 | 同上 | 速度优度 figure of merit（m/s） |
| `/dvl/connected` | std_msgs/Bool | 状态变化时 | TCP 连接状态 |

## 服务

| 服务 | 类型 | 说明 |
|---|---|---|
| `/dvl_odometry/reset` | std_srvs/Trigger | 发送 `reset_dead_reckoning`：位置清零、roll/pitch/yaw 清零，按当前机体系**重新从原点建立坐标系**（约 50 ms 生效）。成功后 odom 位姿跳变到原点——这是 REP-105 odom 漂移帧的合法语义，下游滤波需容忍跳变或自行重启 |
| `/dvl_odometry/calibrate_gyro` | std_srvs/Trigger | 陀螺标定（**DVL 必须静止**，最长 15 s）。A50 标定后 yaw 漂移可降至 0.1–0.3°/min |

```bash
rosservice call /dvl_odometry/reset "{}"
```

## 20 Hz 输出机制

DVL 内置 dead-reckoning（DR，内部 IMU/AHRS + 声学速度卡尔曼滤波）报告固定 **5 Hz**（设备输出
速率，无法提高）。节点以 20 Hz 发布 odom：每次收到 DR 报告即更新"传播锚点"（位置+姿态），
两次报告之间用**最新速度报告**（2–15 Hz）按锚点姿态前向推算：

```
p_ros(t) = anchor_pos + R(anchor_quat)·v_ros·(t_now − t_anchor)
```

- 速度过期（> `~vel_timeout`）或 `velocity_valid=false`：不外推，位姿停在锚点，协方差放大；
- DR 过期（> `~dr_timeout`）：持位并逐秒放大协方差（`~cov_growth_pos`）；
- 未收到过 DR 报告：不发布 odom（不发误导性的全零）。

**注意：20 Hz 中仅 5 Hz 为真实量测、其余为外推，下游传感器融合应按 5 Hz 量测率对待。**

## Reset 语义与启动行为

- 启动时若 `~reset_on_start=true`（默认），**首次**连上 DVL 后自动 reset 一次；
  每次节点运行仅一次，重连不重发（实验中节点中途重启会清掉当前坐标系，属预期语义）。
- DVL 无底锁时位置仅由加速度推算、FOM/std 会显著增大；空气中/无水时 DR 大幅漂移，
  应在水中、有底锁、陀螺已标定后使用。

## 坐标系约定与转换

| | DVL 机体系 | ROS（REP-103） |
|---|---|---|
| x | 向前（A50 上 LED 标 +x） | 向前 |
| y | 向右（starboard） | 向左 |
| z | **向下**（朝向换能器） | 向上 |

转换等效绕 x 轴被动旋转 180°（R = diag(1,−1,−1)），已按共轭推导验证：

| 量 | DVL → ROS |
|---|---|
| 位置 | (x, −y, −z) |
| 速度 | (vx, −vy, −vz) |
| 姿态 | roll 不变，pitch → −pitch，yaw → −yaw |
| 协方差 | C_ros[i][j] = s_i·s_j·C_dvl[i][j]，s = (1,−1,−1) |

`/dvl/odom` 的位姿与 twist 均已转换到 ROS 系；`/dvl/velocity` 为原始 DVL 机体系。
若在 DVL 上配置了 `mounting_rotation_offset`（安装偏置），速度输出已转至车体系，
**勿在节点中二次转换**。校验方法：纯直航时 `/dvl/velocity` 的 vy≈vz≈0。
节点也可代为下发配置：`apply_config:=true speed_of_sound:=1475.0 mounting_rotation_offset:=0.0`。

## TF 切换（避免与 mavros 冲突）

默认 `publish_tf=false`：节点只发布 `/dvl/odom` 消息，**不广播 TF**——其他节点直接订阅
该话题即可，与现有 `map→base_link`（mavros `tf_baselink_map_node`）不冲突。

需要 DVL 定位在 TF 树/RViz 中可用时（三条 AUV_ctrl launcher 已内置）：

- `odom_source:=dvl`：关闭 `tf_baselink_map_node`，DVL 节点广播 `odom→base_link`，
  RViz 加载 `ardusub_display_dvl.rviz`（固定帧 odom）；
- 或单独启动 `dvl_odometry.launch publish_tf:=true` 并自行停用 `tf_baselink_map_node`。

## 参数（节点私有 `~`，均有默认值）

| 参数 | 默认 | 说明 |
|---|---|---|
| `ip` | 192.168.1.95 | DVL 地址 |
| `port` | 16171 | TCP 端口 |
| `reconnect_min_delay` / `reconnect_max_delay` | 1.0 / 30.0 | 指数退避重连 1→2→4→…（s） |
| `reset_on_start` | true | 首次连接后自动 reset（每节点一次） |
| `apply_config` | false | 连接时下发 set_config |
| `speed_of_sound` | 0.0 | 0 表示不下发 |
| `mounting_rotation_offset` | 0.0 | 0 表示不下发（度） |
| `odom_rate` | 20.0 | odom 发布频率（Hz） |
| `publish_tf` | false | 广播 odom→base_link |
| `publish_z` | true | false 时将 odom 位置 z 压零（应对安装倾斜时 z 误差放大：官方注明 5° 倾斜/100 m 水平 → z 误差 ±9 m） |
| `odom_frame` / `robot_frame` / `dvl_frame` | odom / base_link / dvl_link | 帧名 |
| `vel_timeout` | 1.0 | 速度新鲜度超时（s） |
| `dr_timeout` | 3.0 | DR 新鲜度超时（s） |
| `reset_timeout` | 2.0 | reset 命令 ACK 等待（s） |
| `gyro_calib_timeout` | 20.0 | 陀螺标定 ACK 等待（s） |
| `dr_z_std` / `dr_rp_std` / `dr_yaw_std` | 1.0 m / 3° / 5° | 协方差先验 |
| `cov_growth_pos` | 0.01 | 数据过期时位置协方差增长（m²/s） |

## 已知限制

- yaw 非绝对且缓慢漂移（标定后 0.1–0.3°/min）：`odom` 是 REP-105 漂移帧，**不可当 `map` 用**；
- odom twist 角速度恒为 0（速度报告无角速度，姿态由 5 Hz DR 刷新）；
- reset 后位姿跳变到原点，下游滤波需容忍跳变；
- DVL 安装倾斜会在 z 方向放大位置误差（可 `publish_z:=false` 压零）。

## 联调与测试

- 模拟器：`roslaunch dvl_odometry dvl_simulator.launch`（无 rospy 依赖，也可
  `python3 scripts/demos/dvl_simulator.py` 直接跑）；向 16171 端口发送
  `{"command":"sim_dropout"}` 可主动断连，验证驱动重连。
- 验证清单见包开发计划：`rostopic hz /dvl/odom` ≈ 20 Hz、reset 后跳回原点、
  kill/重启模拟器触发退避重连且不重发 reset、`rosrun tf tf_echo odom base_link`。
