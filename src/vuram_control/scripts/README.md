# 轨迹跟踪控制与数据采集

策略控制和 PD 控制使用同一条固定水平 figure8，复用轨迹、可视化、控制保护和
H5 导出。每次启动用当前位置、航向和深度定位轨迹；运行中的各圈不重新定位。
两个 launch 都默认 `save_data:=true`，每完成一圈导出一个训练端 schema v4 H5。

```bash
# 策略控制：正常保存，圈边界切换已准备好的新模型
roslaunch vuram_control policy_deploy.launch
# PD 控制：正常保存，不需要策略快照
roslaunch vuram_control pd_deploy.launch
# 仅测试：继续控制与可视化，不保存 H5
roslaunch vuram_control policy_deploy.launch save_data:=false
roslaunch vuram_control pd_deploy.launch save_data:=false
```

默认周期 20 秒、峰值速度 1.0 m/s、`Ay/Ax=0.4`，可通过 `episode_len_s`、
`ref_speed`、`figure8_ratio` 修改。默认 20 Hz 控制，一圈 400 次转移、401 行
状态；终点行只记录状态，不重复发送动作。未完成圈不会导出。

保存开启时，local 模式默认输出到
`~/mujoco_auv_env/data/vnd/inbox`；LAN 模式默认输出到
`~/mujoco_auv_env/data/vnd/lan/outbox`，由独立同步进程上传，上位机持久化确认后
删除机载文件。两个 launch 都支持 `transport_mode:=lan`、`lan_server_url`、
`robot_id`、`lan_token_file` 和 `data_dir`。LAN 配置参见
[`vuram_policy/README.md`](vuram_policy/README.md)。

`save_data:=false` 不启动 H5 写盘，不创建采集输出目录，也不上传先前留在
outbox 的文件。策略控制在 LAN 模式仍同步模型，按原有规则在圈边界切换；
PD 不需要模型，因此测试模式不启动 LAN 同步进程。该开关不放宽里程计、
飞控状态、使能和控制时序检查。

两种控制器均用 `odom_source` 切换定位。默认 `mavros` 读取
`/mavros/local_position/odom`，可视化使用 `map`；设为 `dvl` 后读取 `/dvl/odom`，
可视化使用 `odom`，并启动一个 DVL 驱动及融合节点：

```bash
roslaunch vuram_control policy_deploy.launch odom_source:=dvl save_data:=false
roslaunch vuram_control pd_deploy.launch odom_source:=dvl save_data:=false
# 修改设备地址或融合方式，两种控制器使用相同参数
roslaunch vuram_control pd_deploy.launch odom_source:=dvl \
  dvl_ip:=192.168.1.95 dvl_mode:=mavros_fused save_data:=false
```

`dvl_ip` 默认 `192.168.1.95`，`dvl_mode` 默认 `dvl_fused`，用设备内置 DR
位姿校正，并按 DVL 速度分段外推位置。`dvl_odom_rate` 默认 `20.0`，控制其
里程计发布率。角速度由相邻有效 DR 四元数的相对旋转除以时间间隔估计，再做
低通滤波，表达在当前 FLU 机体系，单位 rad/s。它约随 5 Hz DR 更新，20 Hz
输出保持最近有效估计，不等同于 20 Hz 实测陀螺。
第一帧、重置、间隔过大或姿态跳变后角速度未知，输出零并标记大协方差。
位置和线速度仍按原有算法输出；未用该估计外推姿态。详细字段与参数见
[`dvl_odometry/README.md`](../../dvl_odometry/README.md)。

显式 `dvl_mode:=mavros_fused` 则融合 DVL 线速度与 MAVROS IMU 姿态、角速度、
加速度，固定输出 20 Hz，公共基础节点请求 MAVROS 遥测频率，控制器验证实际 IMU 频率；角速度来自 IMU。
默认 `dvl_fused` 不需要该 IMU 速率配置。
`dvl_raw` 仍不估计角速度，未知值发布为零并配大协方差。
`dvl_raw` 直接输出约 5 Hz 的设备 DR，不能满足当前默认 20 Hz 控制所需的
启动门控，即使调高 `dvl_odom_rate` 也不会提高原始模式频率。
融合方式与测量有效性说明见 [`dvl_odometry/README.md`](../../dvl_odometry/README.md)。
20 Hz 是融合里程计发布率，不代表声学量测率；应确认真实速度、姿态及设备数据
持续有效，不能仅以话题频率判断 DVL 工作正常。控制器同时监听 DVL 的连接、
重置、速度有效性及原始数据时间；断连、开始重置、速度无效或超过 `dvl_timeout`
（默认 1.0 秒）未收到有效速度时停止控制，使用设备 DR 的模式还检查 DR。
`dvl_fused` 另需 `/dvl/angular_velocity_valid=true` 才开始跟踪，运行中估计
失效会回中停采；短暂失效后又恢复也不自动继续本次控制。
两种控制器仍要求里程计接收与
源时间戳频率不低于控制频率的 95%，最大相邻间隔不超过两个控制周期，运行中
消息超过 `odom_timeout`（默认 0.2 秒）未更新会停止控制。

如果 DVL 已由另一终端启动，使用 `start_dvl:=false` 避免重复连接设备：

```bash
# 终端一：唯一的 DVL 驱动，发布 odom → base_link TF
roslaunch dvl_odometry dvl_odometry.launch mode:=dvl_fused publish_tf:=true
# 终端二：policy_deploy.launch 同样支持这组参数
roslaunch vuram_control pd_deploy.launch odom_source:=dvl start_dvl:=false save_data:=false
```

`odom_topic` 和 `viz_frame_id` 可显式覆盖；话题须为 `nav_msgs/Odometry`，
内置驱动的 `odom_frame` 同步使用 `viz_frame_id`。外部驱动也应提供一致的世界帧、
TF 及标准 `/dvl/connected`、`/dvl/resetting`、`/dvl/velocity_valid`、`/dvl/velocity`
话题；使用设备 DR 的模式还需 `/dvl/dr_valid`、`/dvl/dead_reckoning`。
`dvl_fused` 还须发布锁存的 `/dvl/angular_velocity_valid` 状态。
外部启动的 `mode` 应与控制 launch 的 `dvl_mode` 一致。
当前 MAVROS/DVL 均使用机体系速度，保持
`twist_in_world:=false`。DVL 模式关闭 RViz 的 `map → base_link` 广播，使用 DVL
发布的 `odom → base_link`，避免同一机器人出现两个 TF 父帧。
切换只影响机载控制器读取的里程计及显示，不会自动向飞控 EKF 注入 DVL。
飞控连接、解锁、MANUAL 和桥接使能条件仍需满足。
保存的 H5 属性还记录 `odom_source`、`dvl_mode` 及 `angular_velocity_source`，
其中 `dvl_dr_quaternion_difference_lowpass` 表示姿态差分滤波得到的角速度。

公共模块按用途放在 `scripts/` 下，各目录都是可导入的 Python 包：

| 目录 | 用途 |
| --- | --- |
| `trajectory/` | 固定水平 figure8 和原有参考轨迹生成 |
| `visualization/` | 参考轨迹与实际轨迹的 RViz 标记 |
| `data_collection/` | 训练数据协议、圈内记录、后台原子 H5 保存、独立 LAN 同步入口 |
| `control/` | PD 算法与入口、公共跟踪流程、状态与观测编码、推力接口及控制保护 |
| `utils/` | 坐标和四元数等公共数学工具 |
| `vuram_policy/` | 策略推理、后台加载与策略入口 |

两种控制器的 LAN 模式都启动 `data_collection/lan_sync_node.py`。
公共模块直接从上表对应的 Python 包导入。

`policy_deploy.launch` 和 `pd_deploy.launch` 均包含 `AUV_ctrl_6dof.launch`，
由它统一启动 MAVROS、六自由度桥接、选定的 DVL 栈及公共遥测配置节点
`control/mavros_telemetry_node.py`。沿用原有部署命令即可，单独运行其中一个入口，
无需再额外启动一份基础 launch。部署入口不会额外启动基础 launch 的边界显示。

`mavros_telemetry_rate`（默认 20 Hz）、`message_interval_service` 和
`configure_mavros_rates` 统一在 `AUV_ctrl_6dof.launch` 配置，策略和 PD 节点不再
发送设置频率的请求。公共节点等待桥接 ready 和 MAVROS 参数表同步，再请求
`RAW_IMU`（27）、`ATTITUDE_QUATERNION`（31）和 `LOCAL_POSITION_NED`（32），
自身不发布控制命令。ArduSub 的 RAW_IMU 为 MAVROS 的 `/mavros/imu/data` 补充
真实加速度；仅提高姿态消息频率不能使加速度变为有效。
默认在使用 MAVROS 定位或 `mavros_fused` 时启用，`dvl_fused` 不需要该节点。
修改此基础 launch 的默认遥测频率即可同时作用于两个部署入口；直接运行基础入口
时也可使用 `mavros_telemetry_rate:=30.0` 覆盖。

控制器仍独立检查实际消息频率、新鲜度和飞控状态，`check_mavros_imu` 仅控制
是否额外检查 IMU 流，默认根据定位源选择，与公共节点的频率请求开关分开。
公共参数同步/遥测请求函数位于 `control/mavros_startup.py`。

PD 的输出与策略相同，均为归一化机体系五轴力/力矩 `[X,Y,Z,K,N]`，
沿用六自由度通道映射。位置和速度误差在当前机体系计算；姿态跟踪控制
roll、yaw 两轴。通过 `pd_gains` 参数加载自己的增益文件：

```bash
roslaunch vuram_control pd_deploy.launch \
  pd_gains:=$HOME/robot_config/pd_gains.yaml \
  episode_len_s:=30.0 ref_speed:=0.5 figure8_ratio:=0.6
```

默认增益见 [`config/pd_gains.yaml`](../config/pd_gains.yaml)，数值来自训练侧
PD 的有效质量和惯量换算，需按目标机器人调试。`kp_xyz`、`kd_xyz` 是 XYZ
三轴物理增益，单位分别为 N/m、N/(m/s)；`kp_att`、`kd_att` 是 roll/yaw
两轴增益，单位分别为 Nm/rad、Nm/(rad/s)。命令按车辆推力分配配置归一化并限幅。
PD 保存的数据注明控制器类型及增益；六路推力仍为名义模型估计。

两种控制器都通过 `learning_root`（默认 `~/mujoco_auv_env`）读取兼容的
`auv_learning` 公共模块及 `config/env_config.yaml`、`config/task_config.yaml`。
PD 不加载策略快照、不进行神经网络推理；训练侧公共奖励模块会间接导入
Torch，因此仍需安装训练工程的相关依赖。即使关闭保存，车辆配置与推力分配器
仍用于命令归一化。控制期间不启动 MuJoCo。使用外部规划的策略接口保持原有方式：
`roslaunch vuram_planning policy_goal_deploy.launch`。
