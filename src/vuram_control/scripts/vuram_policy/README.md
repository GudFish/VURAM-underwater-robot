# ROS 策略部署与实机采集

`policy_deploy.launch` 默认持续运行固定的水平 figure8，每圈导出一个 H5，
并在圈边界切换到训练侧已经加载、校验和预热完毕的新策略。
默认读取 `~/mujoco_auv_env/data/vnd/models/latest.json`；没有新版本或新版本
不兼容时，继续运行当前有效策略。首次没有有效策略时回中等待。

```bash
source /home/lhy/ardupilot_ws/devel/setup.bash
# 连续 figure8 采集；默认连接已启动的 ArduSub/SITL，不自动解锁
roslaunch vuram_control policy_deploy.launch
# 实物飞控可通过 fcu_url 指定串口；轨迹参数可手动设置
roslaunch vuram_control policy_deploy.launch \
  fcu_url:=/dev/ttyUSB0:115200 \
  episode_len_s:=30.0 ref_speed:=0.5 figure8_ratio:=0.6
# 输出到其他目录；固定快照会关闭自动策略更新
roslaunch vuram_control policy_deploy.launch \
  data_dir:=$HOME/mujoco_auv_env/data/vnd/inbox \
  model_dir:=$HOME/mujoco_auv_env/data/vnd/models/policy_v000012
# 外部规划参考（RViz 目标点）
roslaunch vuram_planning policy_goal_deploy.launch
```

在另一个终端运行训练，收件箱和模型目录必须与采集端指向同一个数据根目录：

```bash
cd ~/mujoco_auv_env
python3 run_train.py --phase online
```

`learning_root` 默认 `~/mujoco_auv_env`，该目录必须提供 `auv_learning` 包以及
`config/env_config.yaml`、`config/task_config.yaml`。部署进程复用其中的
NumPy 推力分配、电机滞后、奖励函数与 H5 协议，不启动 MuJoCo。
ROS 的 Python 环境需要安装本包 `requirements.txt`，包括 `torch` 和 `h5py`。
`model_root` 默认其 `data/vnd/models`，`data_dir` 默认其 `data/vnd/inbox`；
设置了 `AUV_DATA_DIR` 的训练进程需显式将这两个参数指向对应目录。
独立运行节点的旧内部轨迹模式和外部规划模式仍可使用包内快照；模型权重
目录受 `.gitignore` 忽略，迁移时应复制同一版本的 `deploy.pt` 和 `spec.json`。

默认周期 30 秒、峰值速度 0.5 m/s、横纵幅值比例 `Ay/Ax=0.6`，轨迹自身轴向
全长约 3.06 m、全宽约 1.83 m。周期会对齐控制周期的整数倍。
开始采集时以当前位置为交叉点、当前航向为初始切线、当前深度为恒定深度。
参考横滚和俯仰为零，参考航向随切线变化。每次启动重新定位，圈内和圈间
均复用同一条曲线；圈末不减速，参考位置、速度和姿态连续。

只有桥接 ready、enabled、飞控已连接且解锁并处于 MANUAL、里程计有效时才
开始计圈。`arm_vehicle` 默认 false，等待手动解锁；准备期间不会消耗圈数。
默认等待 MAVROS 参数同步确认。单次参数服务等待超时会保持中立输出并以
`force_pull=false` 重试，复用进行中的下载，日志显示已收到的参数数量。
该阶段的总等待上限由 `param_wait_timeout` 设置，默认 180 秒；期间不会生成
轨迹或计圈，Ctrl+C 可退出。超过总上限仍未确认才停止启动。

使用 MAVROS 定位时，参数同步后会通过 `/mavros/set_message_interval` 请求
`ATTITUDE_QUATERNION`（31）和 `LOCAL_POSITION_NED`（32）到 30 Hz。
随后持续回中，实测里程计与 IMU 至少连续 1 秒：接收和源时间戳频率均需达到
策略频率的 95%（默认 19 Hz），最大相邻间隔不超过两个控制周期（默认 0.1 秒）。
一次偶然收到的新鲜消息不会启动采集；低频、停滞或时钟不一致时会继续等待，
日志显示实测速率、最大间隔和消息年龄。服务 ACK 成功也不会跳过实测检查。
`mavros_telemetry_rate`、`message_interval_service`、`imu_topic`、`odom_ready_window`
和 `odom_timeout` 可在 launch 设置；`odom_timeout` 默认仍为 0.2 秒。
其他定位源（例如 `odom_topic:=/dvl/odom`）默认关闭 `configure_mavros_rates`，
由定位源自行提供策略需要的频率，同样需要通过里程计实测检查。

**通过 MAVProxy 转发时，先在 MAVProxy 控制台执行：**

```text
set streamrate -1
```

也可在启动 MAVProxy 时传入 `--streamrate=-1`。MAVProxy 的默认行为会每 15 秒
重新请求 4 Hz 数据流，覆盖定位消息频率，造成运行中再次超时。`-1` 让 MAVProxy
停止干预流频率，由本 launch 设置所需频率；其他地面站也应避免覆盖这些设置。

使用 `sim_vehicle.py` 启动 SITL 时，它会同时启动 MAVProxy，应显式传入：

```bash
sim_vehicle.py -v ArduSub -l 31.284730,121.211700,0,0 --console \
  --mavproxy-args="--streamrate=-1"
```

每次重新启动 MAVProxy 都需要这个参数；仅修改 `SRx_POSITION` 仍可能被它的
周期性请求覆盖。已有 SITL 不必重启：在负责转发的 MAVProxy 命令终端执行
`set streamrate -1`，再重新启动 `policy_deploy.launch`，由 launch 重新请求 30 Hz。
`--console` 模块还会探测 `FLIGHT_INFORMATION`（264），旧版 ArduSub 可能打印
`No ap_message for mavlink id (264)` 和 `Unexpected command 511, result 2`。
这是不支持该消息的响应，不能据此判断定位消息（32）的速率请求失败。

采集模式 `inference_threads` 默认 1，分别在主控制线程与后台模型加载线程设置，
避免只配置后台线程、实际控制线程仍使用 OpenMP 原有线程数。新模型就绪前会
多轮预热 0..20 行有效历史，覆盖空、部分和完整历史输入。旧内部/外部参考模式
仍默认使用 2 个推理线程。

`control command interval exceeded 0.100s` 表示控制发布间隔超限，与里程计
超时是两个独立检查。日志会附实际间隔、当前 `phase` 及各阶段最近一次的
`last_wall/cpu` 耗时（毫秒）：`policy_inference` 是推理，`episode_record` 是名义
电机与奖励计算，`lap_boundary` 是圈边界处理，`wait_next_tick` 是等待下一拍。
阶段完成后的 wall 时间包含调度等待，CPU 时间仅统计该控制线程的 CPU 用时；
辅助推理线程的 CPU 时间不包含在内。
SITL 与显示程序并行时，可先运行
`roslaunch vuram_control policy_deploy.launch enable_rviz:=false enable_viz:=false`
检查降低显示负载后的时序。默认 0.1 秒保护继续保留，超时仍回中并丢弃未完成圈。

`setup_fc:=true` 会配置 VECTORED（`FRAME_CONFIG=1`）和既有参数设置。
采集过程中，失去上述条件、里程计超过 0.2 秒未更新、时间倒退、控制指令
间隔超过 0.1 秒或写盘失败都会回中停采。FC 状态心跳超过 3 秒未更新也会停止。
急停仍使用桥接节点的 `~enable` 服务；其使能状态额外锁存发布到
`enabled_topic`（默认 `/dof_cmd/enabled`）。修改 `dof_cmd_topic` 时，launch
会同时更新推理端、桥接端及关联状态话题。

H5 默认写入 `~/mujoco_auv_env/data/vnd/inbox/ep_<编号>_ros_<唯一后缀>.h5`。
后台队列容量为 2；完整临时文件关闭后才原子发布，训练器不会读取半写文件。
正常退出排空已完成圈，未完成圈丢弃；新启动会扫描 inbox、rollouts、rejected
继续编号。队列满时停止采集，不阻塞控制线程等待写盘。

每圈在 `/episode` 记录 schema v4 的 `time`、`state`、`action_norm`、
`thruster_cmd`、`motor_force_pre`、`ref`、`reward`、`done`。默认 600 次控制
转移对应 601 行状态，末行 `done=1`，只记录终点、不再次下发动作；下一圈
立即从同一实测状态开始。每圈重置策略历史及 `a_prev`，名义电机状态跨圈保留。
文件元数据包含该圈策略版本、固定轨迹参数和 `collection_source=ros_real_robot`。

`motor_force_pre` 是由策略动作经训练侧相同分配器和滞后模型计算的六路
**名义推力状态估计**，`thruster_cmd` 是对应静态推力的方向限幅归一化值，
两者都不是实测电机推力。文件以 `motor_force_source=nominal_allocator_estimate`
注明来源，飞控混控与名义分配之间的差异仍由实机状态转移反映。
额外数据集 `dof_pwm` 是发往 `/dof_cmd` 的六自由度 RC 通道指令，
顺序为 pitch、roll、throttle、yaw、forward、lateral，单位微秒；不是六路电机 PWM。
`action_applied` 表示本行是否下发了控制话题指令，终点行为 0；终点的 `dof_pwm`
保留上一条指令。`odom_stamp` 和 `command_stamp` 使用 ROS 时间，`time` 为圈内
名义控制时间，便于核查实机定位及控制时序。状态来自控制节拍处的最新有效里程计，
没有额外插值；实机应提供至少 20 Hz、与 ROS 时间一致的里程计。

推理输入为原始物理量，归一化已包含在模型中：

- 状态：`[v_body(6), quat_wxyz(4), pos_world(3)]`；世界系为 ENU，机体系为 FLU。
- 参考：`[pos_world(3), quat_wxyz(4), v_world(3), yaw_rate(1)]`。
  外部参考与定位必须采用同一个世界坐标系，参考线速度使用世界系。
- 新观测编码 `body_relative_tracking_v1`：位置误差和参考线速度分别为
  `R(q_state).T @ (pos_ref - pos)` 和 `R(q_state).T @ v_ref_world`。
  姿态误差仍为当前机体系下的相对旋转向量；参考偏航角速度保持原来的标量定义。
- 历史：最近 20 个原始 `[state13, action5]`，旧到新排列，不足时左侧补零。
  调用模型后才追加当前状态与本次动作；位置相对化在导出的编码器内部完成。
- 动作：归一化机体力/力矩 `[X,Y,Z,K,N]`，沿用现有 6DOF 通道映射。

仅接受 `spec.json` 和脚本化模型都声明 `body_relative_tracking_v1` 的快照。
缺少清单、缺少编码或编码不一致时拒绝启动。
模型校验和、输入维度和控制周期也在开始下发指令之前检查。

`twist_in_world:=false` 适用于 MAVROS/DVL 已经发布机体系 twist 的情况。
使用世界系 twist 的定位源（如部分 Gazebo pose_gt）时设为 `true`。
这个开关仅处理定位速度；观测始终使用当前机体系。

ROS 环境下可运行无需 ROS master 或机器人的回归测试。采集测试还会验证
训练端 H5 接收、窗口采样及各一次动力学和策略训练批次，全部使用临时目录：

```bash
python3 -m pytest -q src/vuram_control/tests
```
