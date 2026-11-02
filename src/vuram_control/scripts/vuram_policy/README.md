# ROS 策略部署

默认快照为 `models/policy_v000012`，来自 `~/mujoco_auv_env/data/vnd/models/policy_v000012`。
模型权重目录受 `.gitignore` 忽略。更换机器或更新模型时，将同一快照的
`deploy.pt` 和 `spec.json` 一起复制到模型目录，或通过 `model_dir` 指向训练侧快照。

```bash
source /home/lhy/ardupilot_ws/devel/setup.bash
# 内部参考轨迹；连接已启动的 ArduSub/SITL，默认不自动解锁
roslaunch vuram_control policy_deploy.launch ref_family:=line
# 外部规划参考（RViz 目标点）
roslaunch vuram_planning policy_goal_deploy.launch
# 指定其他已发布快照
roslaunch vuram_control policy_deploy.launch \
  model_dir:=$HOME/mujoco_auv_env/data/vnd/models/policy_v000012
```

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

ROS 环境下可运行无需 ROS master 的回归测试：

```bash
python3 -m pytest -q src/vuram_control/tests/test_policy_contract.py
```
