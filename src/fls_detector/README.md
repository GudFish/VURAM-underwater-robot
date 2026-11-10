# BHQ-750D 前视声呐仿真

默认后端为 `raytrace`：三维最近交点、斜距成像、20°垂直视场、完整机器人／目标姿态及安装外参。旧 Python 回波斑块后端以 `legacy` 保留。统一跟踪入口 `target_tracking_simulation.launch` 默认使用 15 m，可通过 `sonar_range` 修改；与实机对照的入口使用 10 m 高频模式。两者继续发布 512×256 `mono8` 扇形图，图像左侧为传感器 +Y。

## 构建和启动

```bash
cd ~/ardupilot_ws
catkin_make --only-pkg-with-deps fls_detector -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash

# 10 m 对照：默认历史 494×256 原始栅格，加载实测背景，无中心亮条。
# 输入为现有机器人里程计、目标位姿、障碍物 MarkerArray，不启动飞控。
roslaunch fls_detector bhq750d_compare.launch

# 现有完整跟踪实验：默认新后端，原有控制和二维估计接口保持不变。
roslaunch sub_planning target_tracking_simulation.launch sonar_backend:=raytrace

# 实测背景在不同量程下均启用；量程同步传给仿真、检测、规划和指标。
roslaunch sub_planning target_tracking_simulation.launch sonar_backend:=raytrace sonar_range:=10

# 同一入口选择参考后端，或开启浅层多径。
roslaunch sub_planning target_tracking_simulation.launch sonar_backend:=legacy
roslaunch sub_planning target_tracking_simulation.launch sonar_multipath:=true

# 可选：选择静止近水背景参考。
roslaunch fls_detector bhq750d_compare.launch \
  noise_profile:=$(rospack find fls_detector)/config/bhq750d_noise_stationary_reference.yaml
```

`forward_sonar_sim.launch` 还提供 `config_file`、`config_overlay`、`noise_profile`、`max_range`、`min_range`、`fov_deg`、`publish_rate`、`multipath` 和原有话题映射参数。覆盖文件先加载；这些显式 launch 参数最后覆盖 YAML 中的同名项。`bhq750d_lf.yaml` 是低频覆盖文件；使用 120 m 等长量程时应降低发布频率，节点会拒绝超过声波往返时间上限的频率。

`bhq750d_compare.launch` 在 10 m 时默认加载 `bhq750d_historical_256.yaml`；所有量程均默认加载 `bhq750d_noise_experiment_10m.yaml` 的背景统计，中心亮条模板已清空。文件名中的 10 m 表示实测剖面的距离范围。统计方法、静止／运动数据区分、留出验证及重现命令见 [实测噪声记录](docs/noise_calibration/README.md)。

`forward_sonar_sim.launch` 仅在 `max_range=10` 时默认选择历史固定采样栅格，其他量程使用自动距离栅格。实测噪声通过 `noise_profile_range_m: 10.0` 按实际米数查表，量程改变时同一距离的均值／标准差保持一致；0–10 m 使用实测剖面，超过 10 m 沿用端点统计，属于外推。`target_tracking_simulation.launch` 提供 `sonar_range`、`sonar_min_range`、`sonar_fov_deg`，在仿真、检测、规划和指标中共享。声呐节点重启时清除旧私有参数。

输入消息必须位于与里程计相同的世界坐标系。节点跳过坐标系不匹配的目标／障碍物，避免静默混用坐标；需要跨坐标系时应在上游转换。目标必须有有效姿态四元数。Marker 支持 SPHERE（含椭球尺度）、CUBE、CYLINDER 的完整姿态、DELETE、DELETEALL 和有效期。

## 配置和成像约定

- 主配置为 `config/bhq750d_hf.yaml`。高频 1.2 MHz 对应标称 0.6°角分辨率和 2.5 mm 距离分辨率；低频 750 kHz 对应 1°和 4 mm。标称 512 波束与 0.25°波束间隔是采样规格，不等于独立分辨率单元。
- 默认 `range_resolution: 0.02` 是仿真距离栅格，**不是手册的物理距离分辨率**。默认 10 m 为 501×512 原始图，30 m 为 1501×512。物理距离扩散使用独立的 `range_sigma_m`；默认由频率模式选择。`num_range_bins: 0` 表示由量程及栅格步长推导。
- SDK 获取的 `beam_angles_deg` 数组、`range_resolution` 和 `num_range_bins` 优先用于复现实际采样配置。波束角严格递增，单位为度；角向响应使用实际波束角间距计算。覆盖文件中的采样栅格必须覆盖选择的量程，否则启动报错。
- 历史 bag 中看到的 494×256 只能确定图像形状，不能恢复未录制的 SDK 波束角及距离步长。历史覆盖文件的均匀角度和 `10/493 m` 步长仅是明确标注的近似。
- `sensor_xyz` 为声呐原点在机体系中的米制偏移；`sensor_rpy_deg` 为声呐坐标系相对机体的 roll、pitch、yaw。采用 +X 前、+Y 左、+Z 上的右手系。默认零外参未经过实机标定。节点发布世界系至 `sonar_frame` 的 TF，RViz 视场使用同一安装姿态。
- 所有回波都按**斜距**投影；多径按总传播路程的一半投影，方位来自接收声线。原始图行为距离递增、列为方位递增；扇形图由上至下为前向显示坐标递减。扇形显示坐标是声学投影，不代表目标真实水平位置。
- 二维跟踪仍以近似等深为前提；异深成像已经支持，异深目标的三维状态估计没有在本次扩展。现有场景中目标 z 和障碍物 z 必须与机器人处于同一世界系，不能把 RViz 中的示意高度当成声学深度。

## 几何、散射、抗混叠和多径

目标 STL 网格只在启动时构建一次局部 BVH，运动时只更新完整位姿。所有物体统一参与最近交点计算，因此遮挡和地面阴影来自几何关系，不再通过绘制固定长度暗条实现。未提供几何信息的池底、水面和墙面默认关闭，规划边界不会自动生成实体反射面。

创建自定义覆盖文件，例如：

```yaml
floor: {enabled: true, z: -2.0}         # 世界系 z，非“相对机器人深度”
water_surface: {enabled: true, z: 0.0}
sensor_xyz: [0.2, 0.0, 0.0]
sensor_rpy_deg: [0.0, -5.0, 0.0]       # 例子；实际安装姿态需测量
walls:
  - position: [12.0, 0.0, -1.0]
    size: [0.2, 8.0, 2.0]
    rpy_deg: [0.0, 0.0, 0.0]
```

材料包含漫反射、回散增强和镜面项，按几何扩散及吸收计算每一段路径的传播损失。回波先在线性相对功率域累计，再转 dB，应用 TVG、增益和 gamma。功率不是经过设备声源级标定的 W/m²；不能把默认亮度解释为绝对目标强度。

每个水平波束首先采样 `elevation_samples` 条垂直声线，再按命中／物体／距离变化补射，受 `max_rays_per_beam` 和 `refinement_depth` 限制。连续共面最近交点段采用平面方程解析重建距离单元，避免远处平坦地面需要一条声线对应一个距离单元；缺失命中、物体变化、法向或平面偏移变化会终止重建。曲面及多径使用补射和连续段去重。同一可见段的亮度不按声线数量简单累计，不同可见段可以共同贡献像素。有限声线预算仍可能漏掉小于采样尺度的结构。

`multipath:=true` 启用两种浅层路径，独立开关为 `wrt` 和 `prt`：

- WRT：`O → P2 → P1 → O`，检查发射源到 P2 的可见性。
- PRT：`O → P1 → P2 → P1 → O`，包含两次经过 P1 的反射损失。

两者沿 P1 的镜面方向寻找 P2，保留接收方向。没有加入 ARIS 分组发射滚动畸变，也没有未标定的串扰、相位干涉、衍射或高阶全局多径。BHQ 的单次发射波束形成机制不应套用 ARIS 的分组发射模型。

现有检测器的区域形态学核由固定 5×5 改为可配置 `region_morphology_size`，默认 3×3，以保留物理模型产生的窄距离回波；设为 5 可复现原来的区域开运算。检测器不读取实例真值。

## 输出和实机标定

| 话题 | 消息／含义 |
|---|---|
| `/forward_sonar/image` | `sensor_msgs/Image`，512×256 `mono8` 扇形图 |
| `/forward_sonar/image_raw` | `sensor_msgs/Image`，距离×波束 `mono8`，已经应用显示映射 |
| `/forward_sonar/direct_raw` | `32FC1`，无噪声、无 TVG 的直接回波相对线性功率 |
| `/forward_sonar/wrt_raw`、`prt_raw` | `32FC1`，独立的多径相对线性功率 |
| `/forward_sonar/instance_raw`、`instance_image` | `16UC1`，直接回波的主贡献实例；不对类别编号做双线性插值 |
| `/forward_sonar/metadata` | 锁存 `std_msgs/String` JSON，版本 1，保存实际采样几何、频率、增益、gamma 和种子等 |
| `/forward_sonar/render_stats` | `Float64MultiArray`：`[核心渲染毫秒, 包含消息构造和发布的整帧毫秒, 主声线数量]`；多径求交不计入主声线数 |

实例 0 为无直接回波，1 为目标，2 为池底，3 为水面，4–1003 为显式墙面，1024 起为按 Marker 的 `(namespace,id)` 稳定分配的障碍物编号。真实驱动也发布采样元数据，并在 SDK 缓冲锁内复制元数据与像素；两种图像共享时间戳和 ping 序号。录包入口已加入元数据及原始图像，`record_images` 开关仍控制图像体积。

`gamma` 使用 SDK 的 0–255 参数，150 对应约 0.588，0 和 255 对应 1。模拟器采用归一化 dB 到灰度的幂函数近似，并不声称复现了厂家未公开的全部处理链。`gain_assist` 默认关闭，打开时使用缓慢变化的显示增益近似，避免每帧立即拉伸。标定必须使用固定设备增益与已知显示设置。吸收系数、材料、方向图和噪声都是可配置近似，元数据明确标记 `calibrated: false`。

从无目标水体的波束 ROI 估计背景：

```bash
rosrun fls_detector calibrate_sonar_noise.py real.bag /tmp/noise.yaml \
  --beam-roi 20:60 --frames 100 --gamma-byte 150 --profile-range-m 10 \
  --metadata-output /tmp/acquisition.json

# 对含新元数据的 bag，还可导出实际采样配置。
rosrun fls_detector calibrate_sonar_noise.py real.bag /tmp/noise.yaml \
  --beam-roi 20:60 --device-config-output /tmp/device.yaml

roslaunch fls_detector bhq750d_compare.launch noise_profile:=/tmp/noise.yaml
```

ROI 必须由用户确认没有目标，程序的稳健裁剪不能自动辨认背景。工具将设备 mono8 的近似逆显示映射用于**相对 dB 统计**，输出逐距离均值、标准差及方位相关系数；不是绝对功率测量。实际显示范围与默认不同时应设置 `--db-min/--db-max`。翻转、动态增益或采样配置变化的数据应先按配置分段并统一方向。

也可重复 `--additional-bag` 和 `--beam-roi`，用 `--sampling uniform --frames 100` 跨时间取样，`--start-offset/--duration` 选择相对首个声呐消息的窗口，`--exclude-range-rows START:END` 排除人工识别的混响带，`--report` 保存统计与实际抽样来源。不连续 ROI 之间不计算邻波束相关；被排除距离行仅插值填补，不作为测量值计入报告。

`noise_profile_range_m` 表示距离剖面最后一个采样点对应的米数，起点为 0 m；均值和标准差在该物理距离轴上插值，超出范围保持端点。单元素数组表示常量。旧配置未提供此参数或设为 0 时保留按图像行归一化的兼容行为。校准工具优先从 SDK 距离步长推导物理轴；历史 bag 可用 `--profile-range-m` 显式指定。当前默认实测配置已提供 10 m 距离轴。

`noise_beam_bias_db` 是保留的可选方向干扰接口。当前运动及静止背景配置均显式设为 `[]`，不再生成中心亮条，重新运行校准脚本也不会恢复该模板。

每次实验保存完整私有参数，例如 `rosparam dump run.yaml /forward_sonar_sim`。只有在频率、量程、增益、显示映射和场景位姿一致后才做逐像素 PSNR/SSIM；未配准 bag 应比较距离亮度剖面、噪声、目标宽度和检测行为。现有实测资料不能替代安装外参、方向图和材料反射的标定，也不能据此宣称实机相似度已经验收。

## 验证

```bash
source devel/setup.bash
./devel/lib/fls_detector/test_sonar_renderer src/vuram_description/meshes/AUV.STL
python3 -m pytest -q src/fls_detector/tests/test_sonar_tools.py
python3 src/fls_detector/tests/ros_sonar_smoke.py
./devel/lib/fls_detector/benchmark_sonar_renderer src/vuram_description/meshes/AUV.STL 40 0
./devel/lib/fls_detector/benchmark_sonar_renderer src/vuram_description/meshes/AUV.STL 40 1
```

C++ 测试覆盖斜距、完整姿态、垂直视场、遮挡、阴影、解析形状、能量叠加、采样收敛、平坦地面补齐、浅层多径、噪声种子及真实 STL。ROS 测试启动独立本机 master，检查话题、元数据、现有检测器、遮挡／删除、俯仰和整帧性能，不接入飞控。基准使用完整目标网格、16 个障碍物及池底，显式报告 P95 是否满足 100 ms；不会在耗时超标时自动改变物理模型。OpenMP 可用时默认 4 个线程；性能数字需在实际运行平台重新测量。
