[English](README.md) | 中文

# dronesift — 大疆航拍素材自动切片(只留稳定 & 连续运镜)

输入一个 DJI 视频(MP4,可带同名 SRT),输出:

- `<name>_segments.json`:每段起止时间/帧、运镜标签、抖动量、速度、云台角速度等
- `<name>.edl`:CMX3600 EDL,可直接导入达芬奇 / Premiere / FCP,原片不动
- `<name>_timeline.png`:时间线诊断图(抖动、运动、速度、云台角速度,绿色=保留,红色=剔除)
- `<name>_NN_<label>_<t>s.mp4`:切好的片段(默认 `--mode copy` 关键帧对齐无损)
- 中间缓存 `<name>_telemetry.csv`(逐帧遥测)、`<name>_flow.csv`(逐帧光流)

## 用法

```bash
cd dronesift
uv sync
uv run python -m dronesift /path/to/DJI_xxx.MP4                 # 默认:无损切片到 <video>_cuts/
uv run python -m dronesift /path/to/DJI_xxx.MP4 --mode encode   # 精确到帧,HEVC 10-bit(videotoolbox / libx265)
uv run python -m dronesift /path/to/DJI_xxx.MP4 --mode prores   # ProRes 422 HQ,给达芬奇二次调色
uv run python -m dronesift /path/to/DJI_xxx.MP4 --mode none     # 只出 EDL/JSON/图,不切文件
uv run python -m dronesift x.MP4 --max-shot-s 8                 # 追加切成片级镜头(≤8s,输出到 shots/)
uv run python -m dronesift x.MP4 --min-len-s 2 --jitter-px 0.8 --acc-h 1.5   # 放宽阈值
```

重复运行会复用缓存的 telemetry/flow CSV,只重新分段(几秒钟);调阈值很快。

## 原理

### 1. 数据来源:MP4 内嵌 `djmd` 轨,而不是 SRT

Air 3S(以及 Air 3 / Mini 4 Pro / Mavic 3 等新固件)MP4 里有 `djmd` 数据轨,protobuf 编码(`dvtm_Air3s.proto`),
逐帧(60Hz)带有:

| 字段 | 内容 | 更新率 |
|---|---|---|
| fc_frame_meta.velocity | 飞机速度 NED 三轴 m/s(0.1 m/s 分辨率) | ~10 Hz |
| fc_frame_meta.euler | 飞机 pitch/roll/yaw(罗盘航向) | ~10 Hz |
| fc_frame_meta.gps / rel_alt | 经纬度、绝对/相对高度 | ~10 Hz |
| gimbal_frame_meta.euler / quaternion | 云台 pitch/roll/yaw(即相机实际指向) | ~30 Hz |
| camera_frame_meta | ISO、快门、EV、色温 | 60 Hz |

`dronesift/djmd.py` 按推断出的字段号解析(schema 见文件头注释;`pbraw.py` 是无 schema 的通用解码器,用来对新机型重新推断)。
SRT 只有 GPS + 曝光,没有云台角度和速度;只在没有 `djmd` 时作为退化方案(`srt.py`,GPS 差分求速度)。
`dbgi` 轨是 ISP 调试数据,没有 IMU,不用。

**注意**:云台 yaw 相对罗盘航向有一个固定偏置(这段素材约 -23°,飞机起飞后快速偏航时云台 yaw 没跟上),所以
相机指向用飞机航向(云台 pan 机械范围很小),云台 yaw 只用来算角速度/角加速度。

### 2. 画面分析:光流估计全局运动

`flow.py`:ffmpeg(videotoolbox 硬解,其它平台自动回退软解)把 4K 缩到 640×360 灰度,逐帧 Shi-Tomasi 角点 + LK 光流 + RANSAC 相似变换,
得到每帧的平移 dx/dy、旋转 da、缩放 ds,以及拟合残差、清晰度(Laplacian 方差)。75 秒 4K60 约 2 分钟。

### 3. 融合判定(`segment.py`)

遥测按 `pts` 时间戳重采样到视频帧上(丢帧/时轨偏移不会错位;遥测未覆盖的帧一律判为不保留)。
每帧必须同时满足才"保留"(画面指标都按 640 px 宽、每秒为单位,与帧率无关):

| 指标 | 含义 | 默认阈值 |
|---|---|---|
| `hf_flip` | **感知抖动门控**(所有画面否决的前提):高频光流在 0.1s 内方向反转的帧占比。人眼觉得"抖"的是偏离轨迹又回来的**振荡**;再猛但从不回头的运动(快速甩镜、大师模式的抬镜/下摇)都是连续手势,不该剔。实测:甩镜类动作反转率 ≤0.22,拨杆抖动 ≥0.24。设为 0 恢复旧的"只看幅度"行为 | > 0.25 |
| `motion_rev_amp` | **慢速分量反向**:平移/变焦分量(旋转豁免——环绕 CW→CCW 是编排动作)在镜头内**变号**(先左后右的摇、推近中叠了反向漂移),±2.5s 包络内两侧幅度均超阈值即剔。人对"多种变动叠加且互相矛盾"远比对"单一方向的剧烈变化"敏感 | 25 px/s |
| `jitter_px` | 特征点**中位数**光流减去 0.15s 低通后的高频残差 RMS,仅在 `hf_flip` 判定为振荡时生效。用中位数而不是 RANSAC 全局拟合,是因为近距离视差会让拟合在不同深度层之间跳 | 30 px/s |
| `jitter_spike_px` | **短促抖动爆发**(手工拨杆瞬间的抖一下),同样要求满足 `hf_flip` 振荡门控。0.5s 中值滤波会把 <0.3s 的爆发稀释到 `jitter_px` 以下,此指标用 0.15s 短窗 RMS 抓它;要求连续 ≥3 帧超阈值(单帧光流毛刺不触发),命中后前后各扩 0.35s——豁口大于合并间隔,真正把段切开 | 45 px/s |
| `motion_acc` | 1s 低通后画面运动速度的变化率(运镜起/停/变速) | 60 px/s² 或当前画面速度的 50 %/s,取大者 |
| `acc_h`, `acc_z` | 飞机**沿轨迹**加速度 / 垂直加速度(摇杆操作)。不算向心加速度,否则环绕会被误杀 | 1.2 / 1.0 m/s² |
| `yaw_acc`, `pitch_acc` | 云台角加速度(手动转头/俯仰调整)。环绕时放宽到 `acc_h / 半径`,因为匀加速起环必然带来角加速度 | 8 °/s² |
| `consist` | 2s 窗口内画面运动矢量之和的模 ÷ 模之和(方向一致性)。云台慢速来回晃时遥测是平的,只有这个指标能抓到 | ≥ 0.6(画面运动 > 8 px/s 时生效) |
| `blur_drop` | 清晰度相对局部中值的比值(糊帧) | > 0.45 |
| `expo_jump` | 0.5s 内曝光量变化(log2(ISO×快门)),自动曝光的 ISO 微调约 0.15 档不算 | < 0.3 档 |

**感知优先:只有振荡才算抖。** 人眼对"一卡一卡"或"先左后右互相矛盾"的扰动敏感,而对快速但方向连续的运镜(快甩、快抬、急停)完全宽容。
因此所有画面否决(幅度、爆发、画面加速度)都要求 `hf_flip` 判定振荡;遥测否决(飞机/云台加速度)同样只在画面振荡时生效。
跟拍运动中的车再绕到车头时,地面坐标系里飞机从倒飞 8 m/s 变成正飞 7 m/s,加速度 2–3 m/s²,但画面是平滑的——不剔。

**按连续性分段,段内给出运镜序列。** 连续保留帧构成一段(同类型间隔 < 0.5s 合并,首尾各掐 0.3s,短于 3s 丢弃),
不再在运镜类型切换处切断,否则"跟拍 → 绕车 → 跟拍"这种转场会被拆成三段。段的标签是其中持续 ≥ 3s 的子运镜序列,用 `>` 连接,
例如 `pull_back+pan>orbit_CW`、`push_in>orbit_CCW`。逐帧基础类型(static / pan / tilt / orbit / vertical / push / pull / truck / diagonal,
可带 `+climb` / `+descend`)经 2s 众数滤波后才用于序列。细标签:`push_in`、`pull_back`、`dronie`(后退+爬升)、`truck_L/R`、
`orbit_CW/CCW`(侧向移动、云台反向转动,且 v_lat/yaw_rate 推出的半径在 1–1000 m;JSON 里给出 `radius_m`)、`spiral_in/out`、
`ascend/descend`、`pan_L/R`、`tilt_up/down`、`static`,附加 `+pan`。`--split-labels 1` 恢复按类型切断的旧行为。

### 4. 切片

- `copy`:起点向后对齐到最近关键帧(DJI 每 0.5s 一个 I 帧,最多损失 0.5s),`-c copy` 无损、秒级完成。
- `encode`:精确到帧,macOS 用 `hevc_videotoolbox` Main10 80Mbps(4K60 约 0.5x 实时),其它平台自动回退 `libx265`。
- `prores`:ProRes 422 HQ 10-bit(macOS `prores_videotoolbox`,其它平台 `prores_ks`)。
- 任何模式都会写 EDL,推荐在剪辑软件里用 EDL 看原片 + 标签,最省画质。

## 示例结果(DJI_20261004123758_0011_D.MP4,74.6s)

```
kept frames 69.7%  ->  2 segments, 40.3s
  00:12.246 - 00:46.697  34.5s  push_in>push_in+climb  jitter=2.7px/s v=11.5m/s
  01:08.435 - 01:14.308   5.9s  push_in                jitter=7.3px/s v=9.3m/s
```

(改为画面优先后,37–46 s 的加速与爬升因画面平滑而保留;想按遥测严格剔除可用 `--accel-veto-rel 0`。)

被剔除的部分:0–12s 起步加速 + 两次云台偏航修正,18–21s 爬升,37–47s 加速/爬升/减速,48–52s 快速甩镜(60°/s,画面抖动 2px),
53–62s 手动机动(加速度 2–5 m/s²,角加速度 10–40°/s²),63–65s 纯上升只有 2s(可用 `--min-len-s 2` 保留),68–74s 末尾加速。

## 示例 2:环绕(DJI_202604130004_0027_D.MP4,4K30,120s)

```
kept frames 100%  ->  1 segments, 119.7s
  00:00.300 - 02:00.020  119.7s  truck_L>orbit_CCW  jitter=0.5px/s v=5.8m/s yaw=+3.0°/s r≈111m
```

GPS 轨迹拟合圆半径 111.5 m(残差 2.5 m),与 v_lat / yaw_rate 推出的 111 m 一致;全程相机指向与圆心方位偏差 < 3°。
前 4 秒是起步(速度 < 0.4 m/s 或角速度 < 0.8°/s),被归为 static/truck 丢掉;4–20 秒是匀加速起环,与主体合并为一段。

## 示例 3:近距离环绕(DJI_20261001113453_0010_D.MP4,4K30,16s)

```
kept frames 91.2%  ->  1 segments, 11.0s
  00:00.300 - 00:11.345  11.0s  orbit_CW  jitter=3.7px/s v=1.7m/s yaw=-27.3°/s r≈4m
```

绕一个坐在石头上的人,半径 4 m、高度 2.4 m,速度从 0 匀加速到 2.9 m/s 再刹停,云台角速度最高 52°/s。
这段暴露了两个问题并已修正:近距离视差让 RANSAC 全局运动模型在前景/背景之间跳动,被误判成 1–2 px 的抖动(改用中位数光流后只有 4 px/s,
即 0.13 px/帧);向心加速度 v²/r 达 2 m/s² 被当成摇杆操作(改为只看沿轨迹加速度)。
保留的 1.6–11.3 s 覆盖起环到最高速,11–13 s 的刹停(沿轨迹减速 1.2 m/s²、角加速度 20°/s²)被切掉,之后 3 s 的悬停慢摇不足 3 s 丢弃。

## 示例 4:跟车 + 绕车转场(DJI_20260927115144_0081_D 410s,DJI_20260928132001_0018_D 308s,4K60)

```
0081: 02:07.427 - 02:33.587  26.2s  pull_back+pan>orbit_CW      倒飞跟车 → 绕到车侧/车尾
      04:18.024 - 04:31.955  13.9s  spiral_out>pull_back+pan>orbit_CCW
0018: 01:13.740 - 02:53.990 100.3s  push_in>orbit_CCW           正飞跟车 → 绕到车前
      03:02.282 - 03:33.997  31.7s  orbit_CCW>push_in
```

绕行驶中的车在地面坐标系里不是圆,`radius_m` 只是 v_lat/yaw_rate 的名义值,不要当真。
两段里绕车结束后急刹(沿轨迹减速 2–3 m/s²、角减速 10–12°/s²)各有 1–4 s 被切掉,可用 `--yaw-acc 15 --accel-veto-rel 1.0` 放宽,
但那样整个飞行会合并成几分钟的一段,不如用默认值再在剪辑软件里按序列标签挑。

## 交互式轨迹页面

每次运行都会生成 `<name>_viz.html`(也可单独 `uv run python -m dronesift.viz x.MP4`):上方 3D 轨迹(按运镜类型着色、相机朝向短刺、选区外变灰),
中间时间区间控制、片段芯片和覆盖选区的缩略图序列,下方共用时间轴的速度/云台/画面/保留/高度图,框选即改选区,悬停显示对应帧缩略图并在 3D 中定位。
缩略图只解关键帧(`-skip_frame nokey`),每 1–3 s 一张拼成一张 JPEG 精灵图缓存为 `<name>_thumbs.jpg`,页面自包含,Plotly 从 cdnjs 加载。

## 批处理

`./batch.sh DIR...` 两个并发遍历目录里的所有 MP4;`uv run python -m dronesift.summary OUT DIR...` 汇总所有结果为一张
index.html(每条视频的时长/保留率/运镜序列/时间线缩略图)和 segments.csv。

## GUI(实验性)

一个最小的 PySide6 应用,覆盖"拖入文件夹 → 后台分析 → 审查 → 导出"的完整流程:

```bash
uv sync --extra gui
uv run dronesift-gui
```

添加素材文件夹后队列自动跑(两个并发);双击已完成的行,交互报告会在应用内打开(未装 QtWebEngine
时退回浏览器),右侧是镜头清单——取消勾选不要的镜头,点"重新导出"即无损 copy 到 `<out>/export/`
(含 EDL + FCPXML)。

**镜头化**(`--max-shot-s`,默认关):成片剪辑里每个镜头通常 4–10s,平均 6s 左右。开启后长运镜段会
自动切成不超过 max_shot_s 的成片级镜头:切点优先选运镜子类型边界,否则选画面最平静的时刻
(避开抖动爆发±0.5s),不足 min_shot_s 的尾巴并回上一段。

## 状态与边界

- 阈值是在少量 Air 3S 素材(4K30–60)上调出来的,换机型/场景请从默认值开始自行调参,欢迎反馈其它机型的表现。
- 本项目与 DJI 无关,非官方产品;`djmd` 为逆向所得的格式,DJI 固件更新可能改变字段布局(`pbraw.py` 可用来重新推断)。

Apache-2.0 协议,见 [LICENSE](LICENSE)。
