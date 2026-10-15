[English](README.md) | 中文

# dronesift — 大疆航拍素材自动切片(只留稳定 & 连续运镜)

**把长航拍里的好镜头挑出来，交给下一步剪辑。**

[![真实航拍切片：山谷巨岩拉远、谷地推进、徒步人物跟拍](docs/media/steady-moves.gif)](docs/media/steady-moves.mp4)

*12 秒真实 v6 精选，原速播放。[观看 960px MP4](docs/media/steady-moves.mp4)。*

## 看一次真实切片

[![左侧原片与右侧切片对照，红色标记被剔除的拨杆修正](docs/media/cut-comparison.gif)](docs/media/cut-comparison.mp4)

**10.08 秒原片 → 两段合计 7.81 秒的稳定镜头。** 红色区间包含一次拨杆修正及前后缓冲。
右侧把两段可用原片直接剪接，结束后用提示卡标明已播完；左侧继续展示原片。
两边均为原速，景别的跳变就是实际切点。

[观看 MP4 对照](docs/media/cut-comparison.mp4) · [素材范围与重建方法](docs/showcase.md)

## 看懂一条航拍，再看实际切片

[![当前 v6 分析报告：三维轨迹、缩略图、速度、云台角速度与保留区间](docs/media/analysis-report.png)](docs/showcase.md#flight-analysis--航拍分析)

同一条 **74.59 秒原片，最终选出 3 段，共 27.83 秒**。三维轨迹与下方时间线联动，
可以对照地速、爬升、云台偏航/俯仰、画面运动与高度，查看哪里连贯、哪里需要排除。
页头百分比是逐帧判定比例；最终片段还要经过边缘裁切和最短时长筛选。

[观看这条原片的 8 秒切片](docs/media/analysis-selection.mp4) ·
[放大查看时间线](docs/media/analysis-timeline.png) ·
[交互 HTML 示例](docs/examples/flight-report.html)（下载/克隆项目后打开，保留 `docs` 目录结构；图表需联网）

输入一个 DJI 视频(MP4,可带同名 SRT),输出:

- `<name>_segments.json`:每段起止时间/帧、运镜标签、估计加速上限、保留率和检出的扰动区间
- `<name>_exports.json`:实际导出时长、帧数、源片范围和校验结果
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
uv run python -m dronesift x.MP4 --min-len-s 2 --jitter-px 40  # 放宽阈值
uv run python -m dronesift x.MP4 --out x_cuts_v5 --cache-dir x_cuts_v4
# 复用 v4 测量数据,另存 v5 结果
```

重复运行会复用缓存的 telemetry/flow CSV,只重新分段(几秒钟);调阈值很快。
`--cache-dir` 把已有遥测、光流和缩略图缓存复制到新输出目录,便于保留不同版本的结果。

开发时可运行测试,同时生成终端和 HTML 覆盖率报告:

```bash
uv run --locked pytest tests/ -q --cov=dronesift --cov-config=pyproject.toml --cov-report=term --cov-report=html
```

覆盖统计包括 CLI 子进程;HTML 报告位于 `htmlcov/index.html`。
集成测试需要 `ffmpeg` 和 `ffprobe` 均在 `PATH` 中;本地缺少任一工具时,相关测试会跳过,
CI 已确保在测试前安装两者。

## 原理

每个片段新增 `max_playback_speed` 和 `playback_speed_profile`，HTML 片段按钮显示
类似 `≤1.25×` 的估计上限，并注明评估的输出帧率（默认 30 fps）。
检查使用短时画面/云台变化和每输出帧位移；播放倍率为 k 时，速度、加速度、jerk
分别放大 k、k²、k³。规律匀速、均匀加速与短时修正分开处理，避免只按平均速度判断。
导数超限还需同一运动分量的局部可见位移佐证，避免把微小测量噪声当作扰动；加速本身不放大摆动幅度。

这是按若干倍率档位计算的规则估计，不是人眼舒适度保证。`null` 表示证据不足，
`0` 表示没有通过的档位；通过最高测试档 4× 也不代表可以无限加速。
原有的静止/起步、车辆运动状态和倒放检查仍独立生效。

已有切片可以复用缓存补充标记，不重编码视频：

```bash
uv run python -m dronesift.assess_speed /path/to/DJI_xxx_cuts_v6 --output-fps 30
```

该命令更新片段 JSON、已有导出报告，以及缓存缩略图可用时的 HTML。
导出子片段继承的整段估计只供参考；合成时必须对**实际消费的精确原片区间**、
目标播放倍率和输出帧率重新评估。选片与渲染会拒绝超限加速；旧加速计划缺少评估时也会被拦住。

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

### 3. 融合判定 v5(`segment.py`)

遥测按 `pts` 时间戳重采样到视频帧上(丢帧/时轨偏移不会错位;遥测未覆盖的帧一律判为不保留)。
v4 把很多否决条件都放在高频方向反转门控后,会漏掉单方向拨杆修正和急刹;
仅凭画面分量反向又会误切平滑的自动运镜。v5 增加独立的短时扰动检测,
保留有编排的平滑反向。画面指标按 640 px 宽、每秒为单位,窗口和扩边使用原片实际帧率。

| 指标 | 含义 | 默认阈值 |
|---|---|---|
| `yaw_correction_dps` | 0.12s 平滑后求云台角速度,与其 0.8s 平滑轨迹比较;偏航速度残差捕捉单方向修正 | > 6 °/s |
| `yaw_nudge_dps`, `yaw_nudge_rel`, `yaw_nudge_motion_px` | 较小的独立左右修正:偏航速度残差同时超过固定阈值及局部 4s 角速度绝对值峰值的 50%,且水平中位数光流的 0.12s / 0.8s 平滑残差超阈值;只用水平分量佐证,避免快速俯仰的垂直光流触发 | > 2 °/s、0.50、> 10 px/s |
| `yaw_nudge_episode_s`, `yaw_nudge_gap_s`, `yaw_nudge_quiet_dps` | 已确认的左右修正扩展到完整短运镜过程;角速度活动超过 0.5°/s 与局部峰值 10% 的较大者,连接短暂停顿。仅扩展不超过 5s 的过程,较长连续转向维持局部剔除 | 5s、0.25s、0.5 °/s |
| `pitch_return_dps`, `pitch_return_rel` | 4s 窗口内俯仰角速度正反两侧均 > 1.5°/s,残差同时 > 1.5°/s 和局部速度峰值的 35%,且画面运动残差 > 5px/s;捕捉短暂下摇再回调 | 1.5 °/s、0.35 |
| `roll_correction_dps` | 横滚角速度残差须与光流旋转残差同时出现,避免只凭视差或遥测误判 | > 1.5 °/s,旋转画面残差 > 5 px/s |
| `brake_yaw_acc`, `brake_motion_acc` | 相机偏航速度确实减小,且画面速度也确实下降,两个减速度均超阈值才判急刹 | 12 °/s²、40 px/s² |
| `disturbance_min_s`, `disturbance_pad_s` | 新增扰动证据持续达到最短时间,命中后两侧各留稳定余量 | 0.10s、0.35s |
| `hf_flip` | 高频光流在 0.1s 内方向反转的帧占比;仍用于原有持续抖动、爆发和加速度门控,新增修正与急刹检测独立判定 | > 0.25 |
| `motion_rev_amp` | 旧版平移/缩放分量变号否决;默认关闭,可用 `--motion-rev-amp 25` 启用 | 0(关闭) |
| `jitter_px` | 特征点**中位数**光流减去 0.15s 低通后的高频残差 RMS,仅在 `hf_flip` 判定为振荡时生效。用中位数而不是 RANSAC 全局拟合,是因为近距离视差会让拟合在不同深度层之间跳 | 30 px/s |
| `jitter_spike_px` | 0.15s 短窗 RMS 捕捉被 0.5s 中值滤波稀释的爆发;仍要求 `hf_flip` 振荡和连续 ≥3 帧超阈值,两侧各扩 0.35s | 45 px/s |
| `motion_acc` | 1s 低通后画面运动速度的变化率(运镜起/停/变速) | 60 px/s² 或当前画面速度的 50 %/s,取大者 |
| `acc_h`, `acc_z` | 飞机**沿轨迹**加速度 / 垂直加速度(摇杆操作)。不算向心加速度,否则环绕会被误杀 | 1.2 / 1.0 m/s² |
| `yaw_acc`, `pitch_acc` | 云台角加速度(手动转头/俯仰调整)。环绕时放宽到 `acc_h / 半径`,因为匀加速起环必然带来角加速度 | 8 °/s² |
| `consist` | 2s 窗口内画面运动方向一致性;低一致性须伴随高频振荡才否决,平滑 U 形转向可保留 | ≥ 0.6(画面运动 > 8 px/s 且振荡时生效) |
| `blur_drop` | 清晰度相对局部中值的比值(糊帧) | > 0.45 |
| `expo_jump` | 0.5s 内曝光量变化(log2(ISO×快门)),自动曝光的 ISO 微调约 0.15 档不算 | < 0.3 档 |

**持续运镜与短时修正分别取证。** 原有抖动幅度、爆发和飞机/云台加速度检查仍有振荡门控,
新增偏航修正、俯仰回调、横滚和急刹检查可以独立打断片段。快速自动俯仰不会仅因幅度大就被剔除;
达到判定条件的中等画面振荡也会留下硬排除区,即使未达到爆发阈值,也禁止重新合并。
算法不根据文件名或大师模式身份特殊放行。单方向俯仰调整仍可能与有意快速下摇难以区分,
当前俯仰检测主要针对来回修正。

**按连续性分段,段内给出运镜序列。** 连续保留帧构成一段,小于等于 0.5s 的测量间隙可以合并,
确定的扰动与遥测未覆盖区间始终禁止跨越合并。内部切点两侧各掐 0.3s,安全的原片首尾不额外裁切,短于 3s 丢弃。
不再在运镜类型切换处切断,否则"跟拍 → 绕车 → 跟拍"这种转场会被拆成三段。段的标签是其中持续 ≥ 3s 的子运镜序列,用 `>` 连接,
例如 `pull_back+pan>orbit_CW`、`push_in>orbit_CCW`。逐帧基础类型(static / pan / tilt / orbit / vertical / push / pull / truck / diagonal,
可带 `+climb` / `+descend`)经 2s 众数滤波后才用于序列。细标签:`push_in`、`pull_back`、`dronie`(后退+爬升)、`truck_L/R`、
`orbit_CW/CCW`(侧向移动、云台反向转动,且 v_lat/yaw_rate 推出的半径在 1–1000 m;JSON 里给出 `radius_m`)、`spiral_in/out`、
`ascend/descend`、`pan_L/R`、`tilt_up/down`、`static`,附加 `+pan`。`--split-labels 1` 恢复按类型切断的旧行为。

### 4. 切片

- `copy`:从选定区间内的关键帧开始无损导出,只包含呈现时间完整落在区间内的包。
  末尾同时处理 B 帧解码与显示顺序不同的情况,避免把已剔除的扰动带回切片;因此实际切片可能比选择区间略短。
  导出后逐包校验时间戳与帧数,并写入 `_exports.json`。
- `encode`:精确到帧,macOS 用 `hevc_videotoolbox` Main10 80Mbps(4K60 约 0.5x 实时),其它平台自动回退 `libx265`。
- `prores`:ProRes 422 HQ 10-bit(macOS `prores_videotoolbox`,其它平台 `prores_ks`)。
- 任何模式都会写 EDL,推荐在剪辑软件里用 EDL 看原片 + 标签,最省画质。

## v5 对照验证

以 2026-10-01 素材为例,复用 `_cuts_v4` 缓存并导出到独立验证目录:

```bash
uv run python scripts/validate_v5.py /path/to/shoot \
    --out ./artifacts/20261001_v5 --jobs 2
```

输出 `<video>_cuts_v5/` 和 `v5_validation.html`、`.csv`、`.json`。
把 `--out` 设为原始素材目录,即可生成与 v4 并列的 v5 目录。
验证报告比较实际视频包时长而非日志中的保留帧比例,并检查所有导出范围没有覆盖已检出的扰动。
HTML 内可播放两版切片,查看扰动时间及交互轨迹。

`DJI_20261001113453_0010_D` 是拨杆扰动示例:其 v4 高频反转比例最大只有 0.222,
低于门控阈值,急刹和单向修正被漏掉。v5 分段选择 0–7.44s,剔除后续修正与急刹。
`DJI_20261001123644_0014_D`、`DJI_20261001124446_0017_D` 是大师模式保留率对照,
验证脚本要求实际导出时长占原片至少 99%;分段算法本身不读取这两个文件名。
精确的导出时长和保留率以生成的验证报告与 `_exports.json` 为准。

播放反馈指出 `DJI_20261001123356_0013_D` 原片约 11.3s(首个切片中间)、
`DJI_20261001123307_0012_D` 原片约 33.9s(第二个切片末尾)还有明显左右修正,
偏航残差峰值分别为 4.26°/s、2.59°/s,低于原有 6°/s 阈值。
新增 `yaw_nudge` 同时使用角速度相对突变与水平画面证据,排除这两处并完整保留两个大师模式对照。
第三处反馈是 `DJI_20261001112724_0005_D` 从原片 65.5655s 开始的旧切片第 4–5s:
完整左右调整覆盖原片 67.90–71.99s。确认扰动后扩展到完整短过程,再添加缓冲和裁边,
后续稳定区间从 72.62s 开始,避免切片首尾仍落在调整过程中。规则依据角速度与画面证据,
正常连续转向、规律加速和连贯跟随主体仍使用相同判定。
这不代表所有单向俯仰调整都已解决;剩余切片仍需结合播放判断观感。

## 交互式轨迹页面

每次运行都会生成 `<name>_viz.html`(也可单独 `uv run python -m dronesift.viz x.MP4`):上方 3D 轨迹(按运镜类型着色、估计相机视向短刺、选区外变灰),
中间时间区间控制、片段芯片和覆盖选区的缩略图序列,下方共用时间轴的速度/云台/画面/保留/高度图,框选即改选区,悬停显示附近关键帧缩略图并在 3D 中定位。
视向由机身航向和云台俯仰估计,不是完整相机姿态。
缩略图只解关键帧(`-skip_frame nokey`),采样间隔为 `max(1, ceil(原片秒数 / 200))` 秒,
拼成一张 JPEG 精灵图缓存为 `<name>_thumbs.jpg`。测量数据和缩略图内嵌在 HTML 中,
Plotly 图表库与网页字体需联网加载。

## 批处理

`./batch.sh DIR...` 两个并发遍历目录里的所有 MP4;`uv run python -m dronesift.summary OUT DIR...` 汇总所有结果为一张
index.html(每条视频的时长/保留率/运镜序列/时间线缩略图)和 segments.csv。

当前 v6 批处理用 `scripts/batch_v6.py`,先在工作区独立暂存目录生成逐片 checkpoint、每日验证报告和
`index.html`。重复运行同一命令即可断点续跑;仅复用源文件状态和算法哈希一致且导出已验证的结果。
`--adopt-ready` 可在首次运行时导入当前算法生成的完整已有日报表;可选 `--hwaccel none` 使用软件解码。

```bash
uv run python scripts/batch_v6.py /path/to/day1 /path/to/day2 --out validation/batch_v6 --adopt-ready
uv run python scripts/batch_v6.py /path/to/day1 /path/to/day2 --out validation/batch_v6 --publish
```

所有原片成功且全部导出通过校验后,`--publish` 先检查整个批次,再把 `<video>_cuts_v6` 目录及报告
移动到原视频旁;已有 v6 输出时拒绝覆盖。

## v6 配乐成片

`scripts/build_montage.py` 使用实测节拍和带时间的 KML 轨迹规划成片。默认 `regional`
按沿途区域推进,允许附近不同视角的镜头局部重排,包括轨迹旁的山谷、岩壁和远景,
不要求无人机位于 KML 线上。已知地面平移方向与徒步方向夹角不超过 90°;
旋转镜头互不相邻,同一物理地点最多连续两镜。`--route-mode strict` 使用拍摄时间顺序。
逐窗检查开始 0.5s 已有运动及途中停顿;
只有云台与水平画面都明显回摆才否决,保留画面连续同向移动的正常跟拍。
含人镜头结合首、中、尾抽帧审查,满足人运动或背景持续运动。场景标注用于避免重复地貌。

默认自然播放。`--allow-reverse` 仅允许完整时间窗已明确审核的可逆镜头:
纯景观或人物未运动,且没有逆放会显得异常的物理动作。片段倒放时配乐仍正常正向播放。
`--max-consecutive-visual-group 2 --visual-group-limit-slots 6` 可避免开场前六镜连续三条同类景物。

```bash
uv run python scripts/build_montage.py music.mp3 /path/to/shoot \
    --track hike.kml --annotations scene_annotations.json \
    --max-consecutive-visual-group 2 --visual-group-limit-slots 6 \
    --bar-pattern 3,3,2 -o output/montage.mp4
```

输出 1080p30 视频、`_edit.json` 节拍与来源范围、`_route.png` 轨迹顺序和
`_render.json` 实际帧数/音轨校验。路线图中的航拍 GPS 表示无人机位置,
拍摄景物的具体坐标未推算。配乐从 0 开始完整保留,用绝对输出帧切换,
不会因逐镜取整积累错位。`--plan PLAN.json` 可复用人工审查后的镜头计划;
`--plan-only` 仅保存计划和路线图。人物很小或俯视时自动识别能力有限,抽帧复核仍有必要。

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
