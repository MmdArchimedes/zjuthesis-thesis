# SDCR--Vis: 状态驱动的AR沉浸式可视化系统

基于Unity的状态驱动条件刷新多视图可视化管线(SDCR--Vis)，用于毕业论文《省域数字经济与能源结构的AR沉浸式分析系统研究》的演示展示。

## 系统架构

```
交互层(L1) → 数据层(L2) → 查询与映射层(L3) → 呈现层(L4)
  手势/射线       CSV/JSON      归一化→映射      地图/时间轴/面板/机制图
  语音/LLM      内存索引        Eq 4-4,4-5          Eq 4-6 条件刷新
```

**SDCR Pipeline (thesis Eq 4-4, 4-5, 4-6):**
- 归一化: n = (v - v_min) / (v_max - v_min)
- 通道映射: c_i = Lerp(c_min, c_max, n_i), h_i = h_min + λ_h × n_i
- 状态提交: Δs_t ≠ 0 → RenderUpdate(s_t)

## 快速演示（Python版，立即可运行）

```bash
cd SDCR_Vis_System/Python_Demo

# 安装依赖
pip install plotly pandas kaleido

# 运行交互式可视化（自动打开浏览器）
python demo_visualization.py

# 导出论文用静态图
python demo_visualization.py --export_figures
```

打开后可通过修改代码中的状态参数切换年份/指标/区域。

## Unity项目搭建（C#完整版）

### 1. 新建Unity项目
- Unity 2021.3 LTS 或更高版本
- 模板: 3D Core
- 项目名: SDCR_Vis_System

### 2. 导入脚本和资源
将本目录下的 `Unity/Assets/` 内容复制到Unity项目的 `Assets/` 目录：
```
Assets/
├── Scripts/
│   ├── Core/          ← StateManager, DataManager, SDCRPipeline
│   ├── Map/           ← ChinaMapController, ProvinceVisual, ClickableProvince
│   ├── Timeline/      ← TimelineController
│   ├── Panels/        ← ResultPanelController, MechanismGraphController, InfoPanelController
│   ├── Interaction/   ← ProvinceSelector
│   └── UI/            ← DemoUIController
├── Resources/Data/    ← panel_data.csv, regression_results.json, mechanism_paths.json
├── Shaders/           ← ProvinceShader.shader
├── Materials/         ← (见下方创建步骤)
└── Scenes/            ← (见下方创建步骤)
```

### 3. 场景搭建步骤

#### 3.1 创建核心管理器GameObject
1. 在Hierarchy中创建空GameObject，命名为 `SDCR_System`
2. 依次添加Component（按依赖顺序）:
   - `StateManager` (→ Core/StateManager.cs)
   - `DataManager` (→ Core/DataManager.cs)
   - `SDCRPipeline` (→ Core/SDCRPipeline.cs)

#### 3.2 创建地图
1. 在 `SDCR_System` 下创建空GameObject，命名为 `ChinaMap`
2. 添加Component:
   - `ChinaMapController` (→ Map/ChinaMapController.cs)
   - `ProvinceSelector` (→ Interaction/ProvinceSelector.cs)
3. 将 `ChinaMap` 拖入 `SDCRPipeline` 的 `_mapController` 字段

#### 3.3 创建UI Canvas
1. Hierarchy → UI → Canvas (世界空间模式)
   - Render Mode: World Space
   - Position: (0, 0.5, 1.5), Size: (800, 600)
2. 在Canvas下创建UI面板:
   - **TimelinePanel** (底部):
     - Slider (命名: YearSlider)
     - Text (命名: YearLabel)
     - Button × 3 (Play, Prev, Next)
   - **ResultPanel** (右侧):
     - Panel背景
     - Text × 4 (Title, Baseline, Heterogeneity, Summary)
     - Button × 6 (Tab × 4, Toggle, Close)
   - **MechanismPanel** (左下):
     - Panel背景
     - Node Prefab (Button + Image + Text)
     - Edge Prefab (Image)
   - **InfoPanel** (跟随选中省份):
     - Panel背景
     - Text × 4 (ProvinceName, RegionTag, ES/DEL value, Additional)
3. 创建对应Controller并关联UI引用:
   - 在TimelinePanel上添加 `TimelineController`
   - 在ResultPanel上添加 `ResultPanelController`
   - 在MechanismPanel上添加 `MechanismGraphController`
   - 在InfoPanel上添加 `InfoPanelController`
4. 在Canvas上添加 `DemoUIController` 并关联所有UI引用

#### 3.4 创建材质
1. Assets → Create → Material (命名: ProvinceMat)
2. Shader: 选择 `SDCR/ProvinceShader`
3. 拖入 `ChinaMapController._provinceMaterialTemplate`

#### 3.5 关联所有引用

**ChinaMapController:**
- `_provincePrefab`: 创建基础Cube Prefab（带Collider）
- `_mapRoot`: ChinaMap的Transform
- `_provinceMaterialTemplate`: ProvinceMat材质

**SDCRPipeline:**
- `_mapController`: ChinaMapController组件
- `_timelineController`: TimelineController组件
- `_resultPanel`: ResultPanelController组件
- `_mechanismGraph`: MechanismGraphController组件

**ProvinceSelector:**
- `_infoPanel`: InfoPanelController组件

### 4. 运行
1. 场景中添加Directional Light
2. 主相机位置: (0, 1, 0), 看向地图
3. Play → 系统自动加载数据并生成30省地图
4. 交互操作:
   - 点击省份 → 显示信息卡
   - 左右箭头键 → 切换年份
   - Tab → 切换ES/DEL指标
   - 1/2/3/4 → 区域筛选
   - Esc → 全国视图
   - 空格 → 自动播放
   - P → 切换结果面板
   - M → 显示机制图

### 5. 自动演示
点击UI上的"自动演示"按钮，系统按论文叙事自动运行:
`全国格局 → 时间演化 → 指标切换 → 分区对比 → 省域下钻 → 机制展示 → 结果面板`

## 数据文件

| 文件 | 内容 | 对应论文章节 |
|------|------|-------------|
| `panel_data.csv` | 30省×9年面板数据 (ES, DEL, 控制变量) | §5.1 |
| `regression_results.json` | 基准回归/中介/异质性/稳健性结果 | §5.2 |
| `mechanism_paths.json` | 机制路径图节点和边 | §2.3 |

数据与论文中表5-3（基准回归）、表5-5（区域异质性）等一致。

## 文件清单

| 路径 | 用途 |
|------|------|
| `Scripts/Core/StateManager.cs` | 统一状态向量s_t管理与条件刷新事件 |
| `Scripts/Core/DataManager.cs` | 数据加载、内存索引与O(1)查询 |
| `Scripts/Core/SDCRPipeline.cs` | 归一化→映射→提交三阶段管线编排 |
| `Scripts/Map/ChinaMapController.cs` | 30省3D地图生成与颜色/高度更新 |
| `Scripts/Map/ProvinceVisual.cs` | 单省颜色/高度/高亮/降暗控制 |
| `Scripts/Map/ClickableProvince.cs` | 省份点击事件封装 |
| `Scripts/Timeline/TimelineController.cs` | 年份滑块、自动播放与步进控制 |
| `Scripts/Panels/ResultPanelController.cs` | 四标签结果面板（基准/异质性/中介/稳健性） |
| `Scripts/Panels/MechanismGraphController.cs` | 交互式机制路径图（节点+连边+详情） |
| `Scripts/Panels/InfoPanelController.cs` | 省份信息卡片（ES/DEL/控制变量/拐点提示） |
| `Scripts/Interaction/ProvinceSelector.cs` | 省份选择/取消逻辑 |
| `Scripts/UI/DemoUIController.cs` | 桌面演示UI、键盘快捷键、自动演示编排 |
| `Shaders/ProvinceShader.shader` | 带Emission支持的省域着色器 |
| `Python_Demo/demo_visualization.py` | Python交互式可视化（Plotly，立即可运行） |
