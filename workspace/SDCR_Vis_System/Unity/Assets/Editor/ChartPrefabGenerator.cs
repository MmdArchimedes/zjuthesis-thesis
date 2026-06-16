using UnityEngine;
using UnityEditor;
using System.IO;

/// <summary>
/// Unity Editor工具：一键生成所有30种图表Prefab
/// 使用方法：菜单栏 → Tools → Generate All Chart Prefabs
/// </summary>
public class ChartPrefabGenerator : EditorWindow
{
    private static readonly string PREFAB_PATH = "Assets/Prefabs/Charts";
    private static readonly string MATERIAL_PATH = "Assets/Prefabs/Charts/Materials";

    // 30种图表定义：{文件名, Renderer脚本类名}
    private static readonly (string name, string renderer)[] CHART_TYPES = new[]
    {
        // 基础图表 (8)
        ("BarChart",              "BarChartRenderer"),
        ("LineChart",             "LineChartRenderer"),
        ("PieChart",              "PieChartRenderer"),
        ("ScatterChart",          "ScatterChartRenderer"),
        ("RadarChart",            "RadarChartRenderer"),
        ("AreaChart",             "LineChartRenderer"),      // 面积图复用折线图渲染器
        ("FunnelChart",           "FunnelChartRenderer"),
        ("GaugeChart",            "GaugeChartRenderer"),
        // 进阶图表 (12)
        ("HeatmapChart",          "HeatmapRenderer"),
        ("SankeyChart",           "SankeyRenderer"),
        ("TreemapChart",          "TreemapRenderer"),
        ("BoxplotChart",          "BoxplotRenderer"),
        ("CandlestickChart",      "CandlestickRenderer"),
        ("SunburstChart",         "SunburstRenderer"),
        ("ParallelChart",         "ParallelCoordinatesRenderer"),
        ("ThemeRiverChart",       "ThemeRiverRenderer"),
        ("GraphChart",            "GraphNetworkRenderer"),
        ("TreeChart",             "TreeRenderer"),
        ("WordCloudChart",        "WordCloudRenderer"),
        ("CalendarChart",         "CalendarRenderer"),
        // 地图 (5)
        ("ChinaMapChart",         "ChinaMapRenderer"),
        ("ProvinceMapChart",      "ProvinceMapRenderer"),
        ("ScatterMapChart",       "ScatterMapRenderer"),
        ("FlowMapChart",          "FlowMapRenderer"),
        ("HeatmapMapChart",      "HeatmapMapRenderer"),
        // 复合图表 (5)
        ("BarLineMixChart",       "BarLineMixRenderer"),
        ("ScatterLineMixChart",   "ScatterLineMixRenderer"),
        ("DashboardChart",        "DashboardRenderer"),
        ("TimelineCompositeChart","TimelineCompositeRenderer"),
        ("Bar3DChart",            "Bar3DRenderer"),
    };

    [MenuItem("Tools/Generate All Chart Prefabs")]
    public static void GenerateAll()
    {
        // 1. 确保目录存在
        EnsureDirectoryExists(PREFAB_PATH);
        EnsureDirectoryExists(MATERIAL_PATH);

        // 2. 创建共享材质
        Material chartMat = CreateChartMaterial();
        GameObject labelPrefab = CreateLabelPrefab();

        // 3. 逐个创建图表Prefab
        int created = 0;
        foreach (var (name, renderer) in CHART_TYPES)
        {
            if (CreateChartPrefab(name, renderer, chartMat, labelPrefab))
                created++;
        }

        // 4. 刷新Asset数据库
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        Debug.Log($"[ChartPrefabGenerator] 完成！创建了 {created}/{CHART_TYPES.Length} 个图表Prefab");
    }

    private static bool CreateChartPrefab(string prefabName, string rendererClassName,
                                           Material material, GameObject labelPrefab)
    {
        string prefabPath = $"{PREFAB_PATH}/{prefabName}.prefab";

        // 如果已存在，跳过（避免覆盖手动修改）
        if (AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) != null)
        {
            Debug.Log($"  [{prefabName}] 已存在，跳过");
            return false;
        }

        // 创建GameObject
        GameObject go = new GameObject(prefabName);

        // 添加基础组件
        var meshFilter = go.AddComponent<MeshFilter>();
        var meshRenderer = go.AddComponent<MeshRenderer>();
        meshRenderer.material = material;

        // 添加通用Chart组件
        var materialProvider = go.AddComponent<ChartMaterialProvider>();
        materialProvider.DefaultChartMaterial = material;

        var labelManager = go.AddComponent<ChartLabelManager>();
        if (labelPrefab != null)
            labelManager.LabelPrefab = labelPrefab;

        var axisRenderer = go.AddComponent<ChartAxisRenderer>();

        // 添加该图表类型专用的Renderer脚本
        System.Type rendererType = System.Type.GetType(rendererClassName);
        if (rendererType == null)
        {
            // 尝试从所有Assembly中查找
            foreach (var asm in System.AppDomain.CurrentDomain.GetAssemblies())
            {
                rendererType = asm.GetType(rendererClassName);
                if (rendererType != null) break;
            }
        }

        if (rendererType != null)
        {
            go.AddComponent(rendererType);
        }
        else
        {
            Debug.LogWarning($"  [{prefabName}] 找不到Renderer脚本: {rendererClassName}，请确认脚本已创建");
            // 仍然保存Prefab，但标注缺少脚本
        }

        // 保存为Prefab
        PrefabUtility.SaveAsPrefabAsset(go, prefabPath);
        DestroyImmediate(go);

        Debug.Log($"  [{prefabName}] ✓ 创建成功");
        return true;
    }

    private static Material CreateChartMaterial()
    {
        string matPath = $"{MATERIAL_PATH}/ChartMaterial.mat";

        // 检查是否已存在
        var existing = AssetDatabase.LoadAssetAtPath<Material>(matPath);
        if (existing != null) return existing;

        // 使用Unlit/Color shader（不受灯光影响，颜色由代码控制）
        Material mat = new Material(Shader.Find("Unlit/Color"));
        mat.color = Color.white;
        AssetDatabase.CreateAsset(mat, matPath);

        Debug.Log($"  材质已创建: {matPath}");
        return mat;
    }

    private static GameObject CreateLabelPrefab()
    {
        string labelPath = $"{PREFAB_PATH}/ChartLabel.prefab";

        var existing = AssetDatabase.LoadAssetAtPath<GameObject>(labelPath);
        if (existing != null) return existing;

        // 创建TextMeshPro标签
        GameObject labelGo = new GameObject("ChartLabel");

        // 尝试使用TextMeshPro
        var tmp = labelGo.AddComponent<TMPro.TextMeshPro>();
        tmp.text = "Label";
        tmp.fontSize = 0.04f;
        tmp.color = Color.white;
        tmp.alignment = TMPro.TextAlignmentOptions.Center;
        tmp.fontStyle = TMPro.FontStyles.Normal;

        // 如果TMP不可用，回退到普通TextMesh
        if (tmp.font == null)
        {
            DestroyImmediate(tmp);
            var textMesh = labelGo.AddComponent<TextMesh>();
            textMesh.text = "Label";
            textMesh.fontSize = 36;
            textMesh.color = Color.white;
            textMesh.anchor = TextAnchor.MiddleCenter;
        }

        PrefabUtility.SaveAsPrefabAsset(labelGo, labelPath);
        DestroyImmediate(labelGo);

        Debug.Log($"  标签Prefab已创建: {labelPath}");
        return AssetDatabase.LoadAssetAtPath<GameObject>(labelPath);
    }

    private static void EnsureDirectoryExists(string path)
    {
        if (!AssetDatabase.IsValidFolder(path))
        {
            string parent = Path.GetDirectoryName(path).Replace("\\", "/");
            string folder = Path.GetFileName(path);
            EnsureDirectoryExists(parent);
            AssetDatabase.CreateFolder(parent, folder);
        }
    }
}
