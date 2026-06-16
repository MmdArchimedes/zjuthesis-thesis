using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Interactive mechanism pathway graph.
/// Shows the DEL → ES transmission channels with nodes and edges.
///
/// Thesis reference: Chapter 2 mechanism paths, Section 4.3 (机制图).
/// </summary>
public class MechanismGraphController : MonoBehaviour
{
    [Header("Graph Layout")]
    [SerializeField] private RectTransform _graphContainer;
    [SerializeField] private GameObject _nodePrefab;
    [SerializeField] private GameObject _edgePrefab;
    [SerializeField] private float _graphWidth = 800f;
    [SerializeField] private float _graphHeight = 500f;

    [Header("Detail Panel")]
    [SerializeField] private GameObject _detailPanel;
    [SerializeField] private Text _detailTitle;
    [SerializeField] private Text _detailContent;

    private Dictionary<string, GameObject> _nodeObjects = new Dictionary<string, GameObject>();
    private Dictionary<string, RectTransform> _nodeRects = new Dictionary<string, RectTransform>();

    private static readonly Dictionary<string, string> NodeDescriptions = new Dictionary<string, string>
    {
        {"del", "数字经济发展水平综合指数(DEL)\n\n通过熵权—决策树融合方法构建，涵盖数字基础设施、产业数字化和数字产业化三个维度7项指标。\n\n样本: 2014-2022, 30省份"},
        {"direct", "直接效应\n\n• 资源配置优化: 数字平台打破信息不对称，实现能源精准匹配\n• 消费方式变革: 线上办公、智能家居降低能源消耗\n\n国家数据局(2025)研究表明，数字技术可使能源配置效率提升15%-20%"},
        {"indirect", "间接效应\n\n• 技术创新路径 (12.33%中介)\n• 产业结构升级\n• 绿色治理提升\n\n三大路径协同推动能源结构系统性优化"},
        {"resource", "资源配置优化\n\n• 智能电网实现清洁能源跨区域消纳\n• 大数据分析识别能源消费特征\n• 能源需求精细化管理，降低浪费"},
        {"consumption", "消费方式变革\n\n• 线上业态减少交通能源消耗\n• 智能家居实现生活能源智能化管控\n• 数字服务业降低产业能耗强度"},
        {"tech_innovation", "技术创新 (中介12.33%)\n\n• 大数据+AI模拟清洁能源开发场景\n• 数字仿真降低研发试验成本\n• 数字技术加速绿色技术扩散\n\nDEL每提升1%，技术创新传导至ES约0.054单位"},
        {"industry_upgrade", "产业结构升级\n\n• 工业互联网改造传统高耗能产业\n• 智能制造优化生产流程\n• 带动高端制造业等低耗能产业发展\n\nDEL↑1% → 产业高级化↑0.32% → ES↑0.18%"},
        {"green_governance", "绿色治理提升\n\n• 物联网实时监测重点用能单位\n• 大数据识别能源结构优化关键节点\n• 数字政务实现能源政策精准执行"},
        {"es", "能源结构优化综合指数(ES)\n\n通过熵值法构建，涵盖：\n• 能源消费: 煤炭占比、能源效率\n• 环境和谐: SO₂排放、NOₓ排放\n\n倒U型拐点: DEL≈0.507"},
    };

    private DataManager _data;

    void Start()
    {
        _data = DataManager.Instance;
        if (_data.IsLoaded)
            BuildGraph();
        else
            _data.OnDataLoaded.AddListener(BuildGraph);
    }

    void OnDestroy()
    {
        if (_data != null)
            _data.OnDataLoaded.RemoveListener(BuildGraph);
    }

    // ── Graph Construction ─────────────────────────────────────

    private void BuildGraph()
    {
        if (_data.MechNodes == null || _data.MechNodes.Count == 0) return;

        // Create nodes
        foreach (var nodeData in _data.MechNodes)
        {
            CreateNode(nodeData);
        }

        // Create edges
        foreach (var edgeData in _data.MechEdges)
        {
            if (_nodeRects.TryGetValue(edgeData.from, out var fromRect) &&
                _nodeRects.TryGetValue(edgeData.to, out var toRect))
            {
                CreateEdge(fromRect, toRect, edgeData.label);
            }
        }

        Debug.Log($"[MechanismGraph] Built graph with {_nodeObjects.Count} nodes, {_data.MechEdges.Count} edges");
    }

    private void CreateNode(DataManager.MechanismNode data)
    {
        if (_nodePrefab == null || _graphContainer == null) return;

        var go = Instantiate(_nodePrefab, _graphContainer);
        var rect = go.GetComponent<RectTransform>();

        // Position: normalized (0-1) → pixel coords
        float x = (data.x - 0.5f) * _graphWidth;
        float y = (data.y - 0.5f) * _graphHeight;
        rect.anchoredPosition = new Vector2(x, y);

        // Color
        if (ColorUtility.TryParseHtmlString(data.color, out Color nodeColor))
        {
            var image = go.GetComponent<Image>();
            if (image != null) image.color = nodeColor;
        }

        // Label
        var label = go.GetComponentInChildren<Text>();
        if (label != null) label.text = data.label.Replace("\\n", "\n");

        // Click handler
        var button = go.GetComponent<Button>();
        if (button != null)
        {
            string nodeId = data.id; // capture for closure
            button.onClick.AddListener(() => OnNodeClicked(nodeId));
        }

        _nodeObjects[data.id] = go;
        _nodeRects[data.id] = rect;
    }

    private void CreateEdge(RectTransform from, RectTransform to, string label)
    {
        if (_edgePrefab == null || _graphContainer == null) return;

        var go = Instantiate(_edgePrefab, _graphContainer);
        var rect = go.GetComponent<RectTransform>();

        Vector2 start = from.anchoredPosition;
        Vector2 end = to.anchoredPosition;
        Vector2 dir = end - start;
        float length = dir.magnitude;
        float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

        rect.anchoredPosition = start;
        rect.sizeDelta = new Vector2(length, 3f);
        rect.pivot = new Vector2(0, 0.5f);
        rect.localRotation = Quaternion.Euler(0, 0, angle);

        // Edge label
        if (!string.IsNullOrEmpty(label))
        {
            var labelGO = new GameObject("EdgeLabel");
            labelGO.transform.SetParent(go.transform);
            var labelRT = labelGO.AddComponent<RectTransform>();
            labelRT.anchoredPosition = new Vector2(length * 0.5f, 15f);
            var labelText = labelGO.AddComponent<Text>();
            labelText.text = label;
            labelText.fontSize = 11;
            labelText.alignment = TextAnchor.MiddleCenter;
            labelText.color = new Color(0.3f, 0.3f, 0.3f);
            // Font assignment would need project setup; fallback handled by Unity default
        }
    }

    // ── Interaction ────────────────────────────────────────────

    private void OnNodeClicked(string nodeId)
    {
        if (NodeDescriptions.TryGetValue(nodeId, out var desc))
        {
            ShowDetail(nodeId, desc);
        }
    }

    private void ShowDetail(string nodeId, string content)
    {
        if (_detailPanel != null)
        {
            _detailPanel.SetActive(true);
            if (_detailTitle != null) _detailTitle.text = nodeId;
            if (_detailContent != null) _detailContent.text = content;
        }
    }

    /// <summary>Highlight relevant nodes based on state context.</summary>
    public void UpdateHighlight(StateVector s_t)
    {
        // Highlight changes based on indicator selection
        bool showDEL = s_t.Indicator == "DEL";
        if (_nodeObjects.TryGetValue("del", out var delNode))
        {
            var img = delNode.GetComponent<Image>();
            if (img != null)
                img.color = showDEL ? new Color(0.29f, 0.56f, 0.85f, 1f) : new Color(0.29f, 0.56f, 0.85f, 0.7f);
        }
    }

    public void Hide()
    {
        if (_graphContainer != null)
            _graphContainer.gameObject.SetActive(false);
        if (_detailPanel != null)
            _detailPanel.SetActive(false);
    }

    public void Show()
    {
        if (_graphContainer != null)
            _graphContainer.gameObject.SetActive(true);
    }
}
