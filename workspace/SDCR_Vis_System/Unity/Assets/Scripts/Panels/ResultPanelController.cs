using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Displays regression results dynamically on the AR result panel.
/// Content updates when state changes (year, indicator, region filter).
///
/// Thesis reference: Section 4.4 (结果面板), Section 5.3 (实证结果展示).
/// </summary>
public class ResultPanelController : MonoBehaviour
{
    [Header("Panel UI")]
    [SerializeField] private GameObject _panelRoot;
    [SerializeField] private Text _titleText;
    [SerializeField] private Text _baselineText;
    [SerializeField] private Text _heterogeneityText;
    [SerializeField] private Text _summaryText;
    [SerializeField] private Button _togglePanelButton;
    [SerializeField] private Button _closeButton;

    [Header("Tab Buttons")]
    [SerializeField] private Button _tabBaseline;
    [SerializeField] private Button _tabHeterogeneity;
    [SerializeField] private Button _tabMediation;
    [SerializeField] private Button _tabRobustness;

    private enum PanelTab { Baseline, Heterogeneity, Mediation, Robustness }
    private PanelTab _currentTab = PanelTab.Baseline;

    private StateManager _state;
    private DataManager _data;
    private bool _isVisible = true;

    void Start()
    {
        _state = StateManager.Instance;
        _data = DataManager.Instance;

        if (_togglePanelButton != null)
            _togglePanelButton.onClick.AddListener(ToggleVisibility);
        if (_closeButton != null)
            _closeButton.onClick.AddListener(() => SetVisible(false));

        if (_tabBaseline != null) _tabBaseline.onClick.AddListener(() => SwitchTab(PanelTab.Baseline));
        if (_tabHeterogeneity != null) _tabHeterogeneity.onClick.AddListener(() => SwitchTab(PanelTab.Heterogeneity));
        if (_tabMediation != null) _tabMediation.onClick.AddListener(() => SwitchTab(PanelTab.Mediation));
        if (_tabRobustness != null) _tabRobustness.onClick.AddListener(() => SwitchTab(PanelTab.Robustness));
    }

    /// <summary>Called by SDCRPipeline when state changes.</summary>
    public void UpdateForState(StateVector s_t)
    {
        if (!_isVisible || !_data.IsLoaded) return;

        UpdateTitle(s_t);
        RefreshCurrentTab();
    }

    // ── Tab Switching ──────────────────────────────────────────

    private void SwitchTab(PanelTab tab)
    {
        _currentTab = tab;
        RefreshCurrentTab();
    }

    private void RefreshCurrentTab()
    {
        switch (_currentTab)
        {
            case PanelTab.Baseline:
                ShowBaseline();
                break;
            case PanelTab.Heterogeneity:
                ShowHeterogeneity();
                break;
            case PanelTab.Mediation:
                ShowMediation();
                break;
            case PanelTab.Robustness:
                ShowRobustness();
                break;
        }
    }

    // ── Content Rendering ──────────────────────────────────────

    private void UpdateTitle(StateVector s_t)
    {
        if (_titleText == null) return;

        string provName = "全国";
        if (s_t.ProvinceId > 0 && _data.ProvinceNames.TryGetValue(s_t.ProvinceId, out var name))
            provName = name;

        _titleText.text = $"计量分析面板 — {provName} · {s_t.Year}年 · {s_t.Indicator}指标";
    }

    private void ShowBaseline()
    {
        if (_baselineText == null || _data.BaselineModels.Count == 0) return;

        var sb = new StringBuilder();
        sb.AppendLine("<b>基准回归：DEL → ES (面板固定效应)</b>");
        sb.AppendLine("═══════════════════════════════");

        foreach (var model in _data.BaselineModels)
        {
            sb.AppendLine($"\n<b>{model.name}</b>  (N={model.n}, R²={model.r_squared:F3})");
            sb.AppendLine("─────────────────────────────");
            foreach (var c in model.coefficients)
            {
                string sig = c.significance;
                string sigDisplay = string.IsNullOrEmpty(sig) ? "" : $" [{sig}]";
                sb.AppendLine($"  {c.variable}: {c.estimate:F4} (SE={c.se:F4}){sigDisplay}");
            }
        }

        // Add turning point note
        sb.AppendLine($"\n═══════════════════════════════");
        sb.AppendLine($"<color=#FF6600>倒U型拐点: DEL ≈ {_data.TurningPoint:F3}</color>");
        sb.AppendLine("<color=#888888>多数省份仍位于拐点左侧</color>");

        _baselineText.text = sb.ToString();
    }

    private void ShowHeterogeneity()
    {
        if (_heterogeneityText == null || _data.HeterogeneityRegions.Count == 0) return;

        var sb = new StringBuilder();
        sb.AppendLine("<b>区域异质性：分组回归</b>");
        sb.AppendLine("═══════════════════════════════");
        sb.AppendLine($"{"区域",-6} {"系数",10} {"SE",10} {"显著性",8}");
        sb.AppendLine("─────────────────────────────");

        foreach (var r in _data.HeterogeneityRegions)
        {
            sb.AppendLine($"{r.region,-6} {r.coef_del,10:F3} {r.se,10:F4} {r.significance,8}");
        }

        sb.AppendLine("─────────────────────────────");
        sb.AppendLine("<color=#FF6600>梯度: 中部 > 西部 > 东部 > 东北</color>");
        sb.AppendLine("<color=#888888>提示: 数字化红利与产业结构、要素禀赋相关</color>");

        _heterogeneityText.text = sb.ToString();
    }

    private void ShowMediation()
    {
        if (_baselineText == null || _data.MediationModels.Count == 0) return;

        var sb = new StringBuilder();
        sb.AppendLine("<b>中介效应：技术创新(TEIN)路径</b>");
        sb.AppendLine("═══════════════════════════════");
        sb.AppendLine("<i>逐步回归框架：DEL → TEIN → ES</i>");
        sb.AppendLine("");

        foreach (var model in _data.MediationModels)
        {
            sb.AppendLine($"<b>{model.name}</b>");
            foreach (var c in model.coefficients)
            {
                sb.AppendLine($"  {c.variable}: {c.estimate:F4} (SE={c.se:F4}) {c.significance}");
            }
            sb.AppendLine("");
        }

        sb.AppendLine("─────────────────────────────");
        sb.AppendLine($"<color=#FF6600>间接效应占比: 12.33%</color>");
        sb.AppendLine("<color=#888888>存在可识别的中介传导路径</color>");

        _baselineText.text = sb.ToString();
    }

    private void ShowRobustness()
    {
        if (_baselineText == null) return;

        var sb = new StringBuilder();
        sb.AppendLine("<b>稳健性检验汇总</b>");
        sb.AppendLine("═══════════════════════════════");
        sb.AppendLine($"{"方法",-14} {"DEL系数",10} {"显著性",8}");
        sb.AppendLine("─────────────────────────────");

        var checks = new (string method, float coef, string sig)[]
        {
            ("滞后一期", 0.834f, "***"),
            ("缩尾处理", 0.572f, "***"),
            ("剔除2020", 0.623f, "***"),
            ("IV-2SLS", 1.149f, "**"),
        };

        foreach (var check in checks)
        {
            sb.AppendLine($"{check.method,-14} {check.coef,10:F3} {check.sig,8}");
        }

        sb.AppendLine("─────────────────────────────");
        sb.AppendLine("<color=#00CC00>结论: 核心系数方向与显著性总体稳定</color>");

        _baselineText.text = sb.ToString();
    }

    // ── Visibility ─────────────────────────────────────────────

    public void ToggleVisibility()
    {
        SetVisible(!_isVisible);
    }

    public void SetVisible(bool visible)
    {
        _isVisible = visible;
        if (_panelRoot != null)
            _panelRoot.SetActive(visible);
        if (visible)
            UpdateForState(_state.CaptureState());
    }

    public bool IsVisible => _isVisible;
}
