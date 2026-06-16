using System.Collections;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Main UI controller for the desktop demo mode.
/// Provides on-screen buttons for all actions (keyboard shortcuts also supported).
/// </summary>
public class DemoUIController : MonoBehaviour
{
    [Header("Main UI Panels")]
    [SerializeField] private GameObject _mainUIRoot;
    [SerializeField] private Button _indicatorToggleBtn;
    [SerializeField] private Text _indicatorToggleLabel;
    [SerializeField] private Button _regionAllBtn;
    [SerializeField] private Button _regionEastBtn;
    [SerializeField] private Button _regionCentralBtn;
    [SerializeField] private Button _regionWestBtn;
    [SerializeField] private Button _regionNortheastBtn;
    [SerializeField] private Button _viewNationalBtn;
    [SerializeField] private Button _autoDemoBtn;
    [SerializeField] private Text _statusText;

    [Header("Keyboard Shortcuts")]
    [SerializeField] private bool _enableShortcuts = true;

    private StateManager _state;
    private ChinaMapController _mapController;
    private ProvinceSelector _provinceSelector;
    private ResultPanelController _resultPanel;
    private MechanismGraphController _mechanismGraph;
    private TimelineController _timeline;
    private Camera _mainCamera;

    private bool _autoDemoRunning = false;

    void Start()
    {
        _state = StateManager.Instance;
        _mainCamera = Camera.main;

        // Find dependencies
        _mapController = FindObjectOfType<ChinaMapController>();
        _provinceSelector = FindObjectOfType<ProvinceSelector>();
        _resultPanel = FindObjectOfType<ResultPanelController>();
        _mechanismGraph = FindObjectOfType<MechanismGraphController>();
        _timeline = FindObjectOfType<TimelineController>();

        // Wire up buttons
        if (_indicatorToggleBtn != null)
        {
            _indicatorToggleBtn.onClick.AddListener(ToggleIndicator);
            UpdateIndicatorLabel();
        }

        if (_regionAllBtn != null) _regionAllBtn.onClick.AddListener(() => _state.SetRegionFilter("全部"));
        if (_regionEastBtn != null) _regionEastBtn.onClick.AddListener(() => _state.SetRegionFilter("东部"));
        if (_regionCentralBtn != null) _regionCentralBtn.onClick.AddListener(() => _state.SetRegionFilter("中部"));
        if (_regionWestBtn != null) _regionWestBtn.onClick.AddListener(() => _state.SetRegionFilter("西部"));
        if (_regionNortheastBtn != null) _regionNortheastBtn.onClick.AddListener(() => _state.SetRegionFilter("东北"));
        if (_viewNationalBtn != null) _viewNationalBtn.onClick.AddListener(ResetToNationalView);
        if (_autoDemoBtn != null) _autoDemoBtn.onClick.AddListener(ToggleAutoDemo);

        // Status update
        _state.OnStateChanged.AddListener((nv, ov) => UpdateStatusText());
        UpdateStatusText();
    }

    void OnDestroy()
    {
        if (_state != null)
            _state.OnStateChanged.RemoveAllListeners();
    }

    void Update()
    {
        if (!_enableShortcuts || _autoDemoRunning) return;

        // ── Keyboard Shortcuts (matching thesis gesture semantics) ──
        if (Input.GetKeyDown(KeyCode.RightArrow)) _state.IncrementYear();
        if (Input.GetKeyDown(KeyCode.LeftArrow)) _state.DecrementYear();
        if (Input.GetKeyDown(KeyCode.Tab)) ToggleIndicator();
        if (Input.GetKeyDown(KeyCode.Escape)) ResetToNationalView();
        if (Input.GetKeyDown(KeyCode.Space)) _timeline?.TogglePlay();
        if (Input.GetKeyDown(KeyCode.R)) _state.SetRegionFilter("全部");
        if (Input.GetKeyDown(KeyCode.Alpha1)) _state.SetRegionFilter("东部");
        if (Input.GetKeyDown(KeyCode.Alpha2)) _state.SetRegionFilter("中部");
        if (Input.GetKeyDown(KeyCode.Alpha3)) _state.SetRegionFilter("西部");
        if (Input.GetKeyDown(KeyCode.Alpha4)) _state.SetRegionFilter("东北");
        if (Input.GetKeyDown(KeyCode.P)) _resultPanel?.ToggleVisibility();
        if (Input.GetKeyDown(KeyCode.M)) _mechanismGraph?.Show();
    }

    // ── Actions ─────────────────────────────────────────────────

    private void ToggleIndicator()
    {
        _state.ToggleIndicator();
        UpdateIndicatorLabel();
    }

    private void UpdateIndicatorLabel()
    {
        if (_indicatorToggleLabel != null)
            _indicatorToggleLabel.text = _state.CurrentIndicator == "ES" ? "📊 ES (能源结构)" : "📡 DEL (数字经济)";
    }

    private void ResetToNationalView()
    {
        _provinceSelector?.DeselectAll();
        _state.SetRegionFilter("全部");
    }

    private void UpdateStatusText()
    {
        if (_statusText == null) return;

        string provName = "全国";
        if (_state.SelectedProvinceId > 0)
        {
            var data = DataManager.Instance;
            if (data != null && data.ProvinceNames.TryGetValue(_state.SelectedProvinceId, out var name))
                provName = name;
        }

        _statusText.text =
            $"{_state.CurrentYear}年 | {_state.CurrentIndicator}指标 | " +
            $"{_state.RegionFilter} | 选中: {provName}";
    }

    // ── Auto Demo (runs through key thesis narrative) ──────────

    private void ToggleAutoDemo()
    {
        if (_autoDemoRunning)
        {
            _autoDemoRunning = false;
            StopAllCoroutines();
            if (_autoDemoBtn != null)
            {
                var label = _autoDemoBtn.GetComponentInChildren<Text>();
                if (label != null) label.text = "▶ 自动演示";
            }
        }
        else
        {
            _autoDemoRunning = true;
            if (_autoDemoBtn != null)
            {
                var label = _autoDemoBtn.GetComponentInChildren<Text>();
                if (label != null) label.text = "⏹ 停止";
            }
            StartCoroutine(AutoDemoCoroutine());
        }
    }

    private IEnumerator AutoDemoCoroutine()
    {
        // Follow the thesis Section 5.4 narrative:
        // "全国格局把握 → 计量证据核对 → 分区差异讨论 → 省域下钻阅读"

        Debug.Log("[AutoDemo] Starting thesis demonstration...");
        yield return new WaitForSeconds(1f);

        // Step 1: National overview - ES coloring 2022
        Debug.Log("[AutoDemo] Step 1: 全国ES格局");
        _state.SetIndicator("ES");
        _state.SetYear(2022);
        _state.SetRegionFilter("全部");
        _resultPanel?.SetVisible(true);
        UpdateIndicatorLabel();
        yield return new WaitForSeconds(3f);

        // Step 2: Time evolution 2014→2022 (auto-play)
        Debug.Log("[AutoDemo] Step 2: 时间演化");
        _state.SetYear(2014);
        yield return new WaitForSeconds(0.5f);
        if (_timeline != null)
        {
            _timeline.TogglePlay(); // start play
            yield return new WaitForSeconds(10f); // play through years
            _timeline.TogglePlay(); // stop
        }
        _state.SetYear(2022);
        yield return new WaitForSeconds(2f);

        // Step 3: Toggle to DEL indicator
        Debug.Log("[AutoDemo] Step 3: 切换到DEL指标");
        ToggleIndicator();
        yield return new WaitForSeconds(3f);

        // Step 4: Regional filtering
        Debug.Log("[AutoDemo] Step 4: 东中西部分区对比");
        _state.SetIndicator("ES");
        UpdateIndicatorLabel();
        foreach (var region in new[] { "东部", "中部", "西部" })
        {
            _state.SetRegionFilter(region);
            yield return new WaitForSeconds(3f);
        }
        _state.SetRegionFilter("全部");
        yield return new WaitForSeconds(1f);

        // Step 5: Province drill-down
        Debug.Log("[AutoDemo] Step 5: 省域下钻");
        int[] demoProvinces = { 19, 11, 16, 23 }; // 广东, 浙江, 河南, 四川
        foreach (var pid in demoProvinces)
        {
            _provinceSelector?.OnProvinceClicked(pid);
            yield return new WaitForSeconds(2.5f);
        }
        _provinceSelector?.DeselectAll();
        yield return new WaitForSeconds(1f);

        // Step 6: Show mechanism graph
        Debug.Log("[AutoDemo] Step 6: 机制路径展示");
        _mechanismGraph?.Show();
        yield return new WaitForSeconds(4f);

        // Step 7: Result panel tabs
        Debug.Log("[AutoDemo] Step 7: 结果面板全览");
        _resultPanel?.SetVisible(true);
        yield return new WaitForSeconds(2f);

        // Done
        Debug.Log("[AutoDemo] Demonstration complete!");
        ResetToNationalView();
        _autoDemoRunning = false;

        if (_autoDemoBtn != null)
        {
            var label = _autoDemoBtn.GetComponentInChildren<Text>();
            if (label != null) label.text = "▶ 自动演示";
        }
    }
}
