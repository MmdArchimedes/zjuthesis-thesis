using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// SDCR--Vis Pipeline orchestrator.
/// Implements the Normalization → Channel Mapping → State Submission
/// three-step visualization encoding (thesis Eqs 4-4, 4-5, 4-6).
///
/// Architecture (thesis Section 4.2):
///   Layer 1 (Interaction) → Layer 2 (Data) → Layer 3 (Query+Map) → Layer 4 (Views)
/// </summary>
[DefaultExecutionOrder(-80)]
public class SDCRPipeline : MonoBehaviour
{
    [Header("Visual Encoding")]
    [SerializeField] private Color _esMinColor = new Color(0.85f, 0.15f, 0.15f); // warm red
    [SerializeField] private Color _esMaxColor = new Color(0.15f, 0.75f, 0.25f); // cool green
    [SerializeField] private Color _delMinColor = new Color(0.65f, 0.75f, 0.95f); // light blue
    [SerializeField] private Color _delMaxColor = new Color(0.35f, 0.25f, 0.75f); // deep purple
    [SerializeField] private float _heightAmplifier = 0.3f;
    [SerializeField] private float _baseHeight = 0f;

    [Header("Components")]
    [SerializeField] private ChinaMapController _mapController;
    [SerializeField] private TimelineController _timelineController;
    [SerializeField] private ResultPanelController _resultPanel;
    [SerializeField] private MechanismGraphController _mechanismGraph;

    private StateManager _state;
    private DataManager _data;

    void Start()
    {
        _state = StateManager.Instance;
        _data = DataManager.Instance;

        // Subscribe to state changes
        _state.OnStateChanged.AddListener(OnStateChanged);
        _data.OnDataLoaded.AddListener(OnDataLoaded);
    }

    void OnDestroy()
    {
        if (_state != null)
            _state.OnStateChanged.RemoveListener(OnStateChanged);
        if (_data != null)
            _data.OnDataLoaded.RemoveListener(OnDataLoaded);
    }

    // ── State Change Handler (thesis Eq 4-6: Δs_t ≠ 0 → RenderUpdate) ──

    private void OnDataLoaded()
    {
        // Initial full refresh
        RefreshAllViews(_state.CaptureState());
    }

    private void OnStateChanged(StateVector newState, StateVector oldState)
    {
        // Conditional refresh: update only what changed
        RefreshAllViews(newState);
    }

    // ── Full SDCR Pipeline Execution ────────────────────────────

    public void RefreshAllViews(StateVector s_t)
    {
        if (!_data.IsLoaded) return;

        // ── Step 1: Query data (thesis Layer 3: Query & Map) ──
        var records = _data.GetYearRecords(s_t.Year, s_t.RegionFilter);
        if (records.Count == 0)
        {
            Debug.LogWarning($"[SDCR] No records for year={s_t.Year}, region={s_t.RegionFilter}");
            return;
        }

        // ── Step 2: Normalization (thesis Eq 4-4) ──
        var (vMin, vMax) = _data.GetValueRange(s_t.Indicator, s_t.RegionFilter);
        float vRange = Mathf.Max(vMax - vMin, 0.0001f);

        // ── Step 3: Channel Mapping (thesis Eq 4-5) ──
        //   c_i = Lerp(c_min, c_max, n_i), h_i = h_min + λ_h * n_i
        var provinceColors = new Dictionary<int, Color>();
        var provinceHeights = new Dictionary<int, float>();

        foreach (var record in records)
        {
            float n = (GetIndicatorValue(record) - vMin) / vRange;
            n = Mathf.Clamp01(n);

            provinceColors[record.ProvinceId] = ColorForIndicator(n);
            provinceHeights[record.ProvinceId] = _baseHeight + _heightAmplifier * n;
        }

        // ── Step 4: Dispatch to views (thesis Layer 4: Presentation) ──
        _mapController?.ApplyColorHeightUpdate(provinceColors, provinceHeights, s_t);
        _timelineController?.SyncWithState(s_t);
        _resultPanel?.UpdateForState(s_t);
        _mechanismGraph?.UpdateHighlight(s_t);
    }

    // ── Visual Encoding Helpers ─────────────────────────────────

    private float GetIndicatorValue(DataManager.IndicatorRecord r)
    {
        return _state.CurrentIndicator == "ES" ? r.ES : r.DEL;
    }

    private Color ColorForIndicator(float normalizedValue)
    {
        Color minC = _state.CurrentIndicator == "ES" ? _esMinColor : _delMinColor;
        Color maxC = _state.CurrentIndicator == "ES" ? _esMaxColor : _delMaxColor;
        return Color.Lerp(minC, maxC, normalizedValue);
    }

    // ── Public API for external triggers ────────────────────────

    public void ForceRefresh()
    {
        RefreshAllViews(_state.CaptureState());
    }
}
