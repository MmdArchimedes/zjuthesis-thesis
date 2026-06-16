using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Central state manager for SDCR--Vis pipeline.
/// Maintains unified state vector s_t = {year, indicator, province, region}
/// and dispatches state change events to all subscribers.
///
/// Thesis reference: Section 4.2, Equation (4-6):
///   Δs_t ≠ 0 → RenderUpdate(s_t)
/// </summary>
[DefaultExecutionOrder(-100)] // Run before all other scripts
public class StateManager : MonoBehaviour
{
    // ── State Vector ────────────────────────────────────────────
    public int CurrentYear { get; private set; } = 2022;
    public string CurrentIndicator { get; private set; } = "ES";
    public int SelectedProvinceId { get; private set; } = 0; // 0 = none/national
    public string RegionFilter { get; private set; } = "全部";

    // Valid ranges (matching thesis data: 2014-2022, 30 provinces)
    public int YearMin = 2014;
    public int YearMax = 2022;
    public string[] Indicators = { "ES", "DEL" };
    public string[] Regions = { "全部", "东部", "中部", "西部", "东北" };

    private StateVector _previousState;

    // ── Events ──────────────────────────────────────────────────
    [System.Serializable]
    public class StateChangedEvent : UnityEvent<StateVector, StateVector> { }

    /// <summary>Fires when any state component changes. Args: (newState, oldState)</summary>
    public StateChangedEvent OnStateChanged = new StateChangedEvent();

    /// <summary>Fires only when year changes.</summary>
    public UnityEvent<int> OnYearChanged = new UnityEvent<int>();

    /// <summary>Fires only when indicator toggles.</summary>
    public UnityEvent<string> OnIndicatorChanged = new UnityEvent<string>();

    /// <summary>Fires when province selection changes.</summary>
    public UnityEvent<int> OnProvinceSelected = new UnityEvent<int>();

    /// <summary>Fires when region filter changes.</summary>
    public UnityEvent<string> OnRegionFilterChanged = new UnityEvent<string>();

    // ── Singleton ───────────────────────────────────────────────
    public static StateManager Instance { get; private set; }

    void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        _previousState = CaptureState();
    }

    // ── Public API ──────────────────────────────────────────────

    /// <summary>
    /// Set year. Only dispatches if value actually changes
    /// (implements conditional refresh, thesis Eq 4-6 reasoning).
    /// </summary>
    public void SetYear(int year)
    {
        year = Mathf.Clamp(year, YearMin, YearMax);
        if (year == CurrentYear) return;

        StateVector oldState = CaptureState();
        CurrentYear = year;

        OnYearChanged.Invoke(CurrentYear);
        DispatchIfChanged(oldState);
    }

    public void IncrementYear() => SetYear(CurrentYear + 1);
    public void DecrementYear() => SetYear(CurrentYear - 1);

    /// <summary>Toggle between ES and DEL indicators.</summary>
    public void SetIndicator(string indicator)
    {
        if (indicator == CurrentIndicator) return;

        StateVector oldState = CaptureState();
        CurrentIndicator = indicator;

        OnIndicatorChanged.Invoke(CurrentIndicator);
        DispatchIfChanged(oldState);
    }

    public void ToggleIndicator()
    {
        string next = (CurrentIndicator == "ES") ? "DEL" : "ES";
        SetIndicator(next);
    }

    /// <summary>Select a province by ID (0 = deselect / national view).</summary>
    public void SelectProvince(int provinceId)
    {
        if (provinceId == SelectedProvinceId) return;

        StateVector oldState = CaptureState();
        SelectedProvinceId = provinceId;

        OnProvinceSelected.Invoke(SelectedProvinceId);
        DispatchIfChanged(oldState);
    }

    /// <summary>Set region filter.</summary>
    public void SetRegionFilter(string region)
    {
        if (region == RegionFilter) return;

        StateVector oldState = CaptureState();
        RegionFilter = region;

        OnRegionFilterChanged.Invoke(RegionFilter);
        DispatchIfChanged(oldState);
    }

    /// <summary>Reset to default demo state.</summary>
    public void ResetState()
    {
        StateVector oldState = CaptureState();
        CurrentYear = 2022;
        CurrentIndicator = "ES";
        SelectedProvinceId = 0;
        RegionFilter = "全部";
        DispatchIfChanged(oldState);
    }

    // ── State snapshot ──────────────────────────────────────────

    public StateVector CaptureState()
    {
        return new StateVector
        {
            Year = CurrentYear,
            Indicator = CurrentIndicator,
            ProvinceId = SelectedProvinceId,
            RegionFilter = RegionFilter,
        };
    }

    private void DispatchIfChanged(StateVector oldState)
    {
        StateVector newState = CaptureState();
        if (!newState.Equals(oldState))
        {
            _previousState = newState;
            OnStateChanged.Invoke(newState, oldState);
        }
    }
}

/// <summary>
/// Immutable state vector struct.
/// Thesis Section 3.1 defines: s_t = (y_t, m_t, p_t, r_t, ...)
/// </summary>
[System.Serializable]
public struct StateVector : IEquatable<StateVector>
{
    public int Year;
    public string Indicator;
    public int ProvinceId;
    public string RegionFilter;

    public bool Equals(StateVector other)
    {
        return Year == other.Year
            && Indicator == other.Indicator
            && ProvinceId == other.ProvinceId
            && RegionFilter == other.RegionFilter;
    }

    public override bool Equals(object obj) =>
        obj is StateVector sv && Equals(sv);

    public override int GetHashCode() =>
        HashCode.Combine(Year, Indicator, ProvinceId, RegionFilter);

    public override string ToString() =>
        $"s_t(year={Year}, ind={Indicator}, prov={ProvinceId}, region={RegionFilter})";
}
