using UnityEngine;
using System;

/// <summary>
/// ChartStateSync: Synchronizes ChartForge charts with the thesis state vector s_t.
///
/// The state vector s_t = {year, province, indicator, region, ...} is shared
/// between SDCR-Vis views (province map, timeline, panels) and ChartForge charts.
/// When state changes, relevant charts auto-refresh.
///
/// Integrates with TSTQ-Fusion (Ch3) for conflict resolution and
/// SDCR-Vis (Ch4) for multi-view coordination.
/// </summary>
public class ChartStateSync : MonoBehaviour
{
    [Header("State")]
    public int currentYear = 2022;
    public string currentProvince = "";
    public string currentIndicator = "DEL";  // DEL or ES
    public string currentRegion = "";        // empty = national

    [Header("References")]
    public ChartForgeClient chartForgeClient;
    public ChartRenderer chartRenderer;

    // Event: fired when state changes
    public event Action<StateVector> OnStateChanged;

    [Serializable]
    public struct StateVector
    {
        public int year;
        public string province;
        public string indicator;
        public string region;
    }

    void Start()
    {
        if (chartForgeClient == null)
            chartForgeClient = GetComponent<ChartForgeClient>();
        if (chartRenderer == null)
            chartRenderer = GetComponent<ChartRenderer>();

        if (chartForgeClient != null)
            chartForgeClient.OnChartGenerated += OnChartReceived;
    }

    /// <summary>
    /// Update state and refresh charts.
    /// Called by TSTQ-Fusion after conflict resolution.
    /// </summary>
    public void UpdateState(StateVector newState)
    {
        bool changed = false;

        if (newState.year != currentYear) { currentYear = newState.year; changed = true; }
        if (newState.province != currentProvince) { currentProvince = newState.province; changed = true; }
        if (newState.indicator != currentIndicator) { currentIndicator = newState.indicator; changed = true; }
        if (newState.region != currentRegion) { currentRegion = newState.region; changed = true; }

        if (!changed) return;

        var state = new StateVector
        {
            year = currentYear,
            province = currentProvince,
            indicator = currentIndicator,
            region = currentRegion,
        };

        OnStateChanged?.Invoke(state);
        RefreshRelevantCharts(state);
    }

    private void RefreshRelevantCharts(StateVector state)
    {
        // Build NL query from current state
        string query = BuildQueryFromState(state);

        if (!string.IsNullOrEmpty(query) && chartForgeClient != null)
        {
            Debug.Log($"[ChartStateSync] Refreshing chart for state: " +
                     $"year={state.year}, indicator={state.indicator}, " +
                     $"region={state.region}");
            chartForgeClient.RequestChart(query);
        }
    }

    private string BuildQueryFromState(StateVector state)
    {
        if (!string.IsNullOrEmpty(state.province))
        {
            // Province-level view
            return $"Show line chart of {state.indicator} trend for {state.province} from 2014 to {state.year}";
        }

        if (!string.IsNullOrEmpty(state.region))
        {
            // Region-filtered view
            return $"Show bar chart of {state.indicator} across provinces in {state.region} region for {state.year}";
        }

        // National view
        return $"Show bar chart comparing {state.indicator} across all provinces for {state.year}";
    }

    private void OnChartReceived(string query, ChartSpecData spec)
    {
        // Update chart with actual data from DataManager
        var dataManager = FindObjectOfType<DataManager>();
        if (dataManager != null)
        {
            var data = dataManager.GetDataForChart(
                currentIndicator, currentYear, currentRegion);
            chartRenderer.RenderChart(spec, data);
        }
        else
        {
            chartRenderer.RenderChart(spec);
        }
    }

    /// <summary>
    /// Get current state as vector (read by SDCR-Vis for multi-view refresh).
    /// </summary>
    public StateVector GetCurrentState()
    {
        return new StateVector
        {
            year = currentYear,
            province = currentProvince,
            indicator = currentIndicator,
            region = currentRegion,
        };
    }
}

/// <summary>
/// Placeholder DataManager reference — replace with actual thesis DataManager.
/// </summary>
public class DataManager : MonoBehaviour
{
    public System.Collections.Generic.Dictionary<string, float> GetDataForChart(
        string indicator, int year, string region = "")
    {
        // In production: query the CSV/JSON data engine
        // that loads Ch5 empirical results
        var data = new System.Collections.Generic.Dictionary<string, float>();
        // Placeholder values
        return data;
    }
}
