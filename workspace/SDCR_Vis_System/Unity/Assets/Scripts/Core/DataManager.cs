using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Data layer: loads CSV/JSON assets from Resources/Data/ and builds
/// in-memory indices for O(1) query by (province, year) key.
///
/// Thesis reference: Section 4.2 Layer 2 (Data Layer).
/// </summary>
[DefaultExecutionOrder(-90)]
public class DataManager : MonoBehaviour
{
    // ── Data Structures ─────────────────────────────────────────

    /// <summary>Panel data record for one province-year observation.</summary>
    [System.Serializable]
    public class IndicatorRecord
    {
        public int ProvinceId;
        public string ProvinceName;
        public int Year;
        public float ES;
        public float DEL;
        public float PGDP;
        public float URBAN;
        public float INDS;
        public float LEOP;
        public float FDE;
        public float DEN;
        public float TEIN;
        public string RegionTag;
    }

    /// <summary>Regression model result.</summary>
    [System.Serializable]
    public class RegressionModel
    {
        public string name;
        public int n;
        public float r_squared;
        public List<Coefficient> coefficients;
    }

    [System.Serializable]
    public class Coefficient
    {
        public string variable;
        public float estimate;
        public float se;
        public string significance;
    }

    [System.Serializable]
    public class RegionResult
    {
        public string region;
        public int n;
        public float coef_del;
        public float se;
        public string significance;
    }

    /// <summary>Mechanism graph node.</summary>
    [System.Serializable]
    public class MechanismNode
    {
        public string id;
        public string label;
        public string color;
        public float x;
        public float y;
    }

    [System.Serializable]
    public class MechanismEdge
    {
        public string from;
        public string to;
        public string label;
    }

    // ── Indexed Data ────────────────────────────────────────────

    /// <summary>Key: (provinceId, year) → record. O(1) lookup.</summary>
    private Dictionary<(int, int), IndicatorRecord> _recordIndex;

    /// <summary>All records in flat list (for iteration).</summary>
    public List<IndicatorRecord> AllRecords { get; private set; }

    /// <summary>Province ID → Province Name.</summary>
    public Dictionary<int, string> ProvinceNames { get; private set; }

    /// <summary>Province ID → Region Tag.</summary>
    public Dictionary<int, string> ProvinceRegions { get; private set; }

    /// <summary>Loaded regression results.</summary>
    public List<RegressionModel> BaselineModels { get; private set; }
    public List<RegressionModel> MediationModels { get; private set; }
    public List<RegionResult> HeterogeneityRegions { get; private set; }
    public float TurningPoint { get; private set; } = 0.507f;

    /// <summary>Mechanism graph data.</summary>
    public List<MechanismNode> MechNodes { get; private set; }
    public List<MechanismEdge> MechEdges { get; private set; }

    public UnityEvent OnDataLoaded = new UnityEvent();
    public bool IsLoaded { get; private set; }

    public static DataManager Instance { get; private set; }

    // ── Unity Lifecycle ─────────────────────────────────────────

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    void Start()
    {
        LoadAllData();
    }

    // ── Loading Pipeline ────────────────────────────────────────

    public void LoadAllData()
    {
        LoadPanelData();
        LoadRegressionResults();
        LoadMechanismGraph();
        IsLoaded = true;
        OnDataLoaded.Invoke();
        Debug.Log($"[DataManager] Loaded {AllRecords.Count} panel records, "
                  + $"{ProvinceNames.Count} provinces.");
    }

    private void LoadPanelData()
    {
        _recordIndex = new Dictionary<(int, int), IndicatorRecord>();
        AllRecords = new List<IndicatorRecord>();
        ProvinceNames = new Dictionary<int, string>();
        ProvinceRegions = new Dictionary<int, string>();

        TextAsset csv = Resources.Load<TextAsset>("Data/panel_data");
        if (csv == null)
        {
            Debug.LogError("[DataManager] panel_data.csv not found in Resources/Data/");
            return;
        }

        string[] lines = csv.text.Split('\n');
        // Skip header line
        for (int i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i])) continue;

            string[] cols = lines[i].Split(',');
            if (cols.Length < 12) continue;

            var record = new IndicatorRecord
            {
                ProvinceId = int.Parse(cols[0]),
                ProvinceName = cols[1],
                Year = int.Parse(cols[2]),
                ES = float.Parse(cols[3]),
                DEL = float.Parse(cols[4]),
                PGDP = float.Parse(cols[5]),
                URBAN = float.Parse(cols[6]),
                INDS = float.Parse(cols[7]),
                LEOP = float.Parse(cols[8]),
                FDE = float.Parse(cols[9]),
                DEN = float.Parse(cols[10]),
                TEIN = float.Parse(cols[11]),
                RegionTag = cols[12],
            };

            _recordIndex[(record.ProvinceId, record.Year)] = record;
            AllRecords.Add(record);

            if (!ProvinceNames.ContainsKey(record.ProvinceId))
            {
                ProvinceNames[record.ProvinceId] = record.ProvinceName;
                ProvinceRegions[record.ProvinceId] = record.RegionTag;
            }
        }
    }

    private void LoadRegressionResults()
    {
        BaselineModels = new List<RegressionModel>();
        MediationModels = new List<RegressionModel>();
        HeterogeneityRegions = new List<RegionResult>();

        TextAsset json = Resources.Load<TextAsset>("Data/regression_results");
        if (json == null)
        {
            Debug.LogWarning("[DataManager] regression_results.json not found");
            return;
        }

        try
        {
            var root = JsonUtility.FromJson<RegressionResultsWrapper>(json.text);

            // Parse baseline
            if (root.baseline?.models != null)
            {
                foreach (var m in root.baseline.models)
                {
                    var model = new RegressionModel
                    {
                        name = m.name,
                        n = m.n,
                        r_squared = m.r_squared,
                        coefficients = new List<Coefficient>(),
                    };
                    if (m.coefficients != null)
                    {
                        foreach (var c in m.coefficients)
                            model.coefficients.Add(new Coefficient
                            {
                                variable = c.variable,
                                estimate = c.estimate,
                                se = c.se,
                                significance = c.significance,
                            });
                    }
                    BaselineModels.Add(model);
                    if (m.turning_point > 0)
                        TurningPoint = m.turning_point;
                }
            }

            // Parse heterogeneity
            if (root.heterogeneity?.regions != null)
            {
                foreach (var r in root.heterogeneity.regions)
                    HeterogeneityRegions.Add(new RegionResult
                    {
                        region = r.region,
                        n = r.n,
                        coef_del = r.coef_del,
                        se = r.se,
                        significance = r.significance,
                    });
            }

            // Parse mediation
            if (root.mediation?.models != null)
            {
                foreach (var m in root.mediation.models)
                {
                    var model = new RegressionModel
                    {
                        name = m.name,
                        coefficients = new List<Coefficient>(),
                    };
                    if (m.coefficients != null)
                        foreach (var c in m.coefficients)
                            model.coefficients.Add(new Coefficient
                            {
                                variable = c.variable,
                                estimate = c.estimate,
                                se = c.se,
                                significance = c.significance,
                            });
                    MediationModels.Add(model);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[DataManager] Failed to parse regression JSON: {e.Message}");
        }
    }

    private void LoadMechanismGraph()
    {
        MechNodes = new List<MechanismNode>();
        MechEdges = new List<MechanismEdge>();

        TextAsset json = Resources.Load<TextAsset>("Data/mechanism_paths");
        if (json == null) return;

        try
        {
            var root = JsonUtility.FromJson<MechanismGraphWrapper>(json.text);
            if (root.nodes != null) MechNodes = root.nodes;
            if (root.edges != null) MechEdges = root.edges;
        }
        catch (Exception e)
        {
            Debug.LogError($"[DataManager] Failed to parse mechanism JSON: {e.Message}");
        }
    }

    // ── Query API (matching thesis Data Layer) ──────────────────

    /// <summary>Get value of current indicator for a province in a given year.</summary>
    public float GetValue(int provinceId, int year, string indicator)
    {
        if (_recordIndex.TryGetValue((provinceId, year), out var record))
            return indicator == "ES" ? record.ES : record.DEL;
        return 0f;
    }

    /// <summary>Get full record for (province, year).</summary>
    public IndicatorRecord GetRecord(int provinceId, int year)
    {
        _recordIndex.TryGetValue((provinceId, year), out var record);
        return record;
    }

    /// <summary>Get all records for a given year (optionally filtered by region).</summary>
    public List<IndicatorRecord> GetYearRecords(int year, string regionFilter = "全部")
    {
        var filtered = new List<IndicatorRecord>();
        foreach (var record in AllRecords)
        {
            if (record.Year != year) continue;
            if (regionFilter != "全部" && record.RegionTag != regionFilter) continue;
            filtered.Add(record);
        }
        return filtered;
    }

    /// <summary>Get min/max of an indicator for normalization (Eq 4-4 in thesis).</summary>
    public (float min, float max) GetValueRange(string indicator, string regionFilter = "全部")
    {
        float min = float.MaxValue, max = float.MinValue;
        foreach (var record in AllRecords)
        {
            if (regionFilter != "全部" && record.RegionTag != regionFilter) continue;
            float val = indicator == "ES" ? record.ES : record.DEL;
            if (val < min) min = val;
            if (val > max) max = val;
        }
        return (min, max);
    }

    /// <summary>Get all province IDs (1-30).</summary>
    public List<int> GetAllProvinceIds()
    {
        return ProvinceNames.Keys.OrderBy(k => k).ToList();
    }

    /// <summary>Get province IDs filtered by region.</summary>
    public List<int> GetProvinceIdsByRegion(string region)
    {
        if (region == "全部")
            return GetAllProvinceIds();
        return ProvinceRegions
            .Where(kv => kv.Value == region)
            .Select(kv => kv.Key)
            .OrderBy(k => k)
            .ToList();
    }

    // ── JSON Serialization Helpers ──────────────────────────────

    [System.Serializable]
    private class RegressionResultsWrapper
    {
        public BaselineWrapper baseline;
        public MediationWrapper mediation;
        public HeterogeneityWrapper heterogeneity;
    }

    [System.Serializable]
    private class BaselineWrapper
    {
        public List<ModelWrapper> models;
    }

    [System.Serializable]
    private class ModelWrapper
    {
        public string name;
        public int n;
        public float r_squared;
        public float turning_point;
        public List<Coefficient> coefficients;
    }

    [System.Serializable]
    private class MediationWrapper
    {
        public List<ModelWrapper> models;
    }

    [System.Serializable]
    private class HeterogeneityWrapper
    {
        public List<RegionResult> regions;
    }

    [System.Serializable]
    private class MechanismGraphWrapper
    {
        public List<MechanismNode> nodes;
        public List<MechanismEdge> edges;
    }
}
