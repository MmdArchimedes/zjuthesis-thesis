using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Manages the 3D provincial map: generates province GameObjects,
/// coordinates color/height updates driven by SDCR pipeline.
///
/// For demo: uses simplified rectangle-based province representations.
/// In production: would load pre-made FBX meshes with proper boundaries.
/// </summary>
public class ChinaMapController : MonoBehaviour
{
    [Header("Map Layout")]
    [SerializeField] private GameObject _provincePrefab;
    [SerializeField] private Transform _mapRoot;
    [SerializeField] private float _mapWidth = 4.0f;
    [SerializeField] private float _mapHeight = 3.0f;
    [SerializeField] private Material _provinceMaterialTemplate;

    [Header("Interaction")]
    [SerializeField] private ProvinceSelector _provinceSelector;

    // Runtime state
    private Dictionary<int, ProvinceVisual> _provinceVisuals = new Dictionary<int, ProvinceVisual>();
    private Dictionary<int, GameObject> _provinceObjects = new Dictionary<int, GameObject>();

    private StateManager _state;
    private DataManager _data;
    private bool _initialized = false;

    // Simplified province positions (normalized 0-1, using approximate geographic centroids)
    // These map to the 30 provinces in panel_data.csv (IDs 1-30)
    private static readonly Dictionary<int, Vector2> ProvincePositions = new Dictionary<int, Vector2>
    {
        {1,  new Vector2(0.795f, 0.720f)}, // 北京
        {2,  new Vector2(0.810f, 0.690f)}, // 天津
        {3,  new Vector2(0.770f, 0.650f)}, // 河北
        {4,  new Vector2(0.730f, 0.620f)}, // 山西
        {5,  new Vector2(0.650f, 0.750f)}, // 内蒙古
        {6,  new Vector2(0.830f, 0.800f)}, // 辽宁
        {7,  new Vector2(0.870f, 0.840f)}, // 吉林
        {8,  new Vector2(0.900f, 0.920f)}, // 黑龙江
        {9,  new Vector2(0.830f, 0.570f)}, // 上海
        {10, new Vector2(0.780f, 0.530f)}, // 江苏
        {11, new Vector2(0.800f, 0.490f)}, // 浙江
        {12, new Vector2(0.760f, 0.500f)}, // 安徽
        {13, new Vector2(0.790f, 0.410f)}, // 福建
        {14, new Vector2(0.740f, 0.420f)}, // 江西
        {15, new Vector2(0.800f, 0.600f)}, // 山东
        {16, new Vector2(0.720f, 0.550f)}, // 河南
        {17, new Vector2(0.700f, 0.460f)}, // 湖北
        {18, new Vector2(0.680f, 0.380f)}, // 湖南
        {19, new Vector2(0.720f, 0.280f)}, // 广东
        {20, new Vector2(0.620f, 0.280f)}, // 广西
        {21, new Vector2(0.660f, 0.150f)}, // 海南
        {22, new Vector2(0.580f, 0.450f)}, // 重庆
        {23, new Vector2(0.480f, 0.400f)}, // 四川
        {24, new Vector2(0.550f, 0.320f)}, // 贵州
        {25, new Vector2(0.420f, 0.280f)}, // 云南
        {26, new Vector2(0.580f, 0.550f)}, // 陕西
        {27, new Vector2(0.450f, 0.600f)}, // 甘肃
        {28, new Vector2(0.320f, 0.620f)}, // 青海
        {29, new Vector2(0.520f, 0.620f)}, // 宁夏
        {30, new Vector2(0.180f, 0.720f)}, // 新疆
    };

    void Start()
    {
        _state = StateManager.Instance;
        _data = DataManager.Instance;

        if (_data.IsLoaded)
            InitializeMap();
        else
            _data.OnDataLoaded.AddListener(InitializeMap);
    }

    void OnDestroy()
    {
        if (_data != null)
            _data.OnDataLoaded.RemoveListener(InitializeMap);
    }

    // ── Map Generation ──────────────────────────────────────────

    private void InitializeMap()
    {
        if (_initialized) return;
        _initialized = true;

        if (_mapRoot == null)
        {
            _mapRoot = new GameObject("ChinaMapRoot").transform;
            _mapRoot.SetParent(transform);
        }

        foreach (var pid in _data.GetAllProvinceIds())
        {
            CreateProvinceObject(pid);
        }

        // Center map
        _mapRoot.localPosition = new Vector3(0, 0, 2.0f);

        // Full initial refresh
        var s_t = _state.CaptureState();
        FindObjectOfType<SDCRPipeline>()?.RefreshAllViews(s_t);

        Debug.Log($"[ChinaMap] Generated {_provinceObjects.Count} province objects");
    }

    private void CreateProvinceObject(int provinceId)
    {
        if (!ProvincePositions.TryGetValue(provinceId, out var pos)) return;
        if (!_data.ProvinceNames.TryGetValue(provinceId, out var name)) return;

        // Create province GameObject
        GameObject go;
        if (_provincePrefab != null)
        {
            go = Instantiate(_provincePrefab, _mapRoot);
            go.name = $"Province_{provinceId:D2}_{name}";
        }
        else
        {
            go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.transform.SetParent(_mapRoot);
            go.name = $"Province_{provinceId:D2}_{name}";
        }

        // Position: map normalized coords → world space
        float worldX = (pos.x - 0.5f) * _mapWidth;
        float worldY = 0f; // base height, will be updated by SDCR
        float worldZ = -(pos.y - 0.5f) * _mapHeight;
        go.transform.localPosition = new Vector3(worldX, worldY, worldZ);

        // Size: proportional to map, slightly spaced
        float size = 0.12f;
        go.transform.localScale = Vector3.one * size;

        // Add/configure ProvinceVisual component
        var visual = go.GetComponent<ProvinceVisual>();
        if (visual == null) visual = go.AddComponent<ProvinceVisual>();
        visual.Initialize(provinceId, name);

        // Setup collider for click detection
        var collider = go.GetComponent<Collider>();
        if (collider == null) collider = go.AddComponent<BoxCollider>();

        // Setup material
        var renderer = go.GetComponent<Renderer>();
        if (renderer != null && _provinceMaterialTemplate != null)
        {
            renderer.material = new Material(_provinceMaterialTemplate);
        }

        _provinceVisuals[provinceId] = visual;
        _provinceObjects[provinceId] = go;

        // Register with selector
        if (_provinceSelector != null)
        {
            var clickable = go.GetComponent<ClickableProvince>();
            if (clickable == null) clickable = go.AddComponent<ClickableProvince>();
            clickable.Initialize(provinceId, name, _provinceSelector);
        }
    }

    // ── SDCR Update Interface ───────────────────────────────────

    /// <summary>
    /// Apply color and height updates from SDCR pipeline.
    /// Called by SDCRPipeline.RefreshAllViews().
    /// </summary>
    public void ApplyColorHeightUpdate(
        Dictionary<int, Color> colors,
        Dictionary<int, float> heights,
        StateVector s_t)
    {
        foreach (var kv in colors)
        {
            int pid = kv.Key;
            if (!_provinceVisuals.ContainsKey(pid)) continue;

            // Color update (thesis Eq 4-5: c_i = Lerp(c_min, c_max, n_i))
            _provinceVisuals[pid].SetColor(kv.Value);

            // Height update (thesis Eq 4-5: h_i = h_min + λ_h * n_i)
            if (heights.TryGetValue(pid, out float h))
            {
                _provinceVisuals[pid].SetHeight(h);
            }

            // Region dimming for non-filtered provinces
            if (s_t.RegionFilter != "全部")
            {
                var record = _data.GetRecord(pid, s_t.Year);
                bool inRegion = record != null && record.RegionTag == s_t.RegionFilter;
                _provinceVisuals[pid].SetDimmed(!inRegion);
            }
            else
            {
                _provinceVisuals[pid].SetDimmed(false);
            }
        }

        // Highlight selected province
        if (s_t.ProvinceId > 0 && _provinceVisuals.ContainsKey(s_t.ProvinceId))
        {
            _provinceVisuals[s_t.ProvinceId].SetHighlighted(true);
        }
    }

    // ── Public Accessors ────────────────────────────────────────

    public ProvinceVisual GetProvinceVisual(int provinceId)
    {
        _provinceVisuals.TryGetValue(provinceId, out var v);
        return v;
    }

    public Vector3 GetProvinceWorldPosition(int provinceId)
    {
        if (_provinceObjects.TryGetValue(provinceId, out var go))
            return go.transform.position;
        return Vector3.zero;
    }
}
