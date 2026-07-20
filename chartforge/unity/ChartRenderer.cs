using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// ChartRenderer: Converts ChartForge API ChartSpec into Unity UI visuals.
/// Supports bar, line, pie, and heatmap chart types in AR world space.
/// </summary>
public class ChartRenderer : MonoBehaviour
{
    [Header("Rendering Settings")]
    public GameObject barPrefab;
    public GameObject pointPrefab;
    public GameObject labelPrefab;
    public Material defaultMaterial;
    public float chartWidth = 0.8f;
    public float chartHeight = 0.6f;

    [Header("Colors")]
    public Color[] colorPalette;

    private ChartSpecData _currentSpec;
    private GameObject _chartContainer;
    private Dictionary<string, float> _dataCache = new Dictionary<string, float>();

    void Awake()
    {
        if (colorPalette == null || colorPalette.Length == 0)
        {
            colorPalette = new Color[]
            {
                new Color(0.306f, 0.475f, 0.655f), // #4E79A7
                new Color(0.949f, 0.557f, 0.169f), // #F28E2B
                new Color(0.882f, 0.341f, 0.349f), // #E15759
                new Color(0.463f, 0.718f, 0.698f), // #76B7B2
                new Color(0.349f, 0.631f, 0.310f), // #59A14F
                new Color(0.929f, 0.788f, 0.282f), // #EDC948
            };
        }
    }

    /// <summary>
    /// Render a chart from ChartForge API response.
    /// </summary>
    public void RenderChart(ChartSpecData spec, Dictionary<string, float> data = null)
    {
        _currentSpec = spec;
        if (data != null) _dataCache = data;

        // Clear previous chart
        if (_chartContainer != null)
            Destroy(_chartContainer);
        _chartContainer = new GameObject("ChartContainer");
        _chartContainer.transform.SetParent(transform);
        _chartContainer.transform.localPosition = Vector3.zero;

        // Position chart in AR space
        if (spec.layout != null)
        {
            _chartContainer.transform.localPosition = new Vector3(
                spec.layout.position.x,
                spec.layout.position.y,
                spec.layout.position.z
            );
        }

        // Dispatch by chart type
        switch (spec.chartType.ToLower())
        {
            case "bar":
                RenderBarChart(spec);
                break;
            case "line":
                RenderLineChart(spec);
                break;
            case "pie":
                RenderPieChart(spec);
                break;
            case "scatter":
                RenderScatterChart(spec);
                break;
            case "heatmap":
                RenderHeatmapChart(spec);
                break;
            default:
                RenderBarChart(spec); // fallback
                break;
        }

        // Add annotations
        RenderAnnotations(spec);

        Debug.Log($"[ChartRenderer] Rendered {spec.chartType} chart " +
                 $"({spec.encodings?.x?.field} vs {spec.encodings?.y?.field})");
    }

    private void RenderBarChart(ChartSpecData spec)
    {
        if (_dataCache.Count == 0)
        {
            // Use dummy data for preview
            GenerateDummyData(spec);
        }

        float barSpacing = chartWidth / _dataCache.Count;
        float maxVal = 0;
        foreach (var v in _dataCache.Values)
            if (v > maxVal) maxVal = v;

        int i = 0;
        foreach (var kvp in _dataCache)
        {
            float barHeight = (kvp.Value / maxVal) * chartHeight;
            Vector3 position = _chartContainer.transform.position +
                new Vector3(
                    -chartWidth / 2 + barSpacing * i + barSpacing / 2,
                    barHeight / 2,
                    0
                );

            GameObject bar = Instantiate(barPrefab, position, Quaternion.identity,
                _chartContainer.transform);
            bar.transform.localScale = new Vector3(barSpacing * 0.7f, barHeight, 0.05f);

            // Color
            var renderer = bar.GetComponent<Renderer>();
            if (renderer != null)
                renderer.material.color = colorPalette[i % colorPalette.Length];

            // Label
            CreateLabel(kvp.Key, position + Vector3.down * 0.1f);
            CreateLabel(kvp.Value.ToString("F2"), position + Vector3.up * (barHeight / 2 + 0.03f));

            i++;
        }
    }

    private void RenderLineChart(ChartSpecData spec)
    {
        // Simplified line chart using connected points
        if (_dataCache.Count == 0) GenerateDummyData(spec);

        var lineRenderer = _chartContainer.AddComponent<LineRenderer>();
        lineRenderer.positionCount = _dataCache.Count;
        lineRenderer.startWidth = 0.01f;
        lineRenderer.endWidth = 0.01f;
        lineRenderer.material = defaultMaterial;
        lineRenderer.material.color = colorPalette[0];

        float pointSpacing = chartWidth / (_dataCache.Count - 1);
        float maxVal = 0;
        foreach (var v in _dataCache.Values)
            if (v > maxVal) maxVal = v;

        int i = 0;
        foreach (var kvp in _dataCache)
        {
            Vector3 position = _chartContainer.transform.position +
                new Vector3(-chartWidth / 2 + pointSpacing * i,
                           (kvp.Value / maxVal) * chartHeight - chartHeight / 2,
                           -0.01f);
            lineRenderer.SetPosition(i, position);

            // Create point marker
            if (pointPrefab != null)
            {
                Instantiate(pointPrefab, position, Quaternion.identity,
                    _chartContainer.transform);
            }

            i++;
        }
    }

    private void RenderPieChart(ChartSpecData spec)
    {
        if (_dataCache.Count == 0) GenerateDummyData(spec);

        float total = 0;
        foreach (var v in _dataCache.Values) total += v;

        float startAngle = 0;
        int i = 0;
        foreach (var kvp in _dataCache)
        {
            float fraction = kvp.Value / total;
            float angle = fraction * 360f;

            // Create pie slice (simplified as colored quad sectors)
            GameObject slice = GameObject.CreatePrimitive(PrimitiveType.Quad);
            slice.transform.SetParent(_chartContainer.transform);
            slice.transform.localPosition = _chartContainer.transform.position;
            slice.transform.localRotation = Quaternion.Euler(0, 0, startAngle + angle / 2);
            slice.transform.localScale = new Vector3(0.3f, 0.3f, 1);

            var renderer = slice.GetComponent<Renderer>();
            if (renderer != null)
                renderer.material.color = colorPalette[i % colorPalette.Length];

            startAngle += angle;
            i++;
        }
    }

    private void RenderScatterChart(ChartSpecData spec)
    {
        if (_dataCache.Count == 0) GenerateDummyData(spec);

        float maxX = 0, maxY = 0;
        foreach (var kvp in _dataCache)
        {
            if (kvp.Value > maxY) maxY = kvp.Value;
        }
        maxX = _dataCache.Count;

        int i = 0;
        foreach (var kvp in _dataCache)
        {
            Vector3 position = _chartContainer.transform.position +
                new Vector3(
                    -chartWidth / 2 + (i / (float)_dataCache.Count) * chartWidth,
                    (kvp.Value / maxY) * chartHeight - chartHeight / 2,
                    0
                );

            if (pointPrefab != null)
            {
                var point = Instantiate(pointPrefab, position, Quaternion.identity,
                    _chartContainer.transform);
                var renderer = point.GetComponent<Renderer>();
                if (renderer != null)
                    renderer.material.color = colorPalette[0];
            }
            i++;
        }
    }

    private void RenderHeatmapChart(ChartSpecData spec)
    {
        // Simplified grid of colored quads
        int gridSize = Mathf.CeilToInt(Mathf.Sqrt(_dataCache.Count));
        float cellSize = Mathf.Min(chartWidth, chartHeight) / gridSize;

        if (_dataCache.Count == 0) GenerateDummyData(spec);

        float minVal = float.MaxValue, maxVal = float.MinValue;
        foreach (var v in _dataCache.Values)
        {
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }

        int i = 0;
        foreach (var kvp in _dataCache)
        {
            int row = i / gridSize;
            int col = i % gridSize;
            float t = (kvp.Value - minVal) / (maxVal - minVal + 0.001f);

            Vector3 position = _chartContainer.transform.position +
                new Vector3(
                    -chartWidth / 2 + col * cellSize + cellSize / 2,
                    -chartHeight / 2 + row * cellSize + cellSize / 2,
                    0
                );

            GameObject cell = GameObject.CreatePrimitive(PrimitiveType.Quad);
            cell.transform.SetParent(_chartContainer.transform);
            cell.transform.position = position;
            cell.transform.localScale = new Vector3(cellSize * 0.9f, cellSize * 0.9f, 1);

            var renderer = cell.GetComponent<Renderer>();
            if (renderer != null)
                renderer.material.color = Color.Lerp(Color.blue, Color.red, t);

            i++;
        }
    }

    private void RenderAnnotations(ChartSpecData spec)
    {
        if (spec.annotations == null) return;

        float yOffset = chartHeight / 2 + 0.05f;
        foreach (var annotation in spec.annotations)
        {
            Vector3 pos = _chartContainer.transform.position +
                new Vector3(0, yOffset, 0);
            CreateLabel(annotation.text, pos);
            yOffset -= 0.04f;
        }
    }

    private void CreateLabel(string text, Vector3 position)
    {
        if (labelPrefab == null) return;

        GameObject label = Instantiate(labelPrefab, position, Quaternion.identity,
            _chartContainer.transform);
        var textComponent = label.GetComponent<Text>();
        if (textComponent != null)
            textComponent.text = text;

        // Make label face camera in AR
        label.transform.LookAt(Camera.main.transform);
        label.transform.Rotate(0, 180, 0);
    }

    private void GenerateDummyData(ChartSpecData spec)
    {
        // Generate placeholder data for rendering preview
        _dataCache.Clear();
        string[] provinces = { "北京", "上海", "广东", "浙江", "江苏", "山东" };
        for (int i = 0; i < 6; i++)
        {
            _dataCache[provinces[i]] = Random.Range(0.1f, 0.9f);
        }
    }

    public void Clear()
    {
        if (_chartContainer != null)
        {
            Destroy(_chartContainer);
            _chartContainer = null;
        }
        _dataCache.Clear();
    }
}
