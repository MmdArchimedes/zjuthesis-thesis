using UnityEngine;

/// <summary>
/// ChartWidget: Base class for AR chart widgets.
/// Provides common functionality: interaction handling, lifecycle, and state management.
/// </summary>
public class ChartWidget : MonoBehaviour
{
    [Header("Chart Metadata")]
    public string chartId;
    public string chartType;
    public float svasScore;

    [Header("Interaction")]
    public bool enableHover = true;
    public bool enableClick = true;
    public bool enableZoom = false;
    public bool enableBrush = false;

    protected ChartSpecData spec;
    protected bool isHighlighted;
    protected Vector3 originalScale;

    public virtual void Initialize(ChartSpecData chartSpec)
    {
        spec = chartSpec;
        chartId = chartSpec.chartId;
        chartType = chartSpec.chartType;
        originalScale = transform.localScale;

        // Configure interactions
        if (chartSpec.interactions != null)
        {
            foreach (var interaction in chartSpec.interactions)
            {
                switch (interaction.evt)
                {
                    case "OnHover":
                        enableHover = interaction.enabled;
                        break;
                    case "OnClick":
                        enableClick = interaction.enabled;
                        break;
                    case "OnZoom":
                        enableZoom = interaction.enabled;
                        break;
                    case "OnBrush":
                        enableBrush = interaction.enabled;
                        break;
                }
            }
        }
    }

    void OnMouseEnter()
    {
        if (!enableHover) return;
        isHighlighted = true;
        OnHoverEnter();
    }

    void OnMouseExit()
    {
        if (!enableHover) return;
        isHighlighted = false;
        OnHoverExit();
    }

    void OnMouseDown()
    {
        if (!enableClick) return;
        OnChartClicked();
    }

    protected virtual void OnHoverEnter()
    {
        // Highlight effect: slight scale up
        transform.localScale = originalScale * 1.05f;

        // Show tooltip
        var infoPanel = FindObjectOfType<InfoPanel>();
        if (infoPanel != null && spec != null)
        {
            infoPanel.ShowTooltip(
                $"{spec.chartType.ToUpper()} Chart\n" +
                $"SVAS: {svasScore:F3}"
            );
        }
    }

    protected virtual void OnHoverExit()
    {
        transform.localScale = originalScale;

        var infoPanel = FindObjectOfType<InfoPanel>();
        if (infoPanel != null)
            infoPanel.HideTooltip();
    }

    protected virtual void OnChartClicked()
    {
        Debug.Log($"[ChartWidget] Clicked: {chartId} ({chartType})");
        // Drill-down: could request a more detailed chart
    }

    public virtual void RefreshData(System.Collections.Generic.Dictionary<string, float> newData)
    {
        var renderer = GetComponent<ChartRenderer>();
        if (renderer != null && spec != null)
        {
            renderer.RenderChart(spec, newData);
        }
    }

    public virtual void Dispose()
    {
        Destroy(gameObject);
    }
}

/// <summary>
/// Placeholder InfoPanel reference.
/// </summary>
public class InfoPanel : MonoBehaviour
{
    public void ShowTooltip(string text) { }
    public void HideTooltip() { }
}
