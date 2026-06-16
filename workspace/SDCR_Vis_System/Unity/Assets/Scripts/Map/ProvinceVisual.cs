using UnityEngine;

/// <summary>
/// Per-province visual state: color, height, highlight, dimming.
/// Driven by SDCR pipeline (thesis Eq 4-5 mapping).
/// </summary>
public class ProvinceVisual : MonoBehaviour
{
    public int ProvinceId { get; private set; }
    public string ProvinceName { get; private set; }

    private Renderer _renderer;
    private Material _material;
    private Color _baseColor = Color.white;
    private float _baseHeight;

    // Visual state
    private bool _isHighlighted;
    private bool _isDimmed;
    private Color _currentColor;

    void Awake()
    {
        _renderer = GetComponent<Renderer>();
        if (_renderer != null)
            _material = _renderer.material;
    }

    public void Initialize(int provinceId, string name)
    {
        ProvinceId = provinceId;
        ProvinceName = name;
        _baseHeight = transform.localPosition.y;
    }

    // ── Visual Updates ──────────────────────────────────────────

    public void SetColor(Color color)
    {
        _currentColor = color;
        ApplyVisualState();
    }

    public void SetHeight(float worldHeight)
    {
        Vector3 pos = transform.localPosition;
        pos.y = worldHeight;
        transform.localPosition = pos;
    }

    public void SetHighlighted(bool highlighted)
    {
        if (_isHighlighted == highlighted) return;
        _isHighlighted = highlighted;

        // Scale pulse on highlight
        if (highlighted)
        {
            transform.localScale = Vector3.one * 0.16f; // slightly larger
        }
        else
        {
            transform.localScale = Vector3.one * 0.12f; // back to normal
        }

        ApplyVisualState();
    }

    public void SetDimmed(bool dimmed)
    {
        if (_isDimmed == dimmed) return;
        _isDimmed = dimmed;
        ApplyVisualState();
    }

    private void ApplyVisualState()
    {
        if (_material == null) return;

        Color finalColor = _currentColor;

        // Highlight: add emission-like brightening
        if (_isHighlighted)
        {
            finalColor = Color.Lerp(finalColor, Color.white, 0.3f);
            _material.SetColor("_EmissionColor", finalColor * 0.4f);
            _material.EnableKeyword("_EMISSION");
        }
        else
        {
            _material.SetColor("_EmissionColor", Color.black);
            _material.DisableKeyword("_EMISSION");
        }

        // Dimming: reduce saturation and brightness for filtered-out provinces
        if (_isDimmed)
        {
            finalColor = Color.Lerp(finalColor, Color.gray, 0.6f);
            finalColor.a = 0.4f;
        }
        else
        {
            finalColor.a = 1f;
        }

        _material.color = finalColor;
    }

    // Reset to default
    public void ResetVisual()
    {
        _isHighlighted = false;
        _isDimmed = false;
        transform.localScale = Vector3.one * 0.12f;
        Vector3 pos = transform.localPosition;
        pos.y = _baseHeight;
        transform.localPosition = pos;
        ApplyVisualState();
    }
}
