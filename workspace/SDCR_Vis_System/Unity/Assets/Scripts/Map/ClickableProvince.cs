using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Attached to each province GameObject for ray/click detection.
/// Routes selection events to ProvinceSelector.
/// </summary>
[RequireComponent(typeof(Collider))]
public class ClickableProvince : MonoBehaviour
{
    public int ProvinceId { get; private set; }
    public string ProvinceName { get; private set; }

    private ProvinceSelector _selector;
    private bool _isInitialized = false;

    public void Initialize(int provinceId, string name, ProvinceSelector selector)
    {
        ProvinceId = provinceId;
        ProvinceName = name;
        _selector = selector;
        _isInitialized = true;
    }

    void OnMouseDown()
    {
        if (_isInitialized && _selector != null)
            _selector.OnProvinceClicked(ProvinceId);
    }

    // For Unity's new input system / XR ray interactor
    public void OnSelectEntered()
    {
        if (_isInitialized && _selector != null)
            _selector.OnProvinceClicked(ProvinceId);
    }
}
