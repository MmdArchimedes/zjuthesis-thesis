using UnityEngine;

/// <summary>
/// Handles province selection logic.
/// Routes clicks to StateManager.SelectProvince().
/// </summary>
public class ProvinceSelector : MonoBehaviour
{
    [SerializeField] private InfoPanelController _infoPanel;

    private StateManager _state;
    private DataManager _data;
    private ChinaMapController _map;

    void Start()
    {
        _state = StateManager.Instance;
        _data = DataManager.Instance;
        _map = GetComponent<ChinaMapController>();
    }

    public void OnProvinceClicked(int provinceId)
    {
        // If same province clicked again → deselect
        if (_state.SelectedProvinceId == provinceId)
        {
            DeselectAll();
            return;
        }

        // Select new province
        _state.SelectProvince(provinceId);

        // Show info panel
        if (_infoPanel != null && _data.IsLoaded)
        {
            var record = _data.GetRecord(provinceId, _state.CurrentYear);
            if (record != null)
                _infoPanel.Show(record);
        }
    }

    public void DeselectAll()
    {
        _state.SelectProvince(0);
        _infoPanel?.Hide();
    }

    /// <summary>Deselect when clicking empty space.</summary>
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            if (!Physics.Raycast(ray, out RaycastHit hit, 50f))
            {
                // Clicked empty space → deselect
                if (_state.SelectedProvinceId != 0)
                    DeselectAll();
            }
        }
    }
}
