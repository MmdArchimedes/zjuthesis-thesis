using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Province detail info card shown when a province is selected.
/// Displays current year's ES, DEL, region, and key statistics.
/// </summary>
public class InfoPanelController : MonoBehaviour
{
    [Header("UI Elements")]
    [SerializeField] private GameObject _panelRoot;
    [SerializeField] private Text _provinceNameText;
    [SerializeField] private Text _regionTagText;
    [SerializeField] private Text _esValueText;
    [SerializeField] private Text _delValueText;
    [SerializeField] private Text _additionalInfoText;
    [SerializeField] private Button _closeButton;

    private void Start()
    {
        if (_closeButton != null)
            _closeButton.onClick.AddListener(Hide);
        Hide();
    }

    /// <summary>Show info card for a province-year record.</summary>
    public void Show(DataManager.IndicatorRecord record)
    {
        if (_panelRoot != null) _panelRoot.SetActive(true);

        if (_provinceNameText != null)
            _provinceNameText.text = $"{record.ProvinceName}";

        if (_regionTagText != null)
        {
            string regionLabel = record.RegionTag switch
            {
                "东部" => "东部地区",
                "中部" => "中部地区",
                "西部" => "西部地区",
                "东北" => "东北地区",
                _ => record.RegionTag,
            };
            _regionTagText.text = $"区域: {regionLabel}  ·  {record.Year}年";
        }

        if (_esValueText != null)
            _esValueText.text = $"<b>ES</b> (能源结构优化): {record.ES:F4}";

        if (_delValueText != null)
            _delValueText.text = $"<b>DEL</b> (数字经济发展): {record.DEL:F4}";

        if (_additionalInfoText != null)
        {
            string turningNote = record.DEL > 0.507f
                ? $"<color=#FF6600>⚠ DEL({record.DEL:F3}) 已越过倒U型拐点(0.507)，边际效应趋缓</color>"
                : $"DEL({record.DEL:F3}) 处于拐点(0.507)左侧，数字化红利仍有空间";

            _additionalInfoText.text =
                $"PGDP: {record.PGDP:F2}  ·  城镇化: {record.URBAN:P0}\n" +
                $"产业结构: {record.INDS:F2}  ·  对外开放: {record.LEOP:P1}\n" +
                $"技术创新(TEIN): {record.TEIN:F3}\n" +
                $"{turningNote}";
        }
    }

    public void Hide()
    {
        if (_panelRoot != null)
            _panelRoot.SetActive(false);
    }
}
