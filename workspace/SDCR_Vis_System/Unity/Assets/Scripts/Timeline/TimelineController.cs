using System.Collections;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Timeline controller for year navigation (2014-2022).
/// Dispatches year changes to StateManager.
/// Supports slider, auto-play, and step buttons.
/// </summary>
public class TimelineController : MonoBehaviour
{
    [Header("UI Components")]
    [SerializeField] private Slider _yearSlider;
    [SerializeField] private Text _yearLabel;
    [SerializeField] private Button _playButton;
    [SerializeField] private Button _prevButton;
    [SerializeField] private Button _nextButton;
    [SerializeField] private Text _playButtonLabel;

    [Header("Auto-Play")]
    [SerializeField] private float _playInterval = 1.2f; // seconds per year

    private StateManager _state;
    private bool _isPlaying = false;
    private Coroutine _playCoroutine;

    void Start()
    {
        _state = StateManager.Instance;

        // Setup slider
        if (_yearSlider != null)
        {
            _yearSlider.minValue = _state.YearMin;
            _yearSlider.maxValue = _state.YearMax;
            _yearSlider.wholeNumbers = true;
            _yearSlider.value = _state.CurrentYear;
            _yearSlider.onValueChanged.AddListener(OnSliderChanged);
        }

        // Buttons
        if (_playButton != null)
            _playButton.onClick.AddListener(TogglePlay);
        if (_prevButton != null)
            _prevButton.onClick.AddListener(() => _state.DecrementYear());
        if (_nextButton != null)
            _nextButton.onClick.AddListener(() => _state.IncrementYear());

        // Listen for state changes to keep UI in sync
        _state.OnYearChanged.AddListener(OnYearStateChanged);

        UpdateLabel(_state.CurrentYear);
    }

    void OnDestroy()
    {
        if (_yearSlider != null) _yearSlider.onValueChanged.RemoveListener(OnSliderChanged);
        if (_state != null) _state.OnYearChanged.RemoveListener(OnYearStateChanged);
    }

    // ── Slider → State ─────────────────────────────────────────

    private void OnSliderChanged(float value)
    {
        int year = Mathf.RoundToInt(value);
        _state.SetYear(year);
        UpdateLabel(year);
    }

    // ── State → UI Sync ────────────────────────────────────────

    private void OnYearStateChanged(int year)
    {
        if (_yearSlider != null && Mathf.RoundToInt(_yearSlider.value) != year)
            _yearSlider.SetValueWithoutNotify(year);
        UpdateLabel(year);
    }

    /// <summary>Called by SDCRPipeline to ensure sync.</summary>
    public void SyncWithState(StateVector s_t)
    {
        if (_yearSlider != null && Mathf.RoundToInt(_yearSlider.value) != s_t.Year)
            _yearSlider.SetValueWithoutNotify(s_t.Year);
        UpdateLabel(s_t.Year);
    }

    // ── Auto-Play ──────────────────────────────────────────────

    public void TogglePlay()
    {
        _isPlaying = !_isPlaying;

        if (_isPlaying)
        {
            if (_playButtonLabel != null) _playButtonLabel.text = "⏸";
            _playCoroutine = StartCoroutine(AutoPlayCoroutine());
        }
        else
        {
            if (_playButtonLabel != null) _playButtonLabel.text = "▶";
            if (_playCoroutine != null) StopCoroutine(_playCoroutine);
        }
    }

    private IEnumerator AutoPlayCoroutine()
    {
        while (_isPlaying)
        {
            yield return new WaitForSeconds(_playInterval);

            if (_state.CurrentYear >= _state.YearMax)
            {
                // Loop back to start
                _state.SetYear(_state.YearMin);
            }
            else
            {
                _state.IncrementYear();
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────

    private void UpdateLabel(int year)
    {
        if (_yearLabel != null)
            _yearLabel.text = $"{year}年";
    }

    public void SetPlayInterval(float seconds)
    {
        _playInterval = Mathf.Max(0.3f, seconds);
    }
}
