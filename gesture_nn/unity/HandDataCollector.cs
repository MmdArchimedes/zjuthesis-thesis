using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Records hand skeleton data from Rokid AR device for NN training dataset.
///
/// Usage:
///   1. Attach to GameObject with RokidHandProvider
///   2. Set recording parameters
///   3. Press assigned key or trigger via voice command to start/stop recording
///   4. Data is saved to Application.persistentDataPath as JSON
/// </summary>
public class HandDataCollector : MonoBehaviour
{
    [Header("Recording")]
    [SerializeField] private string _gestureLabel = "index_left";
    [SerializeField] private int _gestureClassId = 1;
    [SerializeField] private float _recordDurationSec = 2.5f;
    [SerializeField] private int _participantId = 0;
    [SerializeField] private int _session = 0;

    [Header("Input")]
    [SerializeField] private KeyCode _startRecordKey = KeyCode.R;
    [SerializeField] private KeyCode _stopRecordKey = KeyCode.S;
    [SerializeField] private RokidHandProvider _handProvider;

    [Header("Status")]
    [SerializeField] private bool _isRecording = false;
    [SerializeField] private float _recordTimer = 0f;
    [SerializeField] private int _frameCount = 0;

    private List<FrameData> _recordedFrames = new List<FrameData>();

    [System.Serializable]
    public struct FrameData
    {
        public int frameIndex;
        public float timestamp;
        public float[] joints; // 78 floats = 26 joints × 3 coords
    }

    [System.Serializable]
    public struct RecordingSession
    {
        public string gestureLabel;
        public int gestureClassId;
        public int participantId;
        public int session;
        public int repetition;
        public List<FrameData> frames;
    }

    void Start()
    {
        if (_handProvider == null)
            _handProvider = FindObjectOfType<RokidHandProvider>();
    }

    void Update()
    {
        if (Input.GetKeyDown(_startRecordKey) && !_isRecording)
            StartRecording();

        if (Input.GetKeyDown(_stopRecordKey) && _isRecording)
            StopRecording();

        if (_isRecording)
        {
            RecordFrame();
            _recordTimer += Time.deltaTime;

            if (_recordTimer >= _recordDurationSec)
                StopRecording();
        }
    }

    public void StartRecording()
    {
        _recordedFrames.Clear();
        _isRecording = true;
        _recordTimer = 0f;
        _frameCount = 0;
        Debug.Log($"[DataCollector] Recording started: {_gestureLabel} "
                 + $"(participant={_participantId}, session={_session})");
    }

    public void StopRecording()
    {
        _isRecording = false;
        SaveRecording();
        Debug.Log($"[DataCollector] Recording stopped: {_frameCount} frames saved");
    }

    private void RecordFrame()
    {
        float[] joints = _handProvider.GetJointPositions(RokidHandProvider.HandSide.Right);
        if (joints == null) return;

        _recordedFrames.Add(new FrameData
        {
            frameIndex = _frameCount,
            timestamp = Time.time,
            joints = (float[])joints.Clone(),
        });
        _frameCount++;
    }

    private void SaveRecording()
    {
        var session = new RecordingSession
        {
            gestureLabel = _gestureLabel,
            gestureClassId = _gestureClassId,
            participantId = _participantId,
            session = _session,
            repetition = GetNextRepetition(),
            frames = _recordedFrames,
        };

        string json = JsonUtility.ToJson(session, prettyPrint: true);
        string filename = $"gesture_p{_participantId}_s{_session}_"
                        + $"{_gestureLabel}_{System.DateTime.Now:yyyyMMdd_HHmmss}.json";
        string path = Path.Combine(Application.persistentDataPath, "hand_data", filename);

        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, json);

        Debug.Log($"[DataCollector] Saved to: {path}");
    }

    private int GetNextRepetition()
    {
        string dir = Path.Combine(Application.persistentDataPath, "hand_data");
        if (!Directory.Exists(dir)) return 0;

        int count = 0;
        foreach (string f in Directory.GetFiles(dir, $"*_p{_participantId}_s{_session}_*"))
            count++;
        return count;
    }
}
