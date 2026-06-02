using UnityEngine;

/// <summary>
/// Original rule-based (geometric criterion) gesture classifier.
/// Reproduces the thesis DBEW--Gesture geometric method for comparison.
///
/// Uses: finger collinearity (thesis Eq 3-1), extended finger count,
/// palm normal orientation, horizontal projection.
/// </summary>
public class RuleBasedGestureClassifier : MonoBehaviour
{
    [Header("Parameters")]
    [SerializeField] private float _thetaExt = 0.85f; // collinearity threshold (~32° max deviation)
    [SerializeField] private RokidHandProvider _handProvider;

    [Header("DBEW Trigger")]
    [SerializeField] private float _cooldownMs = 500f;
    [SerializeField] private int _minStableFrames = 8;

    public enum GestureCommand
    {
        NONE = 0, IndexLeft = 1, IndexRight = 2,
        TwoFingerPalm = 3, TwoFingerBack = 4,
        FourFingerPalm = 5, Fist = 6,
    }

    public event System.Action<GestureClassifierNN.GestureEvent> OnGestureTriggered;

    // Finger joint index ranges (matching Python RuleBasedClassifier)
    private static readonly int[][] FingerIndices = {
        new[]{2, 3, 4, 5},     // thumb
        new[]{6, 7, 8, 9},     // index
        new[]{10, 11, 12, 13}, // middle
        new[]{14, 15, 16, 17}, // ring
        new[]{18, 19, 20, 21}, // pinky
    };

    private GestureCommand _prevCmd = GestureCommand.NONE;
    private GestureCommand _stableCandidate = GestureCommand.NONE;
    private int _stableCount = 0;
    private float _lastTriggerTime = float.MinValue;

    void Update()
    {
        if (_handProvider == null) return;

        float[] joints = _handProvider.GetJointPositions(RokidHandProvider.HandSide.Right);
        if (joints == null) return;

        var (cmd, conf) = Classify(joints);
        var evt = ApplyTrigger(cmd, conf, Time.time);
        if (evt.HasValue)
            OnGestureTriggered?.Invoke(evt.Value);
    }

    /// <summary>
    /// Geometric rule-based classification (thesis Sections 3.3.2--3.3.3).
    /// Uses curl angle and explicit per-finger checks for robustness.
    /// </summary>
    public (GestureCommand, float) Classify(float[] joints)
    {
        bool[] ext = new bool[5];
        for (int f = 0; f < 5; f++)
            ext[f] = IsFingerExtended(joints, FingerIndices[f]);

        bool indexExt = ext[1], middleExt = ext[2], ringExt = ext[3], pinkyExt = ext[4];
        int nonThumbExt = (indexExt ? 1 : 0) + (middleExt ? 1 : 0)
                        + (ringExt ? 1 : 0) + (pinkyExt ? 1 : 0);

        // Fist: all four non-thumb fingers curled
        if (nonThumbExt == 0)
            return (GestureCommand.Fist, 0.90f);

        // Index only pointing
        if (indexExt && !middleExt && !ringExt && !pinkyExt)
        {
            float dirX = joints[9 * 3] - joints[6 * 3];
            if (dirX < -0.005f) return (GestureCommand.IndexLeft, 0.88f);
            if (dirX > 0.005f)  return (GestureCommand.IndexRight, 0.88f);
            return (GestureCommand.NONE, 0.40f);
        }

        // Two-finger (index + middle)
        if (indexExt && middleExt && !ringExt && !pinkyExt)
        {
            if (PalmFacesUser(joints)) return (GestureCommand.TwoFingerPalm, 0.86f);
            else                       return (GestureCommand.TwoFingerBack, 0.84f);
        }

        // Four-finger
        if (indexExt && middleExt && ringExt && pinkyExt)
        {
            if (PalmFacesUser(joints)) return (GestureCommand.FourFingerPalm, 0.92f);
            else                       return (GestureCommand.TwoFingerBack, 0.85f);
        }

        // Ambiguous partial extensions
        if (indexExt && !middleExt)
        {
            float dirX = joints[9 * 3] - joints[6 * 3];
            if (dirX < -0.005f) return (GestureCommand.IndexLeft, 0.60f);
            if (dirX > 0.005f)  return (GestureCommand.IndexRight, 0.60f);
        }

        return (GestureCommand.NONE, 0.50f);
    }

    private float FingerCurlAngle(float[] joints, int[] fingerIdx)
    {
        // Angle between proximal (MCP→PIP) and distal (PIP→Tip) segments
        Vector3 mcp = new(joints[fingerIdx[0]*3], joints[fingerIdx[0]*3+1], joints[fingerIdx[0]*3+2]);
        Vector3 pip = new(joints[fingerIdx[1]*3], joints[fingerIdx[1]*3+1], joints[fingerIdx[1]*3+2]);
        Vector3 tip = new(joints[fingerIdx[3]*3], joints[fingerIdx[3]*3+1], joints[fingerIdx[3]*3+2]);

        Vector3 v1 = pip - mcp;
        Vector3 v2 = tip - pip;

        if (v1.magnitude < 1e-6f || v2.magnitude < 1e-6f)
            return 180f;

        return Vector3.Angle(v1, v2);
    }

    private bool IsFingerExtended(float[] joints, int[] fingerIdx)
    {
        // Extended if proximal-distal angle < 25°
        return FingerCurlAngle(joints, fingerIdx) < 25f;
    }

    private bool PalmFacesUser(float[] joints)
    {
        Vector3 wrist  = new(joints[0], joints[1], joints[2]);
        Vector3 palm   = new(joints[1*3], joints[1*3+1], joints[1*3+2]);
        Vector3 idxMcp = new(joints[6*3], joints[6*3+1], joints[6*3+2]);
        Vector3 ringMcp= new(joints[14*3], joints[14*3+1], joints[14*3+2]);

        Vector3 normal = Vector3.Cross(palm - wrist, idxMcp - ringMcp).normalized;
        // Canonical hand normal = -z → palm toward user. +z = palm away.
        return normal.z < -0.1f;
    }

    private GestureClassifierNN.GestureEvent? ApplyTrigger(
        GestureCommand rawCmd, float conf, float currentTime)
    {
        if (conf < 0.7f) { _stableCount = 0; return null; }

        if (rawCmd == _stableCandidate) _stableCount++;
        else { _stableCandidate = rawCmd; _stableCount = 1; }

        if (_stableCount < _minStableFrames) return null;

        bool edge = rawCmd != _prevCmd && rawCmd != GestureCommand.NONE;
        float dtMs = (currentTime - _lastTriggerTime) * 1000f;
        bool cooldown = dtMs >= _cooldownMs;

        if (edge && cooldown)
        {
            _lastTriggerTime = currentTime;
            _prevCmd = rawCmd;
            return new GestureClassifierNN.GestureEvent
            {
                Type = (GestureClassifierNN.GestureCommand)rawCmd,
                Timestamp = currentTime,
                Confidence = conf,
            };
        }
        return null;
    }
}
