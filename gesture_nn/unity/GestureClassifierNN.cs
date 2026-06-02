using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

/// <summary>
/// Neural network gesture classifier integrated with the DBEW--Gesture pipeline.
///
/// Replaces the rule-based geometric classifier (collinearity, extended-finger count)
/// with a lightweight CNN+Attention model while preserving the edge-trigger +
/// cooldown-window + stable-frame triple-gate mechanism.
///
/// Model: gesture_classifier.onnx
/// Input:  [1, 32, 26, 3] — 32-frame window of 26 joints in (x,y,z) wrist-relative
/// Output: [1, 7] — logits for {NONE, index_left, index_right, two_palm, two_back, four_palm, fist}
/// </summary>
public class GestureClassifierNN : MonoBehaviour
{
    [Header("Model")]
    [SerializeField] private NNModel _modelAsset;
    [SerializeField] private bool _useWorkerThread = true;

    [Header("Input Buffer")]
    [SerializeField] private int _windowSize = 32;
    [SerializeField] private RokidHandProvider _handProvider; // your existing hand data source

    [Header("DBEW Trigger (matching thesis Section 3.3.4)")]
    [SerializeField] private float _cooldownMs = 500f;       // tau
    [SerializeField] private int _minStableFrames = 8;       // k_min
    [SerializeField] private float _confidenceThreshold = 0.85f; // theta_g

    [Header("Debug")]
    [SerializeField] private bool _debugLog = false;

    // Gesture enum matching thesis Table tab:ges_map
    public enum GestureCommand
    {
        NONE = 0,
        IndexLeft = 1,       // year - 1
        IndexRight = 2,      // year + 1
        TwoFingerPalm = 3,   // switch to DEL
        TwoFingerBack = 4,   // switch to ES
        FourFingerPalm = 5,  // main scene
        Fist = 6,            // reset
    }

    // Events (consumed by TSTQ--Fusion multimodal fusion layer)
    public event Action<GestureEvent> OnGestureTriggered;

    public struct GestureEvent
    {
        public GestureCommand Type;
        public float Timestamp;
        public float Confidence;
        public string Payload;
    }

    // ── State ────────────────────────────────────────────────────
    private IWorker _worker;
    private Tensor _inputTensor;
    private float[] _inputBuffer;        // flattened: [32*26*3]
    private Queue<float[]> _skeletonBuffer; // ring buffer of recent frames

    private GestureCommand _prevCommand = GestureCommand.NONE;
    private GestureCommand _stableCandidate = GestureCommand.NONE;
    private int _stableCount = 0;
    private float _lastTriggerTime = float.MinValue;
    private bool _modelReady = false;


    // ── Unity Lifecycle ──────────────────────────────────────────

    void Start()
    {
        _skeletonBuffer = new Queue<float[]>();

        if (_modelAsset != null)
        {
            var model = ModelLoader.Load(_modelAsset);
            _worker = _useWorkerThread
                ? WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model)
                : WorkerFactory.CreateWorker(WorkerFactory.Type.Compute, model);

            _inputBuffer = new float[_windowSize * 26 * 3];
            _modelReady = true;

            if (_debugLog) Debug.Log($"[GestureNN] Model loaded. Worker type: {_worker.GetType()}");
        }
        else
        {
            Debug.LogWarning("[GestureNN] No model asset assigned — falling back to rule-based classifier");
        }

        if (_handProvider == null)
            _handProvider = FindObjectOfType<RokidHandProvider>();
    }

    void Update()
    {
        if (!_modelReady || _handProvider == null) return;

        // 1) Read current frame skeleton from Rokid UXR
        float[] frameJoints = _handProvider.GetJointPositions(HandSide.Right);
        if (frameJoints == null || frameJoints.Length != 26 * 3)
        {
            if (_debugLog) Debug.LogWarning("[GestureNN] Invalid skeleton data this frame");
            return;
        }

        // 2) Normalize to wrist-relative
        NormalizeToWrist(frameJoints);

        // 3) Push to ring buffer
        _skeletonBuffer.Enqueue(frameJoints);
        while (_skeletonBuffer.Count > _windowSize)
            _skeletonBuffer.Dequeue();

        // 4) If buffer full, run inference
        if (_skeletonBuffer.Count < _windowSize)
            return;

        FlattenBuffer();

        // 5) Run neural network
        GestureCommand rawCommand;
        float confidence;
        InferGesture(out rawCommand, out confidence);

        // 6) DBEW Trigger Pipeline (replaces rule classificaton but keeps gating)
        var triggeredEvent = ApplyTriggerPipeline(rawCommand, confidence, Time.time);

        // 7) Emit event if triggered
        if (triggeredEvent.HasValue)
        {
            OnGestureTriggered?.Invoke(triggeredEvent.Value);

            if (_debugLog)
                Debug.Log($"[GestureNN] Triggered: {triggeredEvent.Value.Type} "
                         + $"(conf={triggeredEvent.Value.Confidence:F2})");
        }
    }

    void OnDestroy()
    {
        _worker?.Dispose();
        _inputTensor?.Dispose();
    }


    // ── Normalization ────────────────────────────────────────────

    private void NormalizeToWrist(float[] joints)
    {
        // Joint 0 is wrist — subtract from all joints for wrist-relative coords
        float wx = joints[0], wy = joints[1], wz = joints[2];

        for (int j = 0; j < 26; j++)
        {
            joints[j * 3 + 0] -= wx;
            joints[j * 3 + 1] -= wy;
            joints[j * 3 + 2] -= wz;
        }

        // Unit scale normalization
        float sumSq = 0f;
        for (int i = 0; i < joints.Length; i++)
            sumSq += joints[i] * joints[i];
        float scale = Mathf.Sqrt(sumSq / joints.Length) * 10f;

        if (scale > 1e-6f)
        {
            float invScale = 1f / scale;
            for (int i = 0; i < joints.Length; i++)
                joints[i] *= invScale;
        }
    }

    private void FlattenBuffer()
    {
        int idx = 0;
        foreach (float[] frame in _skeletonBuffer)
        {
            Array.Copy(frame, 0, _inputBuffer, idx, frame.Length);
            idx += frame.Length;
        }
    }


    // ── Neural Network Inference ──────────────────────────────────

    private void InferGesture(out GestureCommand command, out float confidence)
    {
        command = GestureCommand.NONE;
        confidence = 0f;

        // Create input tensor [1, 32, 26, 3]
        _inputTensor?.Dispose();
        _inputTensor = new Tensor(1, _windowSize, 26, 3, _inputBuffer);

        // Execute
        _worker.Execute(_inputTensor);
        Tensor output = _worker.PeekOutput();

        // Find argmax and confidence (softmax probability)
        float maxLogit = float.MinValue;
        int maxIdx = 0;
        float sumExp = 0f;

        for (int i = 0; i < 7; i++)
        {
            float logit = output[0, 0, 0, i];
            float expVal = Mathf.Exp(logit);
            sumExp += expVal;

            if (logit > maxLogit)
            {
                maxLogit = logit;
                maxIdx = i;
            }
        }

        command = (GestureCommand)maxIdx;
        confidence = Mathf.Exp(maxLogit) / sumExp;
        output.Dispose();
    }


    // ── DBEW Trigger Pipeline (matching thesis Algorithm 2) ──────

    private GestureEvent? ApplyTriggerPipeline(GestureCommand rawCommand,
                                                float confidence, float currentTime)
    {
        // Gate 0: confidence threshold
        if (confidence < _confidenceThreshold)
        {
            _stableCount = 0;
            _stableCandidate = GestureCommand.NONE;
            return null;
        }

        // Stability counter (minimum stable frames k_min)
        if (rawCommand == _stableCandidate)
        {
            _stableCount++;
        }
        else
        {
            _stableCandidate = rawCommand;
            _stableCount = 1;
        }

        bool stable = _stableCount >= _minStableFrames;
        if (!stable) return null;

        // Edge trigger (only on transition)
        bool edge = rawCommand != _prevCommand && rawCommand != GestureCommand.NONE;

        // Cooldown window (tau)
        float dtMs = (currentTime - _lastTriggerTime) * 1000f;
        bool cooldownPassed = dtMs >= _cooldownMs;

        // Final trigger decision: u = e · ψ · φ  (thesis Eq 3-5)
        if (edge && cooldownPassed && stable)
        {
            _lastTriggerTime = currentTime;
            _prevCommand = rawCommand;

            return new GestureEvent
            {
                Type = rawCommand,
                Timestamp = currentTime,
                Confidence = confidence,
                Payload = MapCommandToPayload(rawCommand),
            };
        }

        return null;
    }

    private string MapCommandToPayload(GestureCommand cmd)
    {
        return cmd switch
        {
            GestureCommand.IndexLeft => "year:-1",
            GestureCommand.IndexRight => "year:+1",
            GestureCommand.TwoFingerPalm => "indicator:DEL",
            GestureCommand.TwoFingerBack => "indicator:ES",
            GestureCommand.FourFingerPalm => "scene:main",
            GestureCommand.Fist => "reset",
            _ => "",
        };
    }


    // ── Public API ────────────────────────────────────────────────

    /// <summary>
    /// Get raw gesture distribution for debugging/visualization.
    /// </summary>
    public float[] GetGestureProbabilities()
    {
        if (!_modelReady || _skeletonBuffer.Count < _windowSize)
            return new float[7];

        FlattenBuffer();
        _inputTensor?.Dispose();
        _inputTensor = new Tensor(1, _windowSize, 26, 3, _inputBuffer);
        _worker.Execute(_inputTensor);

        Tensor output = _worker.PeekOutput();
        float[] probs = new float[7];
        float sumExp = 0f;

        for (int i = 0; i < 7; i++)
            sumExp += Mathf.Exp(output[0, 0, 0, i]);
        for (int i = 0; i < 7; i++)
            probs[i] = Mathf.Exp(output[0, 0, 0, i]) / sumExp;

        output.Dispose();
        return probs;
    }
}
