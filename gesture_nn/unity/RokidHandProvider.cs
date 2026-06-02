using UnityEngine;
using System;

/// <summary>
/// Abstraction layer for Rokid UXR hand tracking.
///
/// In production, this wraps Rokid's native SDK API to read per-frame
/// joint positions. During editor testing, it falls back to the
/// HandSimulator for synthetic input.
///
/// Joint ordering follows Rokid UXR convention:
///   0=Wrist, 1=Palm, 2-5=Thumb(CMC/MCP/IP/Tip),
///   6-9=Index(MCP/PIP/DIP/Tip), 10-13=Middle, 14-17=Ring, 18-21=Pinky,
///   22-25=Additional tracking points
/// </summary>
public class RokidHandProvider : MonoBehaviour
{
    [Header("Data Source")]
    [SerializeField] private bool _useSimulator = true;
    [SerializeField] private HandSimulator _simulator;

    public enum HandSide { Right, Left }

    /// <summary>
    /// Get joint positions for specified hand.
    /// Returns float[78] = 26 joints × 3 coords (x,y,z) in meters, world-space or tracking-space.
    /// Returns null if hand is not currently tracked.
    /// </summary>
    public float[] GetJointPositions(HandSide side)
    {
        if (_useSimulator)
        {
            if (_simulator == null)
                _simulator = FindObjectOfType<HandSimulator>();
            return _simulator?.GetJointPositions(side);
        }

        // Production path: read from Rokid UXR SDK
        return ReadFromRokidSDK(side);
    }

    /// <summary>
    /// Check if hand is currently tracked with sufficient confidence.
    /// </summary>
    public bool IsHandTracked(HandSide side)
    {
        if (_useSimulator)
            return _simulator != null && _simulator.IsTracked(side);

        return ReadTrackingConfidence(side) > 0.5f;
    }

    /// <summary>
    /// Get raw tracking confidence [0, 1].
    /// </summary>
    public float ReadTrackingConfidence(HandSide side)
    {
        if (_useSimulator)
            return _simulator?.GetConfidence(side) ?? 0f;

        // Production: read from Rokid SDK tracking state
        return 1.0f; // placeholder
    }

    // ── Production integration point ─────────────────────────────
    //
    // Replace this method with actual Rokid UXR SDK calls.
    // References:
    //   Rokid UXR HandTracking API → GetJointLocations()
    //   Transform joint poses from camera-space to world-space
    //   using headpose * camera_extrinsics
    //
    private float[] ReadFromRokidSDK(HandSide side)
    {
        // TODO: Integrate with actual Rokid UXR SDK
        //
        // Example pseudocode:
        //   var tracker = RokidXR.GetHandTracker();
        //   var joints = tracker.GetJointLocations(side == HandSide.Right ?
        //                         HandType.Right : HandType.Left);
        //
        //   float[] result = new float[78];
        //   for (int j = 0; j < 26; j++) {
        //       var pos = joints[j].position;  // Vector3 in tracking space
        //       result[j*3+0] = pos.x;
        //       result[j*3+1] = pos.y;
        //       result[j*3+2] = pos.z;
        //   }
        //   return result;

        Debug.LogWarning("[RokidHandProvider] Real SDK integration not configured — using simulator");
        return null;
    }
}
