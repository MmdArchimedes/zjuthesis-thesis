using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

/// <summary>
/// HTTP client for ChartForge API — bridges Unity AR with Python AI pipeline.
/// Sends NL queries and receives AR-formatted chart specs for rendering.
/// </summary>
[Serializable]
public class ChartForgeClient : MonoBehaviour
{
    [Header("API Configuration")]
    public string apiBaseUrl = "http://192.168.1.100:8001";
    public float requestTimeout = 30f;

    [Header("Debug")]
    public bool verboseLogging = true;

    // Event: fired when chart data is received
    public event Action<string, ChartSpecData> OnChartGenerated;
    public event Action<string> OnError;

    /// <summary>
    /// Request a chart from natural language query.
    /// </summary>
    public void RequestChart(string nlQuery, string outputFormat = "ar")
    {
        StartCoroutine(RequestChartCoroutine(nlQuery, outputFormat));
    }

    private IEnumerator RequestChartCoroutine(string nlQuery, string outputFormat)
    {
        var requestData = new FullPipelineRequest
        {
            query = nlQuery,
            output_format = outputFormat,
            available_fields = new string[] {
                "province", "region", "year",
                "DEL", "ES", "PGDP", "URBAN",
                "INDS", "LEOP", "FDE", "TEIN"
            },
            context = "Provincial digital economy and energy structure analysis"
        };

        string jsonBody = JsonUtility.ToJson(requestData);
        var request = new UnityWebRequest($"{apiBaseUrl}/full-pipeline", "POST");

        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonBody);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.timeout = (int)requestTimeout;

        if (verboseLogging)
            Debug.Log($"[ChartForge] Requesting chart: \"{nlQuery}\"");

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            string error = $"ChartForge API error: {request.error}";
            Debug.LogError(error);
            OnError?.Invoke(error);
            yield break;
        }

        string responseText = request.downloadHandler.text;

        try
        {
            var response = JsonUtility.FromJson<FullPipelineResponse>(responseText);
            var chartSpec = JsonUtility.FromJson<ChartSpecData>(
                JsonUtility.ToJson(response.chart)
            );

            if (verboseLogging)
                Debug.Log($"[ChartForge] Generated {response.chart_type} chart " +
                         $"(SVAS: {response.svas_score:F3}, " +
                         $"Latency: {response.total_latency_ms:F0}ms)");

            OnChartGenerated?.Invoke(nlQuery, chartSpec);
        }
        catch (Exception e)
        {
            string error = $"Failed to parse ChartForge response: {e.Message}";
            Debug.LogError(error);
            OnError?.Invoke(error);
        }
    }

    /// <summary>
    /// Evaluate SVAS score for an existing chart.
    /// </summary>
    public void EvaluateChart(string chartJson, string cifJson, Action<SVASScoreData> callback)
    {
        StartCoroutine(EvaluateChartCoroutine(chartJson, cifJson, callback));
    }

    private IEnumerator EvaluateChartCoroutine(
        string chartJson, string cifJson, Action<SVASScoreData> callback)
    {
        var requestData = new SVASRequest
        {
            chart_spec = chartJson,
            cif = cifJson,
        };

        string jsonBody = JsonUtility.ToJson(requestData);
        var request = new UnityWebRequest($"{apiBaseUrl}/svas-score", "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonBody);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            var response = JsonUtility.FromJson<SVASScoreData>(request.downloadHandler.text);
            callback?.Invoke(response);
        }
        else
        {
            Debug.LogError($"SVAS evaluation failed: {request.error}");
            callback?.Invoke(null);
        }
    }

    private void OnDestroy()
    {
        StopAllCoroutines();
    }
}

// ── Serializable Data Models ─────────────────────────────────────

[Serializable]
public class FullPipelineRequest
{
    public string query;
    public string output_format;
    public string[] available_fields;
    public string context;
}

[Serializable]
public class FullPipelineResponse
{
    public ChartData chart;
    public string chart_type;
    public float svas_score;
    public float total_latency_ms;
    public StageTimes stage_times;
}

[Serializable]
public class ChartData
{
    public string chartType;
    public string chartId;
    public LayoutData layout;
    public DataSection data;
    public EncodingsData encodings;
    public StyleData style;
    public InteractionData[] interactions;
    public AnnotationData[] annotations;
}

[Serializable]
public class LayoutData
{
    public PositionData position;
    public SizeData size;
}

[Serializable]
public class PositionData
{
    public float x, y, z;
}

[Serializable]
public class SizeData
{
    public float width, height;
}

[Serializable]
public class DataSection
{
    public Dictionary<string, string> bindings;
    public string source;
}

[Serializable]
public class EncodingsData
{
    public EncodingField x, y, color, size;
}

[Serializable]
public class EncodingField
{
    public string field;
    public bool enabled;
}

[Serializable]
public class StyleData
{
    public string colorScheme;
    public float fontSize;
    public float titleFontSize;
    public string backgroundColor;
    public bool showGrid;
    public float gridAlpha;
}

[Serializable]
public class InteractionData
{
    public string evt;  // "event" is reserved in C#
    public string handler;
    public bool enabled;
}

[Serializable]
public class AnnotationData
{
    public string text;
    public string position;
}

[Serializable]
public class StageTimes
{
    public float cif_parsing;
    public float coarse_generation;
    public float semantic_verification;
    public float visual_refinement;
    public float interaction_injection;
}

[Serializable]
public class SVASRequest
{
    public string chart_spec;
    public string cif;
}

[Serializable]
public class SVASScoreData
{
    public float phi_sem;
    public float phi_vis;
    public float phi_int;
    public float svas;
    public bool passed_filter;
}

/// <summary>
/// Deserialized chart spec for Unity rendering.
/// </summary>
[Serializable]
public class ChartSpecData
{
    public string chartType;
    public string chartId;
    public LayoutData layout;
    public DataSection data;
    public EncodingsData encodings;
    public StyleData style;
    public InteractionData[] interactions;
    public AnnotationData[] annotations;
}
