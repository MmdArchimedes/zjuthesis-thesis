using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Resolves chart config data references to actual numeric data.
/// Bridges ChartConfig → DataManager → ChartData ready for rendering.
/// </summary>
public static class DataFieldResolver
{
    /// <summary>
    /// Maps logical field names (from LLM-generated JSON) to IndicatorRecord fields.
    /// The JSON uses human-readable names; this maps them to actual C# fields.
    /// </summary>
    public static float GetFieldValue(DataManager.IndicatorRecord record, string fieldName)
    {
        switch (fieldName)
        {
            case "ES":   return record.ES;
            case "DEL":  return record.DEL;
            case "PGDP": return record.PGDP;
            case "URBAN":return record.URBAN;
            case "INDS": return record.INDS;
            case "LEOP": return record.LEOP;
            case "FDE":  return record.FDE;
            case "DEN":  return record.DEN;
            case "TEIN": return record.TEIN;
            // Aliases for common alternate names
            case "GDP":         return record.PGDP;  // GDP ≈ 人均GDP
            case "INNOVATION":  return record.TEIN;  // 技术创新
            case "INDUSTRY":    return record.INDS;  // 产业结构
            case "URBANIZATION":return record.URBAN;  // 城镇化
            case "OPENNESS":    return record.FDE;   // 对外开放
            default:
                Debug.LogWarning($"[DataFieldResolver] Unknown field: {fieldName}");
                return 0f;
        }
    }

    /// <summary>
    /// Resolve chart config to renderable ChartData.
    /// </summary>
    public static ChartData Resolve(ChartConfig config, DataManager dataManager)
    {
        if (config == null || dataManager == null)
            return new ChartData();

        var query = config.dataSource?.query;
        if (query == null) return new ChartData();

        switch (config.chartType)
        {
            case "bar":   return ResolveBar(query, dataManager);
            case "line":
            case "area":  return ResolveTimeSeries(query, dataManager);
            case "pie":   return ResolvePie(query, dataManager);
            case "scatter":return ResolveScatter(query, dataManager);
            case "heatmap":return ResolveHeatmap(query, dataManager);
            case "chinaMap":
            case "provinceMap": return ResolveMap(query, dataManager);
            case "radar": return ResolveRadar(query, dataManager);
            case "dashboard": return ResolveDashboard(query, dataManager);
            default:      return ResolveBar(query, dataManager);
        }
    }

    // ── Individual resolvers ───────────────────────────────────────

    private static ChartData ResolveBar(ChartQuery query, DataManager dm)
    {
        var data = new ChartData();
        string indicator = query.indicator ?? "ES";
        int year = query.years != null && query.years.Length > 0 ? query.years[0] : 2022;

        var records = dm.GetYearRecords(year);
        int n = records.Count;

        data.Labels = new string[n];
        data.SeriesValues = new float[1][];
        data.SeriesValues[0] = new float[n];
        data.SeriesNames = new string[] { indicator };

        for (int i = 0; i < n; i++)
        {
            data.Labels[i] = records[i].ProvinceName;
            data.SeriesValues[0][i] = GetFieldValue(records[i], indicator);
        }

        data.YMin = 0f;
        data.YMax = 1f;
        foreach (float v in data.SeriesValues[0])
            if (v > data.YMax) data.YMax = v;

        return data;
    }

    private static ChartData ResolveTimeSeries(ChartQuery query, DataManager dm)
    {
        var data = new ChartData();
        string indicator = query.indicator ?? "ES";
        int startYear = query.years != null && query.years.Length >= 2 ? query.years[0] : 2014;
        int endYear = query.years != null && query.years.Length >= 2 ? query.years[1] : 2022;

        // Resolve province name → ID
        string provinceName = query.province ?? "广东";
        int provinceId = ResolveProvinceId(dm, provinceName);

        int n = endYear - startYear + 1;
        data.Labels = new string[n];
        data.SeriesValues = new float[1][];
        data.SeriesValues[0] = new float[n];
        data.SeriesNames = new string[] { indicator };

        data.YMin = float.MaxValue;
        data.YMax = float.MinValue;

        for (int i = 0; i < n; i++)
        {
            int y = startYear + i;
            data.Labels[i] = y.ToString();
            float val = dm.GetValue(provinceId, y, indicator);
            data.SeriesValues[0][i] = val;
            if (val < data.YMin) data.YMin = val;
            if (val > data.YMax) data.YMax = val;
        }

        if (n == 0 || data.YMin > data.YMax)
        { data.YMin = 0f; data.YMax = 1f; }

        return data;
    }

    private static ChartData ResolvePie(ChartQuery query, DataManager dm)
    {
        var data = new ChartData();
        string provinceName = query.province ?? "广东";
        int year = query.years != null && query.years.Length > 0 ? query.years[0] : 2022;
        int provinceId = ResolveProvinceId(dm, provinceName);
        var record = dm.GetRecord(provinceId, year);
        if (record == null) return data;

        // Use available fields as pie sectors
        string[] fields = { "ES", "DEL", "PGDP", "TEIN", "INDS", "URBAN" };
        string[] labels = { "能源结构", "数字经济", "人均GDP", "技术创新", "产业结构", "城镇化" };

        data.Labels = labels;
        data.SeriesValues = new float[1][];
        data.SeriesValues[0] = new float[fields.Length];
        data.SeriesNames = new string[] { provinceName };

        for (int i = 0; i < fields.Length; i++)
            data.SeriesValues[0][i] = GetFieldValue(record, fields[i]);

        data.YMin = 0f; data.YMax = 1f;
        return data;
    }

    private static ChartData ResolveScatter(ChartQuery query, DataManager dm)
    {
        var data = new ChartData();
        string indX = query.indicator_x ?? "DEL";
        string indY = query.indicator_y ?? "ES";
        int year = query.years != null && query.years.Length > 0 ? query.years[0] : 2022;

        var records = dm.GetYearRecords(year);
        int n = records.Count;
        data.Labels = new string[n];
        data.SeriesValues = new float[2][];
        data.SeriesValues[0] = new float[n];
        data.SeriesValues[1] = new float[n];
        data.SeriesNames = new string[] { indX, indY };

        for (int i = 0; i < n; i++)
        {
            data.Labels[i] = records[i].ProvinceName;
            data.SeriesValues[0][i] = GetFieldValue(records[i], indX);
            data.SeriesValues[1][i] = GetFieldValue(records[i], indY);
        }

        return data;
    }

    private static ChartData ResolveHeatmap(ChartQuery query, DataManager dm)
    {
        var data = new ChartData();
        string[] indicators = query.indicators ?? new string[] { "ES", "DEL", "TEIN", "PGDP" };
        int year = query.years != null && query.years.Length > 0 ? query.years[0] : 2022;

        var records = dm.GetYearRecords(year);
        int nProv = records.Count;
        int nInd = indicators.Length;

        data.Labels = indicators;
        data.SeriesValues = new float[nProv][];
        data.SeriesNames = new string[nProv];

        for (int i = 0; i < nProv; i++)
        {
            data.SeriesNames[i] = records[i].ProvinceName;
            data.SeriesValues[i] = new float[nInd];
            for (int j = 0; j < nInd; j++)
                data.SeriesValues[i][j] = GetFieldValue(records[i], indicators[j]);
        }

        return data;
    }

    private static ChartData ResolveMap(ChartQuery query, DataManager dm)
    {
        return ResolveBar(query, dm);
    }

    private static ChartData ResolveRadar(ChartQuery query, DataManager dm)
    {
        var data = new ChartData();
        string provinceName = query.province ?? "广东";
        int year = query.years != null && query.years.Length > 0 ? query.years[0] : 2022;
        int provinceId = ResolveProvinceId(dm, provinceName);
        string[] indicators = query.indicators ?? new string[] { "ES", "DEL", "TEIN", "PGDP", "INDS", "URBAN" };

        var record = dm.GetRecord(provinceId, year);
        if (record == null) return data;

        data.Labels = indicators;
        data.SeriesValues = new float[1][];
        data.SeriesValues[0] = new float[indicators.Length];
        data.SeriesNames = new string[] { provinceName };

        for (int i = 0; i < indicators.Length; i++)
            data.SeriesValues[0][i] = GetFieldValue(record, indicators[i]);

        data.YMin = 0f; data.YMax = 1f;
        return data;
    }

    private static ChartData ResolveDashboard(ChartQuery query, DataManager dm)
    {
        return ResolveRadar(query, dm);  // Same logic: multi-indicator for one province
    }

    // ── Helpers ────────────────────────────────────────────────────

    private static int ResolveProvinceId(DataManager dm, string provinceName)
    {
        foreach (var kv in dm.ProvinceNames)
        {
            if (kv.Value == provinceName || kv.Value.StartsWith(provinceName))
                return kv.Key;
        }
        // Fallback: first province
        foreach (var kv in dm.ProvinceNames)
            return kv.Key;
        return 0;
    }
}
