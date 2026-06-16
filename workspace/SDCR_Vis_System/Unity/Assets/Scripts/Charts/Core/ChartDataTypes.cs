using System;

/// <summary>
/// Renderable chart data container produced by DataFieldResolver.
/// Feeds into chart rendering components.
/// </summary>
[Serializable]
public class ChartData
{
    public string[] Labels;
    public string[] SeriesNames;
    public float[][] SeriesValues;
    public float YMin;
    public float YMax;
}

/// <summary>
/// Chart generation config, typically from LLM-generated JSON.
/// Describes what chart type to render with what data query.
/// </summary>
[Serializable]
public class ChartConfig
{
    public string chartType;
    public ChartDataSource dataSource;
}

/// <summary>
/// Nested data source descriptor within a ChartConfig.
/// </summary>
[Serializable]
public class ChartDataSource
{
    public ChartQuery query;
}

/// <summary>
/// Query parameters that specify which data to pull from DataManager.
/// </summary>
[Serializable]
public class ChartQuery
{
    public string indicator;
    public int[] years;
    public string province;
    public string indicator_x;
    public string indicator_y;
    public string[] indicators;
}
