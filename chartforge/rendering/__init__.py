"""ChartForge rendering adapters — ChartSpec to multiple output formats."""
from .spec_to_vegalite import to_vegalite
from .spec_to_echarts import to_echarts
from .ar_adapter import to_ar_format
