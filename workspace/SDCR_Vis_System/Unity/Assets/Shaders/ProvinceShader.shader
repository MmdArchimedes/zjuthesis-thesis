Shader "SDCR/ProvinceShader"
{
    // Custom shader for province visualization with state-driven color/emission.
    // Supports: base color, highlight emission, dimming via alpha.
    Properties
    {
        _Color ("Base Color", Color) = (0.5, 0.5, 0.5, 1)
        _MainTex ("Texture", 2D) = "white" {}
        _EmissionColor ("Emission Color", Color) = (0, 0, 0, 0)
        _EmissionStrength ("Emission Strength", Range(0, 2)) = 0.4
        _Glossiness ("Smoothness", Range(0, 1)) = 0.3
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex;
        fixed4 _Color;
        fixed4 _EmissionColor;
        half _EmissionStrength;
        half _Glossiness;

        struct Input
        {
            float2 uv_MainTex;
        };

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb;
            o.Alpha = c.a;
            o.Smoothness = _Glossiness;
            o.Emission = _EmissionColor.rgb * _EmissionStrength;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
