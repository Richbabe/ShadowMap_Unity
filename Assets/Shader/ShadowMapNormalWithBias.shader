Shader "Shadow/ShadowMapNormalWithBias" {
	Properties
	{
		//主纹理
		_MainTex("Texture", 2D) = "white" {}
		//阴影偏移值
		_Bias("Bias",Range(-0.005,0.005)) = 0.005
	}
	SubShader
	{
		Tags{ "RenderType" = "Opaque" }
		LOD 100

		Pass
		{
			CGPROGRAM
			//声明顶点着色器和片段着色器
			#pragma vertex vert
			#pragma fragment frag

			//包含头文件
			#include "UnityCG.cginc"

			//变量声明  
			sampler2D _MainTex;
			float4 _MainTex_ST;
			float4x4 _LightSpaceMatrix;//光空间变换矩阵，将每个世界坐标变换到光源所见到的空间
			sampler2D _DepthTexture;//深度贴图
			half _Bias;//阴影偏移值

			//顶点着色器输入结构
			struct appdata
			{
				float4 vertex : POSITION;//顶点位置
				float2 texcoord : TEXCOORD0;//纹理坐标
			};

			//顶点着色器输出结构 
			struct v2f
			{
				float4 pos : SV_POSITION;//视口坐标系下的顶点坐标
				float4 worldPos: TEXCOORD0;//世界坐标系下的顶点坐标
				float2 uv : TEXCOORD1;//经过变换（缩放、偏移）后的纹理坐标
			};

			//定义顶点着色器
			v2f vert(appdata v)
			{
				v2f o;
				o.pos = UnityObjectToClipPos(v.vertex);//将顶点从局部坐标系转换到视口坐标系

				//将顶点从局部坐标系转到世界坐标系
				float4 worldPos = mul(UNITY_MATRIX_M, v.vertex);
				o.worldPos.xyz = worldPos.xyz;
				o.worldPos.w = 1;

				o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);//保存变换后的纹理坐标

				return o;
			}

			//定义片段着色器
			fixed4 frag(v2f i) : SV_Target
			{
				//采样主纹理在对应坐标下的颜色值
				fixed4 col = tex2D(_MainTex, i.uv);

				//将顶点从世界坐标系转换到光空间坐标系
				fixed4 lightSpacePos = mul(_LightSpaceMatrix, i.worldPos);
				//将光空间片元位置转换为NDC(裁切空间的标准化设备坐标)
				lightSpacePos.xyz = lightSpacePos.xyz / lightSpacePos.w;
				//将NDC坐标变换为0到1的范围
				float3 pos = lightSpacePos * 0.5 + 0.5;

				//******计算阴影值*****
				float shadow = 0.0;//阴影值,1为在阴影中，0为不在
				//获得深度贴图的颜色值
				fixed4 depthRGBA = tex2D(_DepthTexture, pos.xy);
				//获得深度贴图的深度（即离光源最近物体的深度值）
				float depth = DecodeFloatRGBA(depthRGBA);
				//获取当前像素深度值
				float currentDepth = lightSpacePos.z;
				shadow = currentDepth + _Bias < depth ? 1.0 : 0.0;

				return (1 - shadow) * col;
			}
			ENDCG
		}
	}
	FallBack "Diffuse"
}
