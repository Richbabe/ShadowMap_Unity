Shader "ShadowMap/DepthTextureShader" {
	SubShader {
		Tags { "RenderType"="Opaque" }

		Pass{
			CGPROGRAM

			//声明顶点着色器和片段着色器
			#pragma vertex vert
			#pragma fragment frag

			//包含头文件
			#include "UnityCG.cginc"

			//定义顶点着色器输入
			struct vsInput {
				float4 vertex : POSITION;//顶点局部坐标
			};

			//定义顶点着色器输出
			struct vsOutput {
				float4 vertex : SV_POSITION;//顶点视口坐标
				float2 depth : TEXCOORD0;//将深度保存成深度贴图纹理
			};

			//定义顶点着色器
			vsOutput vert(vsInput v) {
				vsOutput o;
				o.vertex = UnityObjectToClipPos(v.vertex);//将顶点从局部坐标系转换到视口坐标系
				o.depth = o.vertex.zw;//保存深度值
				return o;
			}

			//定义片段着色器
			fixed4 frag(vsOutput i) : SV_Target{
				float depth = i.depth.x / i.depth.y;//把z值深度从视口坐标转换为齐次坐标（除以w值）
				/*
					调用EncodeFloatRGBA将float类型的深度转换成RGBA4个分量的颜色值
				*/
				fixed4 col = EncodeFloatRGBA(depth);
				return col;
			}

			ENDCG
		}		
	}
	FallBack "Diffuse"
}
