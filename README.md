---
layout:     post
title:      用Unity实现Shadow Map
subtitle:   
date:       2018-06-27
author:     Richbabe
header-img: img/u3d技术博客背景.jpg
catalog: true
tags:
    - 计算机图形学
    - Unity
---
# 前言
之前用OpenGL实现过一次Shadow Map，今天再用Unity实现了一遍，顺便重温了这个很经典的算法。

# Shadow Map基本原理
Shadow Map示意图如下：
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/%E5%8E%9F%E7%90%86.png?raw=true)
其分为三个步骤：

（1）以光源视角渲染场景，得到深度贴图（DepthMap），并存储为texture

（2）实际相机渲染物体，将物体从世界坐标转换到光源视角下，与深度纹理对比数据获得阴影信息

（3）根据阴影信息渲染场景以及阴影

# 生成深度图纹理的Shader
我们先来看看我们生成深度图纹理的Shader：

```
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

```
其中需要注意的有两点：

（1）把视点空间的z值深度传入片段着色器里需要除以w转换为齐次坐标，为啥要传入片段找色器处理而不在顶点着色器中处理呢？

因为GPU会对片段找色器传入的参数进行插值计算，这样才能更精确的计算出深度。

（2）计算出深度之后，要转换到一张图片里存储起来，如何把一个float存入图片中呢？

float是4个字节的，刚好可以对应RGBA4个分量，把一个float转换成颜色值就可以存为图片了，Unity中提供了一个内置函数：EncodeFloatRGBA帮助我们转换

# 创建Shadow Map相机
## 摄像机类型
shadow Map的相机会根据光源的不同有所差异，平行光使用正交投影比较合适，点光源和聚光灯带有位置信息，适合使用透视投影。

在这里我以平行光和正交投影为例来实现。对于正交投影相机而言，主要关于方向、近平面、远平面、视场大小。

## 创建摄像机
以光源为父节点创建相机：
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/%E6%91%84%E5%83%8F%E6%9C%BA.png?raw=true)

## 实现摄像机获取深度贴图的功能
脚本CaptureDepth如下：

```
public class CaptureDepth : MonoBehaviour {
    public RenderTexture depthTexture;

    private Camera mCam;
    private Shader mSampleDepthShader;


    void Update()
    {
        mCam = GetComponent<Camera>();

        if (mSampleDepthShader == null)
            mSampleDepthShader = Shader.Find("ShadowMap/DepthTextureShader");

        if (mCam != null)
        {
            mCam.backgroundColor = Color.black;
            mCam.clearFlags = CameraClearFlags.Color; ;
            mCam.targetTexture = depthTexture;
            mCam.enabled = false;

            Shader.SetGlobalTexture("_DepthTexture", depthTexture);
            Shader.SetGlobalFloat("_TexturePixelWidth", depthTexture.width);
            Shader.SetGlobalFloat("_TexturePixelHeight", depthTexture.height);


            mCam.RenderWithShader(mSampleDepthShader, "RenderType");

            //mCam.SetReplacementShader (mSampleDepthShader, "RenderType");

            //Debug.Log("_____________________________SampleDepthShader");
        }
    }

    void Start()
    {
        if (mCam != null)
        {

            //	mCam.SetReplacementShader (mSampleDepthShader, "RenderType");		
        }
    }
```
将CaptureDepth挂载在摄像机上，运行时，该脚本就会调用DepthTextureShader将计算的深度贴图保存在贴图depthTexture中。

# 接受阴影的Shader
接受阴影的Shader如下：

```
Shader "Shadow/ShadowMapNormal"
{
	Properties
	{
		//主纹理
		_MainTex ("Texture", 2D) = "white" {}
	}
	SubShader
	{
		Tags { "RenderType"="Opaque" }
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
			v2f vert (appdata v)
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
			fixed4 frag (v2f i) : SV_Target
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
				shadow = currentDepth < depth ? 1.0 : 0.0;

				return (1 - shadow) * col;
			}
			ENDCG
		}
	}
	FallBack "Diffuse"
}

```
这里有几个点需要注意：
* 在变量声明中我们声明了这两个变量：
```
float4x4 _LightSpaceMatrix;//光空间变换矩阵，将每个世界坐标变换到光源所见到的空间
sampler2D _DepthTexture;//深度贴图
```
其中_DepthTexture很容易理解，就是我们上一步计算得到的深度贴图，在CaptureDetph中我们把深度贴图和他的宽和高传了进来：

```
Shader.SetGlobalTexture("_DepthTexture", depthTexture);
Shader.SetGlobalFloat("_TexturePixelWidth", depthTexture.width);
Shader.SetGlobalFloat("_TexturePixelHeight", depthTexture.height);
```
那么LightSpaceMatrix是什么呢？

LightSpaceMatrix为光空间变换矩阵，负责把每个世界坐标变换到光空间坐标。因为我们的深度贴图中保存的深度值是在光空间坐标系的（光源摄像机下的坐标），因此我们需要把要计算是否是阴影的点转换到光空间坐标才能与深度贴图中的深度值作比较。

*在计算对深度贴图的纹理采样坐标时：
```
//将顶点从世界坐标系转换到光空间坐标系
fixed4 lightSpacePos = mul(_LightSpaceMatrix, i.worldPos);
//将光空间片元位置转换为NDC(裁切空间的标准化设备坐标)
lightSpacePos.xyz = lightSpacePos.xyz / lightSpacePos.w;
//将NDC坐标从[-1,1]转换到[0,1]
float3 pos = lightSpacePos * 0.5 + 0.5;
```
在将顶点从世界坐标系转换到光空间坐标系后，还需要将光空间的片元位置转换为NDC(裁切空间的标准化设备坐标)，因为NDC的坐标范围为[-1,1]，这通过除以w值实现，这一步也叫作透视除法。

而又因为来自深度贴图的深度在0到1的范围，所以我们需要将NDC坐标从[-1,1]转换到[0,1]（为了和深度贴图的深度相比较，z分量需要变换到[0,1]；为了作为从深度贴图中采样的坐标，xy分量也需要变换到[0,1]。所以整个lightSpacePos向量都需要变换到[0,1]范围。）
* 在计算和渲染阴影时：

```
//******计算阴影值*****
float shadow = 0.0;//阴影值,1为在阴影中，0为不在
//获得深度贴图的颜色值
fixed4 depthRGBA = tex2D(_DepthTexture, pos.xy);
//获得深度贴图的深度（即离光源最近物体的深度值）
float depth = DecodeFloatRGBA(depthRGBA);
//获取当前像素深度值
float currentDepth = lightSpacePos.z;
shadow = currentDepth < depth ? 1.0 : 0.0;

return (1 - shadow) * col;
```
我们通过Unity的内置函数DecodeFloatRGBA将深度贴图的颜色值转换成深度值。

在Direct3D11中如果当前像素的深度值小于深度贴图的深度值说明该像素在阴影中。而在OpenGL3.0中这是相反的！这是因为d3d和OpenGL的坐标系不同导致z值的判断不同。在OpenGL中如果当前像素的深度值大于深度贴图的深度值说明该像素在阴影中。

将我们接受阴影的shader应用到我们想要接受应用的平面Plane上，接着我们需要新开一个脚本来计算我们的光空间变换矩阵LightSpaceMatrix

# 计算光空间变换矩阵LightSpaceMatrix
计算光空间变换矩阵的脚本如下：

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BaseShadowMap : MonoBehaviour {
    private Camera LightCamera;

    void Awake()
    {
        LightCamera = gameObject.GetComponentInChildren<Camera>();
    }

    void Start()
    {
        if (LightCamera != null)
        {
            Matrix4x4 lightProjecionMatrix = GetLightProjectMatrix(LightCamera);
            Shader.SetGlobalMatrix("_LightSpaceMatrix", lightProjecionMatrix);
            //Debug.Log("_____________________________BaseShadowMap");
        }
        else
        {
            Debug.LogError("Please Add LightCamera!");
        }
    }


    Matrix4x4 GetLightProjectMatrix(Camera lightCam)
    {
        //将裁剪空间的XY坐标系[-1,1]映射到uv坐标[0,1]
        Matrix4x4 posToUV = new Matrix4x4();
        posToUV.SetRow(0, new Vector4(0.5f, 0, 0, 0.5f));
        posToUV.SetRow(1, new Vector4(0, 0.5f, 0, 0.5f));
        posToUV.SetRow(2, new Vector4(0, 0, 1, 0));
        posToUV.SetRow(3, new Vector4(0, 0, 0, 1));

        //世界坐标系 -> 摄像机的摄影坐标系
        Matrix4x4 worldToView = lightCam.worldToCameraMatrix;
        //摄像机的摄影坐标系 -> 摄像机的投影坐标系
        Matrix4x4 projection = GL.GetGPUProjectionMatrix(lightCam.projectionMatrix, false);

        return posToUV * projection * worldToView;
    }
}

```
可以看到我们通过GetLightProjectMatrix函数来获得光空间变换矩阵。我们从MVP(model、view、projection)三个步骤来看看获得光空间变换矩阵。

（1）model矩阵：局部坐标系 -> 世界坐标系

显然，light space 和 view space 所面对的世界是一样的，那么将坐标从 local 坐标系转换到 world 坐标系的 model 矩阵就是相同的。那么在unity中，在shader中直接使用 _ObjectToWorld 矩阵就可以了

(2)view矩阵: 世界坐标系 -> 摄影坐标系

```
//世界坐标系 -> 摄像机的摄影坐标系
        Matrix4x4 worldToView = lightCam.worldToCameraMatrix;
```
通过Camera.worldToCameraMatrix来获得view矩阵

（3）projection矩阵： 摄影坐标系 -> 投影坐标系（也称作裁剪空间坐标系）

```
//摄像机的摄影坐标系 -> 摄像机的投影坐标系
        Matrix4x4 projection = GL.GetGPUProjectionMatrix(lightCam.projectionMatrix, false);
```
通过GL.GetGPUProjectionMatrix来获得projection矩阵

(4)posToUV矩阵

```
//将裁剪空间的XY坐标系[-1,1]映射到uv坐标[0,1]
        Matrix4x4 posToUV = new Matrix4x4();
        posToUV.SetRow(0, new Vector4(0.5f, 0, 0, 0.5f));
        posToUV.SetRow(1, new Vector4(0, 0.5f, 0, 0.5f));
        posToUV.SetRow(2, new Vector4(0, 0, 1, 0));
        posToUV.SetRow(3, new Vector4(0, 0, 0, 1));
```
一个顶点，经过MVP变化之后，其xyz分量的取值范围是[-1, 1]，现在我们需要使用这个变化过的顶点值来找到 shadow depth map中对应的点来比较深度，即要作为UV使用，而UV的取值范围是[0, 1]，所以需要进行一个值域的变换，这就是这个矩阵的作用。

最后，我们将计算得到的光空间转换矩阵传给shader即可：

```
Shader.SetGlobalMatrix("_LightSpaceMatrix", lightProjecionMatrix);
```

# Shadow acne与Peter Panning
## Shadow acne
通过上面的步骤，我们的运行效果如下：
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/shadowAcne.png?raw=true)
我们可以看到地板四边形渲染出很大一块交替黑线。这种阴影贴图的不真实感叫做阴影失真(Shadow Acne)，下图解释了成因：
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/shadowacne1.png?raw=true)
因为阴影贴图受限于解析度，在距离光源比较远的情况下，多个片元可能从深度贴图的同一个值中去采样。图片每个斜坡代表深度贴图一个单独的纹理像素。你可以看到，多个片元从同一个深度值进行采样。
在图中，每一块黑色区域代表非阴影（每一块黑色区域的深度值都比贴图的深度值大，PS：这里用的是OpenGL版本），每一块黄色区域代表阴影（每一块黄色区域的深度值都比贴图的深度值大，PS：这里用的是OpenGL版本），所以就会重现交错的"在阴影里"、"不在阴影里"、"在阴影里"、"不在阴影里"、"在阴影里"、"不在阴影里"......

## shadow bias
除了提升DepthMap的分辨率外，最简单的办法就是给阴影值加一个偏移值，这个方法也称为shadow bias:
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/shadow_mapping_acne_bias.png?raw=true)
使用了偏移量后，所有采样点都获得了比表面深度更大的深度值，这样整个表面就正确地被照亮，没有任何阴影。我们可以这样实现这个偏移：

```
shadow = currentDepth + _Bias < depth ? 1.0 : 0.0;
```
运行效果为：
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/withbias.png?raw=true)

## Peter Panning
使用阴影偏移的一个缺点是你对物体的实际深度应用了平移。偏移有可能足够大，以至于可以看出阴影相对实际物体位置的偏移，你可以从下图看到这个现象（这是一个夸张的偏移值）：
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/shadow_mapping_peter_panning.png?raw=true)
这个阴影失真叫做悬浮(Peter Panning)，因为物体看起来轻轻悬浮在表面之上（译注Peter Pan就是童话彼得潘，而panning有平移、悬浮之意，而且彼得潘是个会飞的男孩…）。
* 在OpenGL中

我们可以使用一个叫技巧解决大部分的Peter panning问题：当渲染深度贴图时候使用正面剔除（front face culling）你也许记得在面剔除教程中OpenGL默认是背面剔除。我们要告诉OpenGL我们要剔除正面。

因为我们只需要深度贴图的深度值，对于实体物体无论我们用它们的正面还是背面都没问题。使用背面深度不会有错误，因为阴影在物体内部有错误我们也看不见。
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/shadow_mapping_culling.png?raw=true)

为了修复peter游移，我们要进行正面剔除，先必须开启GL_CULL_FACE：

```
glCullFace(GL_FRONT);
RenderSceneToDepthMap();
glCullFace(GL_BACK); // 不要忘记设回原先的culling face
```
这十分有效地解决了peter panning的问题，但只针对实体物体，内部不会对外开口。我们的场景中，在立方体上工作的很好，但在地板上无效，因为正面剔除完全移除了地板。地面是一个单独的平面，不会被完全剔除。如果有人打算使用这个技巧解决peter panning必须考虑到只有剔除物体的正面才有意义。

另一个要考虑到的地方是接近阴影的物体仍然会出现不正确的效果。必须考虑到何时使用正面剔除对物体才有意义。
* 在Unity中，我们使用普通的偏移值就能避免peter panning。

# Slope-Scale Depth Bias
更好的纠正做法是基于物体与光照方向的夹角，也就是Slope-Scale Depth Bias，这种方式的提出主要是基于物体表面和光照的夹角越大， Perspective Aliasing的情况越严重，也就越容易出现Shadow Acne，如下图所以。如果采用统一的shadow bias就会出现物体表面一部分区域存再Peter Panning 一部分区域还存在shadow acne。
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/slope-scale.png?raw=true)
更好的办法是根据这个slope进行计算bias，其计算公式如下，miniBais+maxBais∗SlopeScale , 其中SlopeScale可以理解为光线方向与表面法线方向夹角的tan值（也即是水平方向为1的情况下，不同角度对应的矫正量）。

```
float GetShadowBias(float3 lightDir , float3 normal , float maxBias , float baseBias)
{
     float cos_val = saturate(dot(lightDir, normal));
             float sin_val = sqrt(1 - cos_val*cos_val); // sin(acos(L·N))
             float tan_val = sin_val / cos_val;    // tan(acos(L·N))

             float bias = baseBias + clamp(tan_val,0 , maxBias) ;

             return bias ;
}
```
不过Bias数值是个有点感性的数据，[也可以采用其他方式，只要考虑到这个slopescale就行](http://www.sunandblackcat.com/tipFullView.php?l=eng&topicid=35)，比如：

```
// dot product returns cosine between N and L in [-1, 1] range
// then map the value to [0, 1], invert and use as offset
float offsetMod = 1.0 - clamp(dot(N, L), 0, 1)
float offset = minOffset + maxSlopeOffset * offsetMod;

// another method to calculate offset
// gives very large offset for surfaces parallel to light rays
float offsetMod2 = tan(acos(dot(N, L)))
float offset2 = minOffset + clamp(offsetMod2, 0, maxSlopeOffset);
```

# PCF
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/withoutPCF.png?raw=true)
解决完shadow acne后，放大阴影边缘就会看到这种锯齿现象，其主要原因还在于shadow map的分辨率。物体多个点会采集深度纹理同一个点进行阴影计算。这个问题一般可以通过滤波紧进行处理，比如多重采样。

Pencentage close Filtering（PCF）,最简单的一种处理方式，当前点是否为阴影区域需要考虑周围顶点的情况，处理中需要对当前点周围几个像素进行采集，而且这个采集单位越大PCF的效果会越好，当然性能也越差。现在的GPU一般支持2*2的PCF滤波, 也就是Unity设置中的Hard Shadow 。

我这里用的是一个3 * 3的PCF滤波：

```
//应用PCF求阴影值
				float2 texelSize = float2(1.0 / _TexturePixelWidth, 1.0 / _TexturePixelHeight);//一个纹理像素的大小
				for (int x = -1; x <= 1; x++) {
					for (int y = -1; y <= 1; y++) {
						float2 samplePos = pos.xy + float2(x, y) * texelSize;//采样坐标
						fixed4 pcfDepthRGBA = tex2D(_DepthTexture, samplePos);
						float pcfDepth = DecodeFloatRGBA(pcfDepthRGBA);
						shadow += currentDepth + _Bias < pcfDepth ? 1.0 : 0.0;
					}
				}
				shadow /= 9.0;
```
应用了PCF后的效果如下：
![image](https://github.com/Richbabe/Richbabe.github.io/blob/master/img/ShadowMap/withPCF.png?raw=true)
从稍微远一点的距离看去，阴影效果好多了，也不那么生硬了。如果你放大，仍会看到阴影贴图解析度的不真实感，但通常对于大多数应用来说效果已经很好了。

## PCF改进算法
[Shadow Map Antialiasing](http://http.developer.nvidia.com/GPUGems/gpugems_ch11.html) 对PCF做了一些改进，可以更快的执行。[Improvements for shadow mapping in OpenGL and GLSL](http://www.sunandblackcat.com/tipFullView.php?l=eng&topicid=35/) 结合PCF和泊松滤波处理，使用PCF相对少的采样数，就可以获得很好的效果类似的算法还有很多，不一一列举。

# 结语
关于Shadow Map还有许许多多改进的地方，在这里放一些有用的链接，希望以后有空能逐一实现：
* [Common Techniques to Improve Shadow Depth Maps](https://msdn.microsoft.com/en-us/library/windows/desktop/ee416324%28v=vs.85%29.aspx)：微软的一篇好文章，其中理出了很多提升阴影贴图质量的技术。
* [ShadowMap在手游上实现动态阴影](https://zhuanlan.zhihu.com/p/38129922)
* [ShadowMap原理和改进](https://blog.csdn.net/ronintao/article/details/51649664) 和 [Unity基础6 Shadow Map 阴影实现](https://www.cnblogs.com/zsb517/p/6817373.html)

本博客的代码和资源均可在我的github上下载：
* [Unity版本](https://github.com/Richbabe/ShadowMap_Unity)
* [OpenGL版本](https://github.com/Richbabe/ShadowMap_OpenGL)

别忘了点颗Star哟！













