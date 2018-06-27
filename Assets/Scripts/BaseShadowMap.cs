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
