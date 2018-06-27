using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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
}
