cbuffer ConstantBuffer
{
    float4x4 final;
    float4x4 rotation;    // the rotation matrix
    float4 lightvec;      // the light's vector
    float4 lightcol;      // the light's color
    float4 ambientcol;    // the ambient light's color
}

Texture2D Texture;
SamplerState ss;

struct VOut
{
    float4 color : COLOR;
    float2 texcoord : TEXCOORD;    // texture coordinates
    float4 position : SV_POSITION;
};

VOut VShader(float4 position : POSITION, float4 normal : NORMAL, float2 texcoord : TEXCOORD)
{
    VOut output;

	float2 testtex = Texture.Sample(ss, texcoord);
    output.position = mul(final, position);

    // set the ambient light
    output.color = ambientcol;

    // calculate the diffuse light and add it to the ambient light
    float4 norm = normalize(mul(rotation, normal));
    float diffusebrightness = saturate(dot(norm, lightvec));
    output.color += lightcol * diffusebrightness;

    output.texcoord = texcoord;    // set the texture coordinates, unmodified

    return output;
}

float4 PShaderLidar(float4 color : COLOR, float2 texcoord : TEXCOORD) : SV_TARGET
{
	if( (texcoord.x * 1000000) % 2.0 == 0.0 || (texcoord.y * 1000000) % 2.0 == 0.0 )
	{
		// Simulate scan lines
		float4 black = {0,0,0,0};
		return black;
	}
	float2 horizontal_flip = {1.0f - texcoord.x, texcoord.y};
    float4 input = Texture.Sample(ss, horizontal_flip);
	float4 z = pow(input.r * 4.0f, 0.3333f); // Increase long distance clarity
	float red = z;
	float green = 1.0f - abs(0.5f - z) * 2.0f;
	float blue = (1.0f - z);
	float4 ret = {red, green, blue, 0};
	return ret;
}

float4 PShaderHeatMap(float4 color : COLOR, float2 texcoord : TEXCOORD) : SV_TARGET
{
	float2 horizontal_flip = {1.0f - texcoord.x, texcoord.y};
    float4 input = Texture.Sample(ss, horizontal_flip);
	float4 z = pow(input.r * 4.0f, 0.3333f);
	float red = z;
	float green = 1.0f - abs(0.5f - z) * 2.0f;
	float blue = (1.0f - z);
	float4 ret = {red, green, blue, 0};
	return ret;
}

float4 PShaderPassThrough(float4 color : COLOR, float2 texcoord : TEXCOORD) : SV_TARGET
{
	float2 horizontal_flip = {1.0f - texcoord.x, texcoord.y};
    float4 input = Texture.Sample(ss, horizontal_flip);	
	float4 ret = {input.r, input.g, input.b, input.a};
	return ret;
}

float4 PShaderAqua(float4 color : COLOR, float2 texcoord : TEXCOORD) : SV_TARGET
{
	float green = 0.5f;
	float blue = 1.0f;
	float red = 0.2f;
	float4 ret = {red, green, blue, 0};
	return ret;
}

float4 PShaderIndex(float4 color : COLOR, float2 texcoord : TEXCOORD) : SV_TARGET
{
	int pixdex = 1200 * texcoord.y + texcoord.x;
	float red = 64 & pixdex;
	float green = 64 & (pixdex >> 7);
	float blue = 64 & (pixdex >> 14);
	float4 ret = {texcoord.x, texcoord.y, 0.42f, 0};
	return ret;
}

float4 PShaderExperimentProject1(float4 color : COLOR, float2 texcoord : TEXCOORD) : SV_TARGET
{
	float2 horizontal_flip = {1.0f - texcoord.x, texcoord.y};
    float4 input = Texture.Sample(ss, horizontal_flip);
	float4 z = pow(input.r, 0.3333f);
	float blue = z;
	float green = 1.0f - abs(0.5f - z) * 2.0f;
	float red = (1.0f - z);
	float4 ret = {red, green, blue, 0};
	return ret;
}