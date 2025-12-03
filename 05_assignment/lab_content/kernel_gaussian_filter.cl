__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gaussian_filter(
        __read_only image2d_t in_image,
        __write_only image2d_t out_image,
        __constant float * mask,
        __private int maskSize,
        __private int width,
        __private int height
    ) {

    const int2 pos = {get_global_id(0), get_global_id(1)};
    if (pos.x >= width || pos.y >= height)
        return;
    
    // Collect neighbor values and multiply with gaussian
    float sum = 0.0f;
    // Calculate the mask size based on sigma (larger sigma, larger mask)
    float3 acc = (float3)(0.0f);
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            if ((pos.x + (int)a) < 0 || (pos.x + (int)a) >= width || (pos.y + (int)b) < 0 || (pos.y + (int)b) >= height)
                continue;

            float4 pix = read_imagef(in_image, sampler, pos + (int2)(a,b));
            acc.x += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]*pix.x;
            acc.y += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]*pix.y;
            acc.z += mask[a+maskSize+(b+maskSize)*(maskSize*2+1)]*pix.z;
        }
    }

    write_imagef(out_image,pos,(float4)(acc.x,acc.y,acc.z,1.0f));
}