__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void gaussian_filter(
        __read_only image2d_t in_image,
        __write_only image2d_t out_image,
        __constant float * mask,
        __private int radius,
    ) {

    int2 pos = {get_global_id(0), get_global_id(1)};
    float acc = (float3)(0.0f);

    for(int x = -radius; x <= radius; x++) {
        for(int y = -radius; y <= radius; y++) {
            float4 pix = read_imagef(in_image, sampler, pos + (int2)(x,y));
            acc.x += mask[x+radius+(y+radius)*(radius*2+1)]*pix.x;
            acc.y += mask[x+radius+(y+radius)*(radius*2+1)]*pix.y;
            acc.z += mask[x+radius+(y+radius)*(radius*2+1)]*pix.z;
        }
    }

    write_imagef(out_image,pos,(float4)(acc.x,acc.y,acc.z,1.0f));
}