__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gaussian_filter(
        __global const uchar4* input,
        __global uchar4* output,
        __constant float * mask,
        __private int maskSize,
        __private int width,
        __private int height
    ) {

    int x   = get_global_id(0);
    int y   = get_global_id(1);
    int img = get_global_id(2);

    if (x >= width || y >= height)
        return;

    int img_stride = width * height;
    int base = img * img_stride;
    int idx  = base + y * width + x;

    // Collect neighbor values and multiply with gaussian
    float3 acc = (float3)(0.0f);
    int side = 2 * maskSize + 1;
    for (int dy = -maskSize; dy <= maskSize; ++dy) {
        for (int dx = -maskSize; dx <= maskSize; ++dx) {
            int xx = x + dx;
            int yy = y + dy;

            if (xx < 0 || xx >= width || yy < 0 || yy >= height)
                continue;

            int nidx = base + yy * width + xx;
            float w  = mask[(dy + maskSize) * side + (dx + maskSize)];

            float4 p = convert_float4(input[nidx]);
            acc += w * p.xyz;
        }
    }

    output[idx] = (uchar4)(
        clamp(acc.x, 0.0f, 255.0f),
        clamp(acc.y, 0.0f, 255.0f),
        clamp(acc.z, 0.0f, 255.0f),
        255
    );
}