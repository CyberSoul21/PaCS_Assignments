__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
    CLK_FILTER_NEAREST;

__kernel void gaussian_filter_tiled(
        __read_only  image2d_t in_image,
        __write_only image2d_t out_image,
        __constant float *mask,
        __private int maskSize,
        __private int width,
        __private int height,
        __local float4 *tile
    )
{
    // coords
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    const int Lx = get_local_size(0);
    const int Ly = get_local_size(1);

    const int side = 2*maskSize + 1;

    // tile dims
    const int tileW = Lx + 2*maskSize;
    const int tileH = Ly + 2*maskSize;

    // tile coords
    int tx = lx + maskSize;
    int ty = ly + maskSize;

    // ============================
    // 1. LOAD CENTER
    // ============================
    if (gx < width && gy < height)
        tile[ty*tileW + tx] =
            read_imagef(in_image, sampler, (int2)(gx,gy));

    // ============================
    // 2. HORIZONTAL + VERTICAL HALOS
    // ============================

    if (lx < maskSize) {
        int ix = clamp(gx - maskSize, 0, width - 1);
        tile[ty*tileW + (tx - maskSize)] =
            read_imagef(in_image, sampler, (int2)(ix, gy));
    }

    if (lx >= Lx - maskSize) {
        int ix = clamp(gx + maskSize, 0, width - 1);
        tile[ty*tileW + (tx + maskSize)] =
            read_imagef(in_image, sampler, (int2)(ix, gy));
    }

    if (ly < maskSize) {
        int iy = clamp(gy - maskSize, 0, height - 1);
        tile[(ty - maskSize)*tileW + tx] =
            read_imagef(in_image, sampler, (int2)(gx, iy));
    }

    if (ly >= Ly - maskSize) {
        int iy = clamp(gy + maskSize, 0, height - 1);
        tile[(ty + maskSize)*tileW + tx] =
            read_imagef(in_image, sampler, (int2)(gx, iy));
    }

    // ============================
    // 3. CORNER HALOS (CORREGIDOS)
    // ============================

    if (lx < maskSize && ly < maskSize) {            // TOP-LEFT
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx - maskSize + a, 0, width - 1);
                int iy = clamp(gy - maskSize + b, 0, height - 1);
                tile[(ty - maskSize + b)*tileW + (tx - maskSize + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    if (lx >= Lx - maskSize && ly < maskSize) {      // TOP-RIGHT
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx + a, 0, width - 1);
                int iy = clamp(gy - maskSize + b, 0, height - 1);
                tile[(ty - maskSize + b)*tileW + (tx + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    if (lx < maskSize && ly >= Ly - maskSize) {      // BOTTOM-LEFT
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx - maskSize + a, 0, width - 1);
                int iy = clamp(gy + b, 0, height - 1);
                tile[(ty + b)*tileW + (tx - maskSize + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    if (lx >= Lx - maskSize && ly >= Ly - maskSize) { // BOTTOM-RIGHT
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx + a, 0, width - 1);
                int iy = clamp(gy + b, 0, height - 1);
                tile[(ty + b)*tileW + (tx + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    // ============================
    // 4. SYNC
    // ============================
    barrier(CLK_LOCAL_MEM_FENCE);

    // ============================
    // 5. CONVOLUTION
    // ============================
    if (gx < width && gy < height) {
        float3 acc = (float3)(0);

        for (int a = -maskSize; a <= maskSize; a++) {
            for (int b = -maskSize; b <= maskSize; b++) {
                float4 pix = tile[(ty + b)*tileW + (tx + a)];
                float w = mask[(a + maskSize)*side + (b + maskSize)];
                acc += w * pix.xyz;
            }
        }

        write_imagef(out_image, (int2)(gx,gy),
                     (float4)(acc.x, acc.y, acc.z, 1.0f));
    }
}
