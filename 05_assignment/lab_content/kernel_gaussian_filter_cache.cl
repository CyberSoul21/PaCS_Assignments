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
        __local float4 *tile     // tile cooperativo
    )
{
    // Coordinates
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    const int Lx = get_local_size(0);
    const int Ly = get_local_size(1);

    const int side = 2*maskSize + 1;

    // Tile dimensions (WG region + halo)
    const int tileW = Lx + 2*maskSize;
    const int tileH = Ly + 2*maskSize;

    // Position inside tile
    int tx = lx + maskSize;
    int ty = ly + maskSize;

    // ---------------------------
    // 1. LOAD CENTER PIXEL
    // ---------------------------
    if (gx < width && gy < height) {
        tile[ty*tileW + tx] =
            read_imagef(in_image, sampler, (int2)(gx,gy));
    }

    // ---------------------------
    // 2. LOAD HALOS (LEFT / RIGHT / TOP / BOTTOM)
    // ---------------------------

    // LEFT halo
    if (lx < maskSize) {
        int ix = clamp(gx - maskSize, 0, width - 1);
        tile[ty*tileW + (tx - maskSize)] =
            read_imagef(in_image, sampler, (int2)(ix, gy));
    }

    // RIGHT halo
    if (lx >= Lx - maskSize) {
        int ix = clamp(gx + maskSize, 0, width - 1);
        tile[ty*tileW + (tx + maskSize)] =
            read_imagef(in_image, sampler, (int2)(ix, gy));
    }

    // TOP halo
    if (ly < maskSize) {
        int iy = clamp(gy - maskSize, 0, height - 1);
        tile[(ty - maskSize)*tileW + tx] =
            read_imagef(in_image, sampler, (int2)(gx, iy));
    }

    // BOTTOM halo
    if (ly >= Ly - maskSize) {
        int iy = clamp(gy + maskSize, 0, height - 1);
        tile[(ty + maskSize)*tileW + tx] =
            read_imagef(in_image, sampler, (int2)(gx, iy));
    }

    // ---------------------------
    // 3. LOAD 4 CORNER HALOS
    //    (each WG loads up to maskSize×maskSize pixels per corner)
    // ---------------------------

    // TOP-LEFT corner
    if (lx < maskSize && ly < maskSize) {
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx - maskSize + a, 0, width - 1);
                int iy = clamp(gy - maskSize + b, 0, height - 1);
                tile[(ty - maskSize + b)*tileW + (tx - maskSize + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    // TOP-RIGHT corner
    if (lx >= Lx - maskSize && ly < maskSize) {
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx + a, 0, width - 1);
                int iy = clamp(gy - maskSize + b, 0, height - 1);
                tile[(ty - maskSize + b)*tileW + (tx + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    // BOTTOM-LEFT corner
    if (lx < maskSize && ly >= Ly - maskSize) {
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx - maskSize + a, 0, width - 1);
                int iy = clamp(gy + b, 0, height - 1);
                tile[(ty + b)*tileW + (tx - maskSize + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    // BOTTOM-RIGHT corner
    if (lx >= Lx - maskSize && ly >= Ly - maskSize) {
        for (int a = 0; a < maskSize; a++) {
            for (int b = 0; b < maskSize; b++) {
                int ix = clamp(gx + a, 0, width - 1);
                int iy = clamp(gy + b, 0, height - 1);
                tile[(ty + b)*tileW + (tx + a)] =
                    read_imagef(in_image, sampler, (int2)(ix, iy));
            }
        }
    }

    // ------------------------------------------------
    // 4. SYNC BEFORE USING TILE MEMORY
    // ------------------------------------------------
    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------
    // 5. CONVOLUTION USING LOCAL MEMORY ONLY
    // ------------------------------------------------
    if (gx < width && gy < height) {

        float3 acc = (float3)(0);

        for (int a = -maskSize; a <= maskSize; a++) {
            for (int b = -maskSize; b <= maskSize; b++) {
                float4 pix = tile[(ty + b)*tileW + (tx + a)];
                float w = mask[(a + maskSize)*side + (b + maskSize)];

                acc.x += w * pix.x;
                acc.y += w * pix.y;
                acc.z += w * pix.z;
            }
        }

        write_imagef(out_image, (int2)(gx,gy),
                     (float4)(acc.x, acc.y, acc.z, 1.0f));
    }
}
