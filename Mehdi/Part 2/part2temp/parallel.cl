// parallel.cl — OpenCL kernel for the graphics pass (BGRA output)

__kernel void shade(
    __global uchar4* out_pixels,           // SIZE = width*height (BGRA)
    __global const float* sat_pos_x,       // SATELLITE_COUNT
    __global const float* sat_pos_y,       // SATELLITE_COUNT
    __global const float* id_r,            // SATELLITE_COUNT
    __global const float* id_g,            // SATELLITE_COUNT
    __global const float* id_b,            // SATELLITE_COUNT
    const int   sat_count,
    const int   width,
    const int   height,
    const float bh_r2,                     // BLACK_HOLE_RADIUS^2
    const float sat_r2,                    // SATELLITE_RADIUS^2
    const int   mouse_x,                   // black hole center X
    const int   mouse_y)                   // black hole center Y
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const float px = (float)x;
    const float py = (float)y;

    // Black hole check (no sqrt)
    float dxBH = px - (float)mouse_x;
    float dyBH = py - (float)mouse_y;
    float d2BH = dxBH * dxBH + dyBH * dyBH;
    if (d2BH < bh_r2) {
        out_pixels[idx] = (uchar4)(0, 0, 0, 0);   // BGRA = black
        return;
    }

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    float weights = 0.0f;
    float shortestD2 = INFINITY;
    float nR = 0.0f, nG = 0.0f, nB = 0.0f;        // nearest id
    int   hit = 0;

    // Single-pass satellite loop
    for (int j = 0; j < sat_count; ++j) {
        float dx = px - sat_pos_x[j];
        float dy = py - sat_pos_y[j];
        float d2 = dx * dx + dy * dy;

        if (d2 < sat_r2) {
            out_pixels[idx] = (uchar4)(255, 255, 255, 0); // BGRA = white
            hit = 1;
            break;
        }

        float inv = 1.0f / d2;      // fast: use -cl-fast-relaxed-math build
        float w = inv * inv;      // 1/(d2*d2)
        weights += w;

        sumR += id_r[j] * w;
        sumG += id_g[j] * w;
        sumB += id_b[j] * w;

        if (d2 < shortestD2) {
            shortestD2 = d2;
            nR = id_r[j]; nG = id_g[j]; nB = id_b[j];
        }
    }

    if (!hit) {
        float invW = 1.0f / weights;
        float r = nR + 3.0f * (sumR * invW);
        float g = nG + 3.0f * (sumG * invW);
        float b = nB + 3.0f * (sumB * invW);

        // Convert to BGRA 0..255
        uchar ur = (uchar)(r * 255.0f);
        uchar ug = (uchar)(g * 255.0f);
        uchar ub = (uchar)(b * 255.0f);
        out_pixels[idx] = (uchar4)(ub, ug, ur, (uchar)0);
    }
}