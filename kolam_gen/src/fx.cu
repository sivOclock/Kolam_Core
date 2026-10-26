// fx.rs on the GPU: one thread per pixel of a batch of B pictures, s x s each. The parameter layout
// (P_*, NP, MAX_DEBRIS) is prepended by gpu.rs from look::cuda_defines(). Keep every formula equal to
// fx.rs: same hash noise, same passes, same order.

typedef unsigned int u32;
typedef unsigned char u8;

__device__ __forceinline__ u32 mix(u32 h) {
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; return h ^ (h >> 16);
}
__device__ __forceinline__ u32 hash4(u32 seed, u32 stream, int x, int y) {
    return mix(seed ^ mix(stream * 0x9e3779b9u ^ mix(((u32)x) * 0x85ebca6bu ^ mix(((u32)y) + 0xc2b2ae35u))));
}
__device__ __forceinline__ float u01(u32 seed, u32 stream, int x, int y) {
    return (float)(hash4(seed, stream, x, y) >> 8) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float gauss(u32 seed, u32 stream, int x, int y) {
    float u = fmaxf(u01(seed, stream, x, y), 1e-7f);
    float v = u01(seed, stream ^ 0x55555555u, x, y);
    return sqrtf(-2.0f * logf(u)) * cosf(6.2831855f * v);
}
__device__ __forceinline__ void cubic(float t, float* w) {
    float t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] = 1.5f * t3 - 2.5f * t2 + 1.0f;
    w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    w[3] = 0.5f * t3 - 0.5f * t2;
}
__device__ float smooth(u32 seed, u32 stream, float cy, float cx, int x, int y, int s) {
    int cyi = (int)cy, cxi = (int)cx;
    float u = ((float)x + 0.5f) * cx / (float)s - 0.5f;
    float v = ((float)y + 0.5f) * cy / (float)s - 0.5f;
    float i0 = floorf(u), j0 = floorf(v);
    float wu[4], wv[4];
    cubic(u - i0, wu); cubic(v - j0, wv);
    float acc = 0.0f;
    for (int b = 0; b < 4; b++) {
        int j = min(max((int)j0 + b - 1, 0), cyi - 1);
        float row = 0.0f;
        for (int a = 0; a < 4; a++) {
            int i = min(max((int)i0 + a - 1, 0), cxi - 1);
            row += wu[a] * (u01(seed, stream, i, j) * 2.0f - 1.0f);
        }
        acc += wv[b] * row;
    }
    return fminf(fmaxf(acc, -1.0f), 1.0f);
}

#define S_CHALK 1u
#define S_CHALK_DOT 2u
#define S_POWDER 3u
#define S_MULTI 10u
#define S_TEX 20u
#define S_FLOOR 30u
#define S_GRAIN 40u
#define S_SPECK 41u
#define S_TILE 42u
#define S_NOISE 50u

__device__ __forceinline__ u32 seed_of(const float* p) { return __float_as_uint(p[P_SEED]); }

// ------------------------------------------------------------------ per pixel
extern "C" __global__ void warp_chalk(const u8* mask, const float* P, float* out, int s, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * s;
    if (idx >= B * n) return;
    int b = idx / n, k = idx % n, x = k % s, y = k / s;
    const float* p = P + b * NP;
    const u8* m = mask + (size_t)b * n;
    float fx = x + 0.5f, fy = y + 0.5f;
    float w = p[P_HI6] * fx + p[P_HI7] * fy + p[P_HI8];
    float sx = (p[P_HI0] * fx + p[P_HI1] * fy + p[P_HI2]) / w - 0.5f;
    float sy = (p[P_HI3] * fx + p[P_HI4] * fy + p[P_HI5]) / w - 0.5f;
    float x0 = floorf(sx), y0 = floorf(sy), ax = sx - x0, ay = sy - y0;
    int i = (int)x0, j = (int)y0;
#define AT(ii, jj) (((ii) < 0 || (jj) < 0 || (ii) >= s || (jj) >= s) ? 0.0f : m[(jj) * s + (ii)] / 255.0f)
    float a = (1.0f - ay) * ((1.0f - ax) * AT(i, j) + ax * AT(i + 1, j)) + ay * ((1.0f - ax) * AT(i, j + 1) + ax * AT(i + 1, j + 1));
#undef AT
    if (p[P_CHALK] > 0.0f) {
        u32 seed = seed_of(p);
        float ink = a > 0.12f ? 1.0f : 0.0f;
        float keep = smooth(seed, S_CHALK, p[P_CHALK_CELLS], p[P_CHALK_CELLS], x, y, s);
        float fade = fminf(fmaxf((p[P_CHALK_T] - keep) * 2.5f, 0.0f), 1.0f) * ink;
        if (u01(seed, S_CHALK_DOT, x, y) < 0.05f) fade = fmaxf(fade, 0.4f * ink);
        a *= 1.0f - fade;
    }
    out[idx] = a;
}

__device__ void floor_px(const float* p, int x, int y, int s, float* c) {
    u32 seed = seed_of(p);
    float fl[3] = {p[P_FL_R], p[P_FL_G], p[P_FL_B]};
#define N(k, cy, cx) smooth(seed, S_FLOOR + (k), (cy), (cx), x, y, s)
#define GRAIN(amt) ((amt) * gauss(seed, S_GRAIN, x, y))
    float v = 0.0f;
    switch ((int)p[P_SURF]) {
    case 0: v = 18.0f * N(0, 6.f, 6.f) + 10.0f * N(1, 40.f, 40.f) + GRAIN(9.0f) - (u01(seed, S_SPECK, x, y) < 0.004f ? 40.0f : 0.0f); break;
    case 1: v = 14.0f * N(0, 5.f, 5.f) + GRAIN(4.0f) + 35.0f * fmaxf(N(1, 3.f, 3.f), 0.0f); break;
    case 2: {
        float a = p[P_SA];
        float streak = p[P_SB] > 0.5f ? N(0, a, 4.f) : N(0, 4.f, a);
        v = 16.0f * streak + 10.0f * N(1, 12.f, 12.f) + GRAIN(6.0f);
    } break;
    case 3: v = 10.0f * N(0, 8.f, 8.f) + GRAIN(14.0f); break;
    case 4: {
        int size = (int)p[P_SA], ox = (int)p[P_SB], oy = (int)p[P_SC];
        int line = max(size / 40, 2);
        if ((y + oy) % size < line || (x + ox) % size < line) {
            for (int i = 0; i < 3; i++) c[i] = fl[i] * 0.55f;
            return;
        }
        int ti = (y + oy) / size, tj = (x + ox) / size;
        float vein = fabsf(sinf(N(1, 5.f, 5.f) * 9.0f + N(2, 17.f, 17.f) * 3.0f));
        float t = 6.0f * N(0, 20.f, 20.f) + GRAIN(3.0f) - (vein < 0.08f ? p[P_SD] : 0.0f);
        for (int i = 0; i < 3; i++) c[i] = fl[i] + u01(seed, S_TILE + i, tj, ti) * 24.0f - 12.0f + t;
        return;
    }
    case 5: {
        float speck = u01(seed, S_SPECK, x, y) < 0.05f ? 30.0f * (u01(seed, S_SPECK + 1, x, y) * 2.0f - 1.0f) : 0.0f;
        v = GRAIN(7.0f) + speck;
    } break;
    case 6: {
        bool along_y = p[P_SA] > 0.5f;
        float along = (float)(along_y ? y : x);
        float t = along / (float)s * p[P_SB] + 0.6f * N(0, 3.f, 8.f);
        float m = powf(0.5f + 0.5f * sinf(t * 6.2831855f), 3.0f);
        float fine = along_y ? N(1, 90.f, 4.f) : N(1, 4.f, 90.f);
        float w = 6.0f * fine + GRAIN(4.0f);
        for (int i = 0; i < 3; i++) c[i] = fl[i] * (1.0f - m) + p[P_FL2_R + i] * m + w;
        return;
    }
    case 7: v = 5.0f * N(0, 10.f, 10.f) + 4.0f * N(1, 80.f, 80.f) + GRAIN(3.0f); break;
    case 8: v = 12.0f * fmaxf(N(0, 7.f, 7.f), 0.0f) + 4.0f * N(1, 30.f, 30.f) + GRAIN(3.0f); break;
    case 9: v = 8.0f * N(0, 10.f, 10.f) + GRAIN(16.0f); break;
    default:
        c[0] = p[P_BG_R]; c[1] = p[P_BG_G]; c[2] = p[P_BG_B];
        return;
    }
#undef N
#undef GRAIN
    for (int i = 0; i < 3; i++) c[i] = fl[i] + v;
}

extern "C" __global__ void colour(const float* A, const float* P, const float* D, float* rgb, int s, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * s;
    if (idx >= B * n) return;
    int b = idx / n, k = idx % n, x = k % s, y = k / s;
    const float* p = P + b * NP;
    const float* debris = D + b * MAX_DEBRIS * 7;
    u32 seed = seed_of(p);
    float a = A[idx];
    float fl[3];
    floor_px(p, x, y, s, fl);
    float ink[3] = {p[P_INK_R], p[P_INK_G], p[P_INK_B]};
    int multi = (int)p[P_MULTI];
    if (multi > 0) {
        float sum = 0.0f, acc[3] = {0.0f, 0.0f, 0.0f};
        for (int q = 0; q < multi; q++) {
            float cl = p[P_MCELLS0 + q];
            float w = expf(6.0f * smooth(seed, S_MULTI + q, cl, cl, x, y, s));
            sum += w;
            for (int i = 0; i < 3; i++) acc[i] += w * p[P_MC0 + 3 * q + i];
        }
        for (int i = 0; i < 3; i++) ink[i] = acc[i] / sum;
    }
    if (p[P_POWDER] > 0.0f) a *= 1.0f - p[P_POWDER] * u01(seed, S_POWDER, x, y);
    float c[3];
    for (int i = 0; i < 3; i++) c[i] = fl[i] * (1.0f - a) + ink[i] * a;
    float fx = x + 0.5f, fy = y + 0.5f;
    int nd = (int)p[P_DEBRIS];
    for (int q = 0; q < nd; q++) {
        const float* d = debris + 7 * q;
        float dx = (fx - d[0]) / d[2], dy = (fy - d[1]) / d[3];
        if (dx * dx + dy * dy <= 1.0f) { c[0] = d[4]; c[1] = d[5]; c[2] = d[6]; }
    }
    float f = 1.0f;
    if (p[P_TEX_CELLS] > 0.0f) f *= 1.0f + 0.08f * smooth(seed, S_TEX, p[P_TEX_CELLS], p[P_TEX_CELLS], x, y, s);
    float u = fx / (float)s, v = fy / (float)s;
    if (p[P_LIGHT] > 0.0f) {
        float t = p[P_LIGHT_T];
        f *= 1.0f + p[P_LIGHT] * (cosf(t) * (u - 0.5f) + sinf(t) * (v - 0.5f)) * 2.0f;
    }
    if (p[P_SHADOW] > 0.0f) {
        float t = p[P_SHADOW_T];
        float d = cosf(t) * (fx - p[P_SHADOW_X]) + sinf(t) * (fy - p[P_SHADOW_Y]);
        f *= 1.0f - p[P_SHADOW] / (1.0f + expf(-d / p[P_SHADOW_SOFT]));
    }
    for (int i = 0; i < 3; i++) c[i] *= f;
    if (p[P_GLARE] >= 0.0f) {
        float cx = p[P_GLARE_X], cy = p[P_GLARE_Y], t = p[P_GLARE_T], g;
        int kind = (int)p[P_GLARE];
        if (kind == 0) {
            g = expf(-((u - cx) * (u - cx) + (v - cy) * (v - cy)) / (2.0f * p[P_GLARE_A] * p[P_GLARE_A]));
        } else if (kind == 1) {
            float d = cosf(t) * (u - cx) + sinf(t) * (v - cy);
            float along = -sinf(t) * (u - cx) + cosf(t) * (v - cy);
            g = expf(-d * d / (2.0f * p[P_GLARE_A] * p[P_GLARE_A])) * expf(-along * along / (2.0f * p[P_GLARE_B] * p[P_GLARE_B]));
        } else {
            float h = fminf(fmaxf(0.5f + cosf(t) * (u - 0.5f) + sinf(t) * (v - 0.5f), 0.0f), 1.0f);
            g = h * h;
        }
        float w = g * p[P_GLARE_S];
        const float white[3] = {255.0f, 252.0f, 240.0f};
        for (int i = 0; i < 3; i++) c[i] += (white[i] - c[i]) * w;
    }
    if (p[P_VIG] > 0.0f) {
        float r2 = ((u - p[P_VIG_X]) * (u - p[P_VIG_X]) + (v - p[P_VIG_Y]) * (v - p[P_VIG_Y])) * 2.0f;
        for (int i = 0; i < 3; i++) c[i] *= 1.0f - p[P_VIG] * r2;
    }
    if (p[P_COLOUR] > 0.0f) {
        float m = p[P_COL_MEAN];
        for (int i = 0; i < 3; i++) {
            float w = ((c[i] - m) * p[P_COL_C] + m * p[P_COL_B]) * p[P_COL_G0 + i];
            c[i] = 255.0f * powf(fminf(fmaxf(w, 0.0f), 255.0f) / 255.0f, p[P_COL_GAMMA]);
        }
    }
    if (p[P_GRAY] > 0.0f) {
        float m = (c[0] + c[1] + c[2]) / 3.0f;
        c[0] = c[1] = c[2] = m;
    }
    for (int i = 0; i < 3; i++) rgb[((size_t)b * 3 + i) * n + k] = c[i];
}

// ------------------------------------------------------------------ neighbourhood passes
// Gaussian blur along one axis of C planes per picture, sigma = P[pidx] (0: copy). Edges clamped.
extern "C" __global__ void blur(const float* src, float* dst, const float* P, int pidx, int C, int s, int B, int vertical) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * s;
    if (idx >= B * C * n) return;
    int b = idx / (C * n), k = idx % n, x = k % s, y = k / s;
    const float* plane = src + (size_t)(idx / n) * n;
    float sigma = P[b * NP + pidx];
    if (sigma <= 0.0f) { dst[idx] = src[idx]; return; }
    int r = min(max((int)ceilf(3.0f * sigma), 1), 10);
    float acc = 0.0f, tot = 0.0f;
    for (int q = -r; q <= r; q++) {
        float w = expf(-(float)(q * q) / (2.0f * sigma * sigma));
        int xx = vertical ? x : min(max(x + q, 0), s - 1);
        int yy = vertical ? min(max(y + q, 0), s - 1) : y;
        acc += w * plane[yy * s + xx];
        tot += w;
    }
    dst[idx] = acc / tot;
}

// Max filter along one axis (a thicker line), radius P[P_THICK_R] (0: copy). One plane per picture.
extern "C" __global__ void max_filter(const float* src, float* dst, const float* P, int s, int B, int vertical) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * s;
    if (idx >= B * n) return;
    int b = idx / n, k = idx % n, x = k % s, y = k / s;
    int r = (int)P[b * NP + P_THICK_R];
    const float* plane = src + (size_t)b * n;
    float m = 0.0f;
    for (int q = -r; q <= r; q++) {
        int xx = vertical ? x : x + q, yy = vertical ? y + q : y;
        if (xx >= 0 && yy >= 0 && xx < s && yy < s) m = fmaxf(m, plane[yy * s + xx]);
    }
    dst[idx] = m;
}

// Camera shake (P_MOTION_N copies along P_MOTION_T, wrapping), then sensor noise and the clamp.
extern "C" __global__ void motion_noise(const float* src, float* dst, const float* P, int s, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * s;
    if (idx >= B * n) return;
    int b = idx / n, k = idx % n, x = k % s, y = k / s;
    const float* p = P + b * NP;
    const float* img = src + (size_t)b * 3 * n;
    float c[3];
    int nm = (int)p[P_MOTION_N];
    if (nm > 0) {
        float t = p[P_MOTION_T];
        c[0] = c[1] = c[2] = 0.0f;
        for (int q = 0; q < nm; q++) {
            float sh = -(float)nm / 2.0f + (nm > 1 ? (float)q * (float)nm / (float)(nm - 1) : 0.0f);
            int dy = (int)roundf(sh * sinf(t)), dx = (int)roundf(sh * cosf(t));
            int sx = ((x - dx) % s + s) % s, sy = ((y - dy) % s + s) % s;
            for (int i = 0; i < 3; i++) c[i] += img[i * n + sy * s + sx];
        }
        for (int i = 0; i < 3; i++) c[i] /= (float)nm;
    } else {
        for (int i = 0; i < 3; i++) c[i] = img[i * n + k];
    }
    if (p[P_NOISE] > 0.0f) {
        u32 seed = seed_of(p);
        bool mono = p[P_NOISE_MONO] > 0.5f;
        for (int i = 0; i < 3; i++) c[i] += p[P_NOISE] * gauss(seed, S_NOISE + (mono ? 0u : (u32)i), x, y);
    }
    for (int i = 0; i < 3; i++) dst[((size_t)b * 3 + i) * n + k] = fminf(fmaxf(c[i], 0.0f), 255.0f);
}

// Low resolution (box-shrink to m x m, bilinear back up), then bytes: RGB interleaved.
__device__ float small_px(const float* plane, int i, int j, int m, int s) {
    int x0 = i * s / m, x1 = max((i + 1) * s / m, x0 + 1);
    int y0 = j * s / m, y1 = max((j + 1) * s / m, y0 + 1);
    float t = 0.0f;
    for (int yy = y0; yy < y1; yy++)
        for (int xx = x0; xx < x1; xx++) t += plane[yy * s + xx];
    return t / (float)((x1 - x0) * (y1 - y0));
}

extern "C" __global__ void lowres_bytes(const float* src, u8* out, const float* P, int s, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * s;
    if (idx >= B * n) return;
    int b = idx / n, k = idx % n, x = k % s, y = k / s;
    const float* img = src + (size_t)b * 3 * n;
    float lr = P[b * NP + P_LOWRES];
    for (int ch = 0; ch < 3; ch++) {
        float c;
        if (lr > 0.0f) {
            int m = max((int)((float)s * lr), 16);
            float u = fminf(fmaxf(((float)x + 0.5f) * m / (float)s - 0.5f, 0.0f), (float)(m - 1));
            float v = fminf(fmaxf(((float)y + 0.5f) * m / (float)s - 0.5f, 0.0f), (float)(m - 1));
            int i = (int)floorf(u), j = (int)floorf(v);
            float a = u - i, bb = v - j;
            int i1 = min(i + 1, m - 1), j1 = min(j + 1, m - 1);
            const float* pl = img + ch * n;
            c = (1.0f - bb) * ((1.0f - a) * small_px(pl, i, j, m, s) + a * small_px(pl, i1, j, m, s))
              + bb * ((1.0f - a) * small_px(pl, i, j1, m, s) + a * small_px(pl, i1, j1, m, s));
        } else {
            c = img[ch * n + k];
        }
        out[((size_t)b * n + k) * 3 + ch] = (u8)fminf(fmaxf(roundf(c), 0.0f), 255.0f);
    }
}
