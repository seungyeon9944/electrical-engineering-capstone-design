/*
 * fast_preprocess.c
 * 번호판 이미지 고속 전처리 C 구현
 * Python ctypes를 통해 호출됨
 *
 * 빌드: gcc -O2 -shared -fPIC -o fast_preprocess.so fast_preprocess.c
 */

#include "fast_preprocess.h"
#include <string.h>
#include <stdlib.h>

/* ── 가우시안 블러 (3×3, σ≈0.85) ──────────────────────────── */
static const int GAUSS_K[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};
static const int GAUSS_SUM = 16;

void fast_gaussian_blur(
    const uint8_t *src, uint8_t *dst, int width, int height
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int acc = 0;
            for (int ky = -1; ky <= 1; ky++) {
                int ny = y + ky;
                if (ny < 0) ny = 0;
                if (ny >= height) ny = height - 1;
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    if (nx < 0) nx = 0;
                    if (nx >= width) nx = width - 1;
                    acc += src[ny * width + nx] * GAUSS_K[ky + 1][kx + 1];
                }
            }
            dst[y * width + x] = (uint8_t)(acc / GAUSS_SUM);
        }
    }
}

/* ── Otsu 이진화 임계값 계산 ──────────────────────────────── */
uint8_t otsu_threshold(const uint8_t *gray, int width, int height) {
    int hist[256] = {0};
    int total = width * height;

    for (int i = 0; i < total; i++)
        hist[gray[i]]++;

    double sum = 0.0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];

    double sum_b = 0.0, w_b = 0.0;
    double max_var = 0.0;
    uint8_t best_thresh = 0;

    for (int t = 0; t < 256; t++) {
        w_b += hist[t];
        if (w_b == 0) continue;
        double w_f = total - w_b;
        if (w_f == 0) break;
        sum_b += t * hist[t];
        double m_b = sum_b / w_b;
        double m_f = (sum - sum_b) / w_f;
        double var = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if (var > max_var) {
            max_var = var;
            best_thresh = (uint8_t)t;
        }
    }
    return best_thresh;
}

void fast_binarize(
    const uint8_t *src, uint8_t *dst,
    int width, int height, uint8_t threshold
) {
    int total = width * height;
    for (int i = 0; i < total; i++)
        dst[i] = src[i] >= threshold ? 255 : 0;
}

/* ── 팽창 (Dilation, 3×3) ─────────────────────────────────── */
void fast_dilate(
    const uint8_t *src, uint8_t *dst, int width, int height
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t max_val = 0;
            for (int ky = -1; ky <= 1; ky++) {
                int ny = y + ky;
                if (ny < 0 || ny >= height) continue;
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    if (nx < 0 || nx >= width) continue;
                    uint8_t v = src[ny * width + nx];
                    if (v > max_val) max_val = v;
                }
            }
            dst[y * width + x] = max_val;
        }
    }
}

/* ── 침식 (Erosion, 3×3) ──────────────────────────────────── */
void fast_erode(
    const uint8_t *src, uint8_t *dst, int width, int height
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t min_val = 255;
            for (int ky = -1; ky <= 1; ky++) {
                int ny = y + ky;
                if (ny < 0 || ny >= height) continue;
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = x + kx;
                    if (nx < 0 || nx >= width) continue;
                    uint8_t v = src[ny * width + nx];
                    if (v < min_val) min_val = v;
                }
            }
            dst[y * width + x] = min_val;
        }
    }
}
