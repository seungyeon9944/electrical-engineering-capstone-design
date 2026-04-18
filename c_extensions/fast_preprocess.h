#ifndef FAST_PREPROCESS_H
#define FAST_PREPROCESS_H

#include <stdint.h>

/* 고속 가우시안 블러 (3×3 고정 커널) */
void fast_gaussian_blur(
    const uint8_t *src, uint8_t *dst, int width, int height
);

/* Otsu 이진화 */
uint8_t otsu_threshold(const uint8_t *gray, int width, int height);

void fast_binarize(
    const uint8_t *src, uint8_t *dst,
    int width, int height, uint8_t threshold
);

/* 팽창 연산 (3×3 커널) */
void fast_dilate(
    const uint8_t *src, uint8_t *dst, int width, int height
);

/* 침식 연산 (3×3 커널) */
void fast_erode(
    const uint8_t *src, uint8_t *dst, int width, int height
);

#endif /* FAST_PREPROCESS_H */
