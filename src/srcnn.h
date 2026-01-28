#ifndef _SRCNN_H_
#define _SRCNN_H_

#include "ap_fixed.h"
#include "hls_stream.h"

#define W  256
#define H  256
#define UP 3

typedef ap_fixed<16,4>  ftmap_t;
typedef ap_fixed<16,4>  param_t;
typedef ap_fixed<32,8>  acc_t;

#define TILE_H 32
#define TILE_W 32

static inline ftmap_t relu(ftmap_t x) { return x > (ftmap_t)0 ? x : (ftmap_t)0; }
static inline int clamp_i(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

#define N0 1
#define N1 64
#define F1 9
#define N2 32
#define F2 1
#define N3 1
#define F3 5

extern "C" void srcnn(
    ftmap_t *input_axi,
    param_t *conv1_w_axi, param_t *conv1_b_axi,
    param_t *conv2_w_axi, param_t *conv2_b_axi,
    param_t *conv3_w_axi, param_t *conv3_b_axi,
    ftmap_t *output_axi
);

void srcnn_core(
    ftmap_t input_ftmap[N0][H][W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    param_t conv2_weights[N2][N1][F2][F2],
    param_t conv2_biases[N2],
    param_t conv3_weights[N3][N2][F3][F3],
    param_t conv3_biases[N3],
    ftmap_t output_ftmap[N3][H][W]
);

void conv1(
    ftmap_t input_ftmap[N0][H][W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    ftmap_t output_ftmap[N1][H][W]
);

void conv1_stream(
    ftmap_t input_ftmap[N0][H][W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    hls::stream<ftmap_t> &S1,
    int y0, int x0, int th, int tw
);

#endif
