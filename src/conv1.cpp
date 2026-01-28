#include "srcnn.h"

void conv1_stream(
    ftmap_t input_ftmap[N0][H][W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    hls::stream<ftmap_t> &S1,
    int y0, int x0, int th, int tw)
{
#pragma HLS INLINE off
    const int K = F1;
    const int PAD = K / 2;
#pragma HLS ARRAY_RESHAPE variable=conv1_weights cyclic factor=F1 dim=3
#pragma HLS ARRAY_RESHAPE variable=conv1_weights cyclic factor=F1 dim=4
    for (int co = 0; co < N1; ++co) {
        static acc_t acc_tile[TILE_H][TILE_W];
#pragma HLS BIND_STORAGE variable=acc_tile type=ram_t2p impl=bram
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                acc_tile[y][x] = (acc_t)0;
            }
        for (int ci = 0; ci < N0; ++ci) {
            static ftmap_t linebuf[F1 - 1][TILE_W + F1 - 1];
            static acc_t   win    [F1][F1];
#pragma HLS BIND_STORAGE  variable=linebuf type=ram_t2p impl=bram
#pragma HLS ARRAY_PARTITION variable=win complete dim=0
            for (int ty = 0; ty < th + K - 1; ++ty)
                for (int tx = 0; tx < tw + K - 1; ++tx) {
#pragma HLS PIPELINE II=1
                    int yg = clamp_i(y0 + ty - PAD, 0, H - 1);
                    int xg = clamp_i(x0 + tx - PAD, 0, W - 1);
                    ftmap_t pix = input_ftmap[ci][yg][xg];
                    for (int wy = 0; wy < K; ++wy)
                        for (int wx = 0; wx < K - 1; ++wx)
                            win[wy][wx] = win[wy][wx + 1];
                    for (int wy = 0; wy < K - 1; ++wy) {
                        ftmap_t v = linebuf[wy][tx];
                        win[wy][K - 1] = (acc_t)v;
                    }
                    win[K - 1][K - 1] = (acc_t)pix;
                    for (int wy = 0; wy < K - 2; ++wy)
                        linebuf[wy][tx] = linebuf[wy + 1][tx];
                    if (K > 1) linebuf[K - 2][tx] = pix;
                    if (ty >= K - 1 && tx >= K - 1) {
                        int y = ty - (K - 1);
                        int x = tx - (K - 1);
                        acc_t s = (acc_t)0;
                        for (int ky = 0; ky < K; ++ky) {
#pragma HLS UNROLL
                            acc_t r = (acc_t)0;
                            for (int kx = 0; kx < K; ++kx) {
#pragma HLS UNROLL
                                r += win[ky][kx] * (acc_t)conv1_weights[co][ci][ky][kx];
                            }
                            s += r;
                        }
                        acc_tile[y][x] += s;
                    }
                }
        }
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                acc_t acc = (acc_t)conv1_biases[co] + acc_tile[y][x];
                ftmap_t v = relu((ftmap_t)acc);
                S1.write(v);
            }
    }
}

void conv1(
    ftmap_t input_ftmap[N0][H][W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    ftmap_t output_ftmap[N1][H][W])
{
#pragma HLS INLINE off
    const int K = F1;
    const int PAD = K / 2;
#pragma HLS ARRAY_RESHAPE variable=conv1_weights cyclic factor=F1 dim=3
#pragma HLS ARRAY_RESHAPE variable=conv1_weights cyclic factor=F1 dim=4
    for (int y0 = 0; y0 < H; y0 += TILE_H)
        for (int x0 = 0; x0 < W; x0 += TILE_W) {
            const int th = (y0 + TILE_H <= H) ? TILE_H : (H - y0);
            const int tw = (x0 + TILE_W <= W) ? TILE_W : (W - x0);
            for (int co = 0; co < N1; ++co) {
                static acc_t acc_tile[TILE_H][TILE_W];
#pragma HLS BIND_STORAGE variable=acc_tile type=ram_t2p impl=bram
                for (int y = 0; y < th; ++y)
                    for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                        acc_tile[y][x] = (acc_t)0;
                    }
                for (int ci = 0; ci < N0; ++ci) {
                    static ftmap_t linebuf[F1 - 1][TILE_W + F1 - 1];
                    static acc_t   win    [F1][F1];
#pragma HLS BIND_STORAGE  variable=linebuf type=ram_t2p impl=bram
#pragma HLS ARRAY_PARTITION variable=win complete dim=0
                    for (int ty = 0; ty < th + K - 1; ++ty)
                        for (int tx = 0; tx < tw + K - 1; ++tx) {
#pragma HLS PIPELINE II=1
                            int yg = clamp_i(y0 + ty - PAD, 0, H - 1);
                            int xg = clamp_i(x0 + tx - PAD, 0, W - 1);
                            ftmap_t pix = input_ftmap[ci][yg][xg];
                            for (int wy = 0; wy < K; ++wy)
                                for (int wx = 0; wx < K - 1; ++wx)
                                    win[wy][wx] = win[wy][wx + 1];
                            for (int wy = 0; wy < K - 1; ++wy) {
                                ftmap_t v = linebuf[wy][tx];
                                win[wy][K - 1] = (acc_t)v;
                            }
                            win[K - 1][K - 1] = (acc_t)pix;
                            for (int wy = 0; wy < K - 2; ++wy)
                                linebuf[wy][tx] = linebuf[wy + 1][tx];
                            if (K > 1) linebuf[K - 2][tx] = pix;
                            if (ty >= K - 1 && tx >= K - 1) {
                                int y = ty - (K - 1);
                                int x = tx - (K - 1);
                                acc_t s = (acc_t)0;
                                for (int ky = 0; ky < K; ++ky) {
#pragma HLS UNROLL
                                    acc_t r = (acc_t)0;
                                    for (int kx = 0; kx < K; ++kx) {
#pragma HLS UNROLL
                                        r += win[ky][kx] * (acc_t)conv1_weights[co][ci][ky][kx];
                                    }
                                    s += r;
                                }
                                acc_tile[y][x] += s;
                            }
                        }
                }
                for (int y = 0; y < th; ++y)
                    for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                        acc_t acc = (acc_t)conv1_biases[co] + acc_tile[y][x];
                        output_ftmap[co][y0 + y][x0 + x] = relu((ftmap_t)acc);
                    }
            }
        }
}
