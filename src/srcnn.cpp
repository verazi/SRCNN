#include "srcnn.h"

#define Tn 2
#define Tm 2

struct TileDesc { int y0, x0, th, tw; };
typedef hls::stream<ftmap_t> pix_stream_t;
typedef hls::stream<TileDesc> ctrl_stream_t;

static ftmap_t mid_buf[N2][TILE_H][TILE_W];

static void stage_conv1_tiles(
    ftmap_t input_ftmap[N0][H][W],
    param_t conv1_weights[N1][N0][F1][F1],
    param_t conv1_biases[N1],
    pix_stream_t &S1,
    ctrl_stream_t &ctrl12)
{
#pragma HLS INLINE off
    for (int y0 = 0; y0 < H; y0 += TILE_H)
        for (int x0 = 0; x0 < W; x0 += TILE_W) {
            const int th = (y0 + TILE_H <= H) ? TILE_H : (H - y0);
            const int tw = (x0 + TILE_W <= W) ? TILE_W : (W - x0);
            TileDesc t{y0, x0, th, tw};
            ctrl12.write(t);
            conv1_stream(input_ftmap, conv1_weights, conv1_biases, S1, y0, x0, th, tw);
        }
}

static void conv2(
    pix_stream_t &S1,
    param_t  conv2_weights[N2][N1][F2][F2],
    param_t  conv2_biases[N2],
    int th, int tw)
{
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=mid_buf type=ram_t2p impl=uram
#pragma HLS DEPENDENCE variable=mid_buf inter false
    static ftmap_t in_buf[N1][TILE_H][TILE_W];
#pragma HLS BIND_STORAGE variable=in_buf type=ram_t2p impl=bram
#pragma HLS DEPENDENCE variable=in_buf inter false
    for (int n = 0; n < N1; ++n)
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                in_buf[n][y][x] = S1.read();
            }
    for (int m0 = 0; m0 < N2; m0 += Tm)
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                acc_t acc[Tm];
#pragma HLS ARRAY_PARTITION variable=acc complete
                for (int tm = 0; tm < Tm; ++tm) acc[tm] = (m0+tm<N2)?(acc_t)conv2_biases[m0+tm]:(acc_t)0;
                for (int n0 = 0; n0 < N1; n0 += Tn) {
                    ftmap_t v[Tn];
#pragma HLS ARRAY_PARTITION variable=v complete
                    for (int tn = 0; tn < Tn; ++tn) v[tn] = (n0+tn<N1)?in_buf[n0+tn][y][x]:(ftmap_t)0;
                    for (int tm = 0; tm < Tm; ++tm) {
#pragma HLS UNROLL
                        acc_t s = acc[tm];
                        for (int tn = 0; tn < Tn; ++tn) {
#pragma HLS UNROLL
                            param_t w = (m0+tm<N2 && n0+tn<N1)?conv2_weights[m0+tm][n0+tn][0][0]:(param_t)0;
                            s += (acc_t)v[tn] * (acc_t)w;
                        }
                        acc[tm] = s;
                    }
                }
                for (int tm = 0; tm < Tm; ++tm)
                    if (m0 + tm < N2) mid_buf[m0 + tm][y][x] = relu((ftmap_t)acc[tm]);
            }
}

static void stage_conv2_tiles(
    param_t  conv2_weights[N2][N1][F2][F2],
    param_t  conv2_biases[N2],
    pix_stream_t &S1,
    ctrl_stream_t &ctrl12,
    ctrl_stream_t &ctrl23)
{
#pragma HLS INLINE off
    for (int y0 = 0; y0 < H; y0 += TILE_H)
        for (int x0 = 0; x0 < W; x0 += TILE_W) {
            TileDesc t = ctrl12.read();
            ctrl23.write(t);
            conv2(S1, conv2_weights, conv2_biases, t.th, t.tw);
        }
}

static void conv3(
    param_t  conv3_weights[N3][N2][F3][F3],
    param_t  conv3_biases[N3],
    ftmap_t  output_ftmap[N3][H][W],
    int y0, int x0, int th, int tw, int K3)
{
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=mid_buf type=ram_t2p impl=uram
#pragma HLS DEPENDENCE variable=mid_buf inter false
    const int PAD = K3 / 2;
    static acc_t out_acc[Tm][TILE_H][TILE_W];
#pragma HLS BIND_STORAGE variable=out_acc type=ram_t2p
#pragma HLS ARRAY_PARTITION variable=out_acc complete dim=1
    static ftmap_t linebuf[Tn][F3-1][TILE_W + F3 - 1];
#pragma HLS BIND_STORAGE variable=linebuf type=ram_t2p impl=uram
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=2
    static acc_t window[Tn][F3][F3];
#pragma HLS ARRAY_PARTITION variable=window complete dim=1
#pragma HLS ARRAY_PARTITION variable=window complete dim=2
#pragma HLS ARRAY_PARTITION variable=window complete dim=3
    for (int m0 = 0; m0 < N3; m0 += Tm) {
        for (int tm = 0; tm < Tm; ++tm) {
#pragma HLS UNROLL
            acc_t b = (m0+tm < N3) ? (acc_t)conv3_biases[m0+tm] : (acc_t)0;
            for (int y = 0; y < th; ++y)
                for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                    out_acc[tm][y][x] = b;
                }
        }
        for (int n0 = 0; n0 < N2; n0 += Tn) {
            for (int t = 0; t < Tn; ++t) {
#pragma HLS UNROLL
                for (int ky = 0; ky < K3-1; ++ky) {
#pragma HLS UNROLL
                    for (int xp = 0; xp < tw + K3 - 1; ++xp) {
#pragma HLS PIPELINE II=1
                        linebuf[t][ky][xp] = (ftmap_t)0;
                    }
                }
                for (int ky = 0; ky < K3; ++ky)
                    for (int kx = 0; kx < K3; ++kx) {
#pragma HLS UNROLL
                        window[t][ky][kx] = (acc_t)0;
                    }
            }
            for (int yp = 0; yp < th + K3 - 1; ++yp)
                for (int xp = 0; xp < tw + K3 - 1; ++xp) {
#pragma HLS PIPELINE II=1
                    for (int t = 0; t < Tn; ++t) {
#pragma HLS UNROLL
                        ftmap_t newpix_ft = (ftmap_t)0;
                        int n = n0 + t;
                        if (yp >= PAD && yp < th + PAD && xp >= PAD && xp < tw + PAD && n < N2)
                            newpix_ft = mid_buf[n][yp - PAD][xp - PAD];
                        acc_t newpix = (acc_t)newpix_ft;
                        for (int ky = 0; ky < K3; ++ky) {
#pragma HLS UNROLL
                            for (int kx = 0; kx < K3 - 1; ++kx) {
#pragma HLS UNROLL
                                window[t][ky][kx] = window[t][ky][kx + 1];
                            }
                        }
                        for (int ky = 0; ky < K3 - 1; ++ky) {
#pragma HLS UNROLL
                            acc_t from_lb = (acc_t)linebuf[t][ky][xp];
                            window[t][ky][K3 - 1] = from_lb;
                        }
                        for (int ky = 0; ky < K3 - 1; ++ky) {
#pragma HLS UNROLL
                            linebuf[t][ky][xp] = (ftmap_t)window[t][ky + 1][K3 - 1];
                        }
                        window[t][K3 - 1][K3 - 1] = newpix;
                    }
                    if (yp >= K3 - 1 && xp >= K3 - 1) {
                        int y = yp - (K3 - 1);
                        int x = xp - (K3 - 1);
                        for (int tm = 0; tm < Tm; ++tm) {
#pragma HLS UNROLL
                            acc_t acc = out_acc[tm][y][x];
                            for (int t = 0; t < Tn; ++t) {
#pragma HLS UNROLL
                                if (n0 + t < N2) {
                                    acc_t sum = (acc_t)0;
                                    for (int ky = 0; ky < K3; ++ky)
                                        for (int kx = 0; kx < K3; ++kx) {
#pragma HLS UNROLL
                                            sum += window[t][ky][kx] * (acc_t)conv3_weights[m0+tm][n0+t][ky][kx];
                                        }
                                    acc += sum;
                                }
                            }
                            out_acc[tm][y][x] = acc;
                        }
                    }
                }
        }
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
#pragma HLS PIPELINE II=1
                for (int tm = 0; tm < Tm; ++tm) {
#pragma HLS UNROLL
                    if (m0 + tm < N3)
                        output_ftmap[m0 + tm][y0 + y][x0 + x] = (ftmap_t)out_acc[tm][y][x];
                }
            }
    }
}

static void stage_conv3_tiles(
    param_t  conv3_weights[N3][N2][F3][F3],
    param_t  conv3_biases[N3],
    ctrl_stream_t &ctrl23,
    ftmap_t  output_ftmap[N3][H][W])
{
#pragma HLS INLINE off
    for (int y0 = 0; y0 < H; y0 += TILE_H)
        for (int x0 = 0; x0 < W; x0 += TILE_W) {
            TileDesc t = ctrl23.read();
            conv3(conv3_weights, conv3_biases, output_ftmap, t.y0, t.x0, t.th, t.tw, F3);
        }
}

void srcnn_core(
    ftmap_t input_ftmap[N0][H][W],
    param_t  conv1_weights[N1][N0][F1][F1],
    param_t  conv1_biases[N1],
    param_t  conv2_weights[N2][N1][F2][F2],
    param_t  conv2_biases[N2],
    param_t  conv3_weights[N3][N2][F3][F3],
    param_t  conv3_biases[N3],
    ftmap_t  output_ftmap[N3][H][W])
{
#pragma HLS DATAFLOW
    pix_stream_t S1;
#pragma HLS STREAM variable=S1 depth=8192
    ctrl_stream_t ctrl12, ctrl23;
#pragma HLS STREAM variable=ctrl12 depth=128
#pragma HLS STREAM variable=ctrl23 depth=128
    stage_conv1_tiles(input_ftmap, conv1_weights, conv1_biases, S1, ctrl12);
    stage_conv2_tiles(conv2_weights, conv2_biases, S1, ctrl12, ctrl23);
    stage_conv3_tiles(conv3_weights, conv3_biases, ctrl23, output_ftmap);
}

extern "C" void srcnn(
    ftmap_t *input_axi,
    param_t *conv1_w_axi, param_t *conv1_b_axi,
    param_t *conv2_w_axi, param_t *conv2_b_axi,
    param_t *conv3_w_axi, param_t *conv3_b_axi,
    ftmap_t *output_axi)
{
#pragma HLS INTERFACE m_axi port=input_axi   offset=slave bundle=gmem0 depth=(N0*H*W)
#pragma HLS INTERFACE m_axi port=output_axi  offset=slave bundle=gmem1 depth=(N3*H*W)

// conv1
#pragma HLS INTERFACE m_axi port=conv1_w_axi offset=slave bundle=gmem2_c1w depth=(N1*N0*F1*F1)
#pragma HLS INTERFACE m_axi port=conv1_b_axi offset=slave bundle=gmem2_c1b depth=(N1)

// conv2
#pragma HLS INTERFACE m_axi port=conv2_w_axi offset=slave bundle=gmem2_c2w depth=(N2*N1*F2*F2)
#pragma HLS INTERFACE m_axi port=conv2_b_axi offset=slave bundle=gmem2_c2b depth=(N2)

// conv3
#pragma HLS INTERFACE m_axi port=conv3_w_axi offset=slave bundle=gmem2_c3w depth=(N3*N2*F3*F3)
#pragma HLS INTERFACE m_axi port=conv3_b_axi offset=slave bundle=gmem2_c3b depth=(N3)

// control
#pragma HLS INTERFACE s_axilite port=input_axi   bundle=control
#pragma HLS INTERFACE s_axilite port=output_axi  bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_w_axi bundle=control
#pragma HLS INTERFACE s_axilite port=conv1_b_axi bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_w_axi bundle=control
#pragma HLS INTERFACE s_axilite port=conv2_b_axi bundle=control
#pragma HLS INTERFACE s_axilite port=conv3_w_axi bundle=control
#pragma HLS INTERFACE s_axilite port=conv3_b_axi bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    static ftmap_t input_buf [N0][H][W];
    static ftmap_t output_buf[N3][H][W];
    static param_t c1w[N1][N0][F1][F1], c2w[N2][N1][F2][F2], c3w[N3][N2][F3][F3];
    static param_t c1b[N1], c2b[N2], c3b[N3];
#pragma HLS BIND_STORAGE variable=input_buf  type=ram_t2p impl=uram
#pragma HLS BIND_STORAGE variable=output_buf type=ram_t2p
#pragma HLS BIND_STORAGE variable=c1w type=ram_t2p
#pragma HLS BIND_STORAGE variable=c2w type=ram_t2p
#pragma HLS BIND_STORAGE variable=c3w type=ram_t2p

    for (int c=0; c<N0; ++c)
        for (int y=0; y<H; ++y)
            for (int x=0; x<W; ++x) {
#pragma HLS PIPELINE II=1
                input_buf[c][y][x] = input_axi[c*H*W + y*W + x];
            }

    for (int co=0; co<N1; ++co)
        for (int ci=0; ci<N0; ++ci)
            for (int ky=0; ky<F1; ++ky)
                for (int kx=0; kx<F1; ++kx) {
#pragma HLS PIPELINE II=1
                    int idx = (((co*N0)+ci)*F1 + ky)*F1 + kx;
                    c1w[co][ci][ky][kx] = conv1_w_axi[idx];
                }
    for (int i=0; i<N1; ++i) {
#pragma HLS PIPELINE II=1
        c1b[i] = conv1_b_axi[i];
    }

    for (int co=0; co<N2; ++co)
        for (int ci=0; ci<N1; ++ci)
            for (int ky=0; ky<F2; ++ky)
                for (int kx=0; kx<F2; ++kx) {
#pragma HLS PIPELINE II=1
                    int idx = (((co*N1)+ci)*F2 + ky)*F2 + kx;
                    c2w[co][ci][ky][kx] = conv2_w_axi[idx];
                }
    for (int i=0; i<N2; ++i) {
#pragma HLS PIPELINE II=1
        c2b[i] = conv2_b_axi[i];
    }

    for (int co=0; co<N3; ++co)
        for (int ci=0; ci<N2; ++ci)
            for (int ky=0; ky<F3; ++ky)
                for (int kx=0; kx<F3; ++kx) {
#pragma HLS PIPELINE II=1
                    int idx = (((co*N2)+ci)*F3 + ky)*F3 + kx;
                    c3w[co][ci][ky][kx] = conv3_w_axi[idx];
                }
    for (int i=0; i<N3; ++i) {
#pragma HLS PIPELINE II=1
        c3b[i] = conv3_b_axi[i];
    }

    srcnn_core(input_buf, c1w, c1b, c2w, c2b, c3w, c3b, output_buf);

    for (int c=0; c<N3; ++c)
        for (int y=0; y<H; ++y)
            for (int x=0; x<W; ++x) {
#pragma HLS PIPELINE II=1
                output_axi[c*H*W + y*W + x] = output_buf[c][y][x];
            }
}
