/*
 * Single-vehicle, single-precision implementation of one 50 Hz step of the proposed controller
 * (FDI residuals, law lookup on detection, heading-steering guidance, position + attitude PID with model
 * feed-forward, restoring compensation, two-level priority allocation, thrust -> RPM), used to measure the
 * per-step cost and the law-switch latency (reviewer MC9).  Numerics follow sim.py; the plant is not simulated
 * here (inputs are randomised each step so the compiler cannot hoist anything).
 *
 * Build: gcc -O2 -std=c99 -o controller_step controller_step.c -lm
 * Run:   ./controller_step [iterations]
 */
#include <math.h>
#define EYAW_MAX 0.785398f   /* 45 deg heading-error saturation */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "laws.h"
#include "bmat.h"

#define DT 0.02f
static const float M1[3] = {19.86f, 20.62f, 32.18f}, M2[3] = {0.449f, 0.365f, 0.592f};
static const float DL[6] = {13.7f, 0.f, 33.f, 0.f, 0.8f, 0.f}, DQ[6] = {141.f, 217.f, 190.f, 1.19f, 0.47f, 1.5f};
static const float KP = 1.f, KD = 2.f, KI = 0.25f, KPRP = 9.f, KDRP = 6.f, KIRP = .3f, KPY = 2.25f, KDY = 3.f, KIY = .1f;
static const float W_N = 13.5f * 9.82f, B_N = 13.5f * 9.82f + 1.3f, ZB = -0.01f;
static const float UMAX = 51.5f, UMIN = -40.2f, KF = 51.5f / (3500.f * 3500.f), KR = 40.2f / (3500.f * 3500.f);

typedef struct {
    unsigned char health;         /* believed-healthy mask */
    const float (*K)[6];          /* active law */
    float d0[2]; int mode;
    float ei[3], eia[3], nexp[8]; int rescnt[8];
    float psi_state; int branch, hov_turn;
} ctrl_t;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static float wrapf(float a) { while (a > 3.14159265f) a -= 6.2831853f; while (a < -3.14159265f) a += 6.2831853f; return a; }

static void eulR(const float e[3], float R[3][3]) {
    float cr = cosf(e[0]), sr = sinf(e[0]), cp = cosf(e[1]), sp = sinf(e[1]), cy = cosf(e[2]), sy = sinf(e[2]);
    R[0][0] = cy * cp; R[0][1] = cy * sp * sr - sy * cr; R[0][2] = cy * sp * cr + sy * sr;
    R[1][0] = sy * cp; R[1][1] = sy * sp * sr + cy * cr; R[1][2] = sy * sp * cr - cy * sr;
    R[2][0] = -sp;     R[2][1] = cp * sr;                R[2][2] = cp * cr;
}

/* largest s in [0,1] with lo <= a + s b <= hi */
static float max_scale(const float a[8], const float b[8], const float lo[8], const float hi[8]) {
    float s = 1.f;
    for (int i = 0; i < 8; i++) {
        if (b[i] > 1e-12f) { float c = (hi[i] - a[i]) / b[i]; if (c < s) s = c; }
        else if (b[i] < -1e-12f) { float c = (lo[i] - a[i]) / b[i]; if (c < s) s = c; }
    }
    return s < 0.f ? 0.f : s;
}

/* one control step; returns 1 if a law switch happened */
static int step(ctrl_t *c, const float pm[3], const float em[3], const float num[6], const float nmeas[8],
                const float pr[3], const float vr[3], const float ar[3], float psir, float rr, float ncmd[8]) {
    int sw = 0;
    /* --- FDI: RPM residual + 0.25 s persistence, law lookup on detection */
    for (int i = 0; i < 8; i++) {
        float thr = fmaxf(250.f, 0.3f * fabsf(c->nexp[i]));
        c->rescnt[i] = (fabsf(nmeas[i] - c->nexp[i]) > thr) ? c->rescnt[i] + 1 : 0;
        if (c->rescnt[i] * DT >= 0.25f && (c->health >> i & 1)) { c->health &= ~(1u << i); sw = 1; }
    }
    if (sw) { c->K = LAW_K[c->health]; c->d0[0] = LAW_D0[c->health][0]; c->d0[1] = LAW_D0[c->health][1];
              c->mode = LAW_MODE[c->health]; c->psi_state = em[2]; c->branch = 0;
              for (int i = 0; i < 3; i++) c->ei[i] *= 0.5f; }
    float R[3][3]; eulR(em, R);
    /* --- position loop */
    float e[3], vw[3], ac[3];
    for (int i = 0; i < 3; i++) { e[i] = pr[i] - pm[i]; vw[i] = R[i][0] * num[0] + R[i][1] * num[1] + R[i][2] * num[2]; }
    for (int i = 0; i < 3; i++) {
        c->ei[i] = fminf(6.f, fmaxf(-6.f, c->ei[i] + e[i] * DT));
        ac[i] = ar[i] + KP * e[i] + KD * (vr[i] - vw[i]);
        ac[i] = fminf(0.6f, fmaxf(-0.6f, ac[i])) + KI * c->ei[i];      /* integral outside the limit */
    }
    /* --- guidance (heading-steered class) */
    float psid = psir, rd = rr;
    if (c->mode == 2) {
        float vh = hypotf(vr[0], vr[1]), eh = hypotf(e[0], e[1]); int moving = vh > 0.05f;
        float F0, F1;
        if (moving) { F0 = vr[0] + 0.4f * e[0]; F1 = vr[1] + 0.4f * e[1]; }
        else { c->hov_turn = eh > 0.3f ? 1 : (eh < 0.15f ? 0 : c->hov_turn); F0 = e[0]; F1 = e[1]; }
        float ang = atan2f(F1, F0) - atan2f(c->d0[1], c->d0[0]);
        float cur = c->psi_state + wrapf(ang + c->branch * 3.14159265f - c->psi_state);
        float alt = c->psi_state + wrapf(ang + (1 - c->branch) * 3.14159265f - c->psi_state);
        if (fabsf(alt - c->psi_state) + 0.698f < fabsf(cur - c->psi_state)) { c->branch = 1 - c->branch; cur = alt; }
        int upd = moving || (hypotf(F0, F1) > 0.08f && c->hov_turn);
        float rl = (moving ? 15.f : 10.f) * 0.0174533f * DT;
        float stp = fminf(rl, fmaxf(-rl, cur - c->psi_state));
        if (upd) c->psi_state += stp;
        psid = c->psi_state; rd = upd ? stp / DT : 0.f;
    } else if (c->mode >= 3) { psid = em[2]; rd = 0.f; }
    /* --- body wrench */
    float tau[6], ab[3], vb[3];
    for (int i = 0; i < 3; i++) { ab[i] = R[0][i] * ac[0] + R[1][i] * ac[1] + R[2][i] * ac[2];
                                  vb[i] = R[0][i] * vr[0] + R[1][i] * vr[1] + R[2][i] * vr[2]; }  /* reference velocity, body */
    for (int i = 0; i < 3; i++) tau[i] = M1[i] * ab[i] + DL[i] * vb[i] + DQ[i] * fabsf(vb[i]) * vb[i];
    float z[3] = {R[2][0], R[2][1], R[2][2]};
    float eatt[3] = {-(z[1] * 1.f - z[2] * 0.f), -(z[2] * 0.f - z[0] * 1.f), 0.f};
    float ey = wrapf(psid - em[2]);
    float eys = fminf(EYAW_MAX, fmaxf(-EYAW_MAX, ey));
    float patt[3];
    for (int i = 0; i < 3; i++) { patt[i] = KPRP * eatt[i] + KPY * eys * z[i]; eatt[i] += ey * z[i]; }
    const float kda[3] = {KDRP, KDRP, KDY}, kia[3] = {KIRP, KIRP, KIY};
    for (int i = 0; i < 3; i++) {
        c->eia[i] = fminf(.5f, fmaxf(-.5f, c->eia[i] + eatt[i] * DT));
        float w = num[3 + i];
        tau[3 + i] = M2[i] * (patt[i] + kda[i] * (rd * z[i] - w) + kia[i] * c->eia[i]) + DL[3 + i] * w + DQ[3 + i] * fabsf(w) * w;
    }
    /* restoring compensation  g = (W - B) z ; moment rb x (-B z) */
    float fg[3] = {(W_N - B_N) * z[0], (W_N - B_N) * z[1], (W_N - B_N) * z[2]};
    float mb[3] = {-(0.f * (-B_N * z[2]) - ZB * (-B_N * z[1])), -(ZB * (-B_N * z[0]) - 0.f), 0.f};
    for (int i = 0; i < 3; i++) { tau[i] -= fg[i]; tau[3 + i] -= mb[i]; }
    /* --- two-level priority allocation */
    float lo[8], hi[8], uv[8], uh[8], zero[8] = {0};
    for (int i = 0; i < 8; i++) { int h = c->health >> i & 1; lo[i] = h ? UMIN : 0.f; hi[i] = h ? UMAX : 0.f; uv[i] = uh[i] = 0.f; }
    for (int i = 0; i < 8; i++) {
        uv[i] = c->K[i][2] * tau[2] + c->K[i][3] * tau[3] + c->K[i][4] * tau[4];
        uh[i] = c->K[i][0] * tau[0] + c->K[i][1] * tau[1] + c->K[i][5] * tau[5];
    }
    float sv = max_scale(zero, uv, lo, hi);
    for (int i = 0; i < 8; i++) uv[i] *= sv;
    float sh = max_scale(uv, uh, lo, hi);
    for (int i = 0; i < 8; i++) {
        float u = fminf(hi[i], fmaxf(lo[i], uv[i] + sh * uh[i]));
        ncmd[i] = (c->health >> i & 1) ? (u >= 0 ? sqrtf(u / KF) : -sqrtf(-u / KR)) : 0.f;
        c->nexp[i] += 0.221199f * (ncmd[i] - c->nexp[i]);          /* 1 - exp(-DT / 0.08) */
    }
    return sw;
}

/* reference: recompute the weighted pseudo-inverse online (8x8 Gauss-Jordan) instead of a table lookup */
static void online_law(unsigned char h, float K[8][6]) {
    const float W2[6] = {1, 1, 4, 4, 4, 6.25f};
    float H[8][14];
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) {
        float s = 0; for (int r = 0; r < 6; r++) s += BMAT[r][i] * W2[r] * BMAT[r][j];
        int hi = h >> i & 1, hj = h >> j & 1;
        H[i][j] = (hi && hj) ? s + (i == j ? 1e-3f : 0.f) : (i == j ? 1.f : 0.f);
    }
    for (int i = 0; i < 8; i++) for (int r = 0; r < 6; r++) H[i][8 + r] = (h >> i & 1) ? BMAT[r][i] * W2[r] : 0.f;
    for (int p = 0; p < 8; p++) {
        float piv = H[p][p];
        for (int j = 0; j < 14; j++) H[p][j] /= piv;
        for (int i = 0; i < 8; i++) if (i != p) { float f = H[i][p]; for (int j = 0; j < 14; j++) H[i][j] -= f * H[p][j]; }
    }
    for (int i = 0; i < 8; i++) for (int r = 0; r < 6; r++) K[i][r] = H[i][8 + r];
}

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 2000000;
    ctrl_t c; memset(&c, 0, sizeof c); c.health = 0xFF; c.K = LAW_K[0xFF]; c.mode = 0;
    srand(1);
    float pm[3], em[3], num[6], nm[8], pr[3], vr[3], ar[3], ncmd[8];
    double worst = 0, sum = 0, sum2 = 0; long switches = 0; float sink = 0;
    double *samples = malloc(sizeof(double) * n);
    for (long it = 0; it < n; it++) {
        for (int i = 0; i < 3; i++) { pm[i] = rand() / (float)RAND_MAX; em[i] = 0.3f * (rand() / (float)RAND_MAX - .5f);
            pr[i] = rand() / (float)RAND_MAX; vr[i] = 0.3f * (rand() / (float)RAND_MAX - .5f); ar[i] = 0.f; }
        for (int i = 0; i < 6; i++) num[i] = 0.2f * (rand() / (float)RAND_MAX - .5f);
        for (int i = 0; i < 8; i++) nm[i] = c.nexp[i] + 20.f * (rand() / (float)RAND_MAX - .5f);
        if (it % 50000 == 49999) {                   /* inject a failure on a random healthy thruster: forces a switch */
            int th = rand() % 8; nm[th] = c.nexp[th] + 5000.f; c.rescnt[th] = 12;
            if (c.health == 0 || __builtin_popcount(c.health) < 4) { c.health = 0xFF; c.K = LAW_K[0xFF]; c.mode = 0; }
        }
        double t0 = now_s();
        switches += step(&c, pm, em, num, nm, pr, vr, ar, 0.1f, 0.f, ncmd);
        double dtc = now_s() - t0;
        samples[it] = dtc; sum += dtc; sum2 += dtc * dtc; if (dtc > worst) worst = dtc;
        sink += ncmd[0];
    }
    /* percentiles */
    int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
    qsort(samples, n, sizeof(double), cmp);
    double mean = sum / n;
    /* online law recomputation cost */
    float K[8][6]; double t0 = now_s(); int nl = 200000;
    for (int i = 0; i < nl; i++) { online_law((unsigned char)(0xFF & ~(1u << (i % 8))), K); sink += K[0][0]; }
    double tl = (now_s() - t0) / nl;
    printf("{\"iterations\": %ld, \"law_switches\": %ld, \"mean_us\": %.4f, \"median_us\": %.4f, \"p99_us\": %.4f, "
           "\"p99_99_us\": %.4f, \"worst_us\": %.3f, \"online_law_recompute_us\": %.4f, "
           "\"law_table_bytes\": %zu, \"ctrl_state_bytes\": %zu, \"sink\": %g}\n",
           n, switches, 1e6 * mean, 1e6 * samples[n / 2], 1e6 * samples[(long)(0.99 * n)], 1e6 * samples[(long)(0.9999 * n)],
           1e6 * worst, 1e6 * tl, sizeof(LAW_K) + sizeof(LAW_D0) + sizeof(LAW_MODE), sizeof(ctrl_t), sink);
    free(samples);
    return 0;
}
