#pragma once

#include <cstdlib>
#include <cmath>
#include "accessor.hpp"

// Fração da zona de absorção (mantido do original)
#define FRACABS 0.03125f

// RandomVelocityBoundary: creates a boundary with random velocity around domain
template <typename VIEW>
inline
void RandomVelocityBoundary(int sx, int sy, int sz,
                            int nx, int ny, int nz,
                            int bord, int absorb,
                            VIEW vpz, VIEW vsv)
{
    int i, ix, iy, iz;
    int distx, disty, distz, dist;
    int ivelx, ively, ivelz;
    float bordDist;
    float frac, rfac;
    int firstIn, bordLen;
    float maxP, maxS;

    using A = accessor<VIEW>;

    // --------------------------------------------------
    // maximum speed of P and S within bounds
    // --------------------------------------------------
    maxP = 0.0f;
    maxS = 0.0f;

    for (iz = bord + absorb; iz < nz + bord + absorb; iz++) {
        for (iy = bord + absorb; iy < ny + bord + absorb; iy++) {
            for (i = ind(bord + absorb, iy, iz, sx, sy);
                 i < ind(nx + bord + absorb, iy, iz, sx, sy);
                 i++) {

                maxP = fmaxf(A::get(vpz, i), maxP);
                maxS = fmaxf(A::get(vsv, i), maxS);
            }
        }
    }

    // --------------------------------------------------
    // boundary geometry
    // --------------------------------------------------
    bordLen = bord + absorb - 1;   // last index on low absorption zone
    firstIn = bordLen + 1;         // first index inside input grid
    frac    = 1.0f / (float)absorb;

    // --------------------------------------------------
    // fill boundary
    // --------------------------------------------------
    for (iz = 0; iz < sz; iz++) {
        for (iy = 0; iy < sy; iy++) {
            for (ix = 0; ix < sx; ix++) {

                i = ind(ix, iy, iz, sx, sy);

                // --------------------------------------------------
                // do nothing inside input grid
                // --------------------------------------------------
                if ((iz >= firstIn && iz <= bordLen + nz) &&
                    (iy >= firstIn && iy <= bordLen + ny) &&
                    (ix >= firstIn && ix <= bordLen + nx)) {
                    continue;
                }

                // --------------------------------------------------
                // random speed inside absorption zone
                // --------------------------------------------------
                else if ((iz >= bord && iz <= 2 * bordLen + nz) &&
                         (iy >= bord && iy <= 2 * bordLen + ny) &&
                         (ix >= bord && ix <= 2 * bordLen + nx)) {

                    // Z direction
                    if (iz > bordLen + nz) {
                        distz = iz - bordLen - nz;
                        ivelz = bordLen + nz;
                    } else if (iz < firstIn) {
                        distz = firstIn - iz;
                        ivelz = firstIn;
                    } else {
                        distz = 0;
                        ivelz = iz;
                    }

                    // Y direction
                    if (iy > bordLen + ny) {
                        disty = iy - bordLen - ny;
                        ively = bordLen + ny;
                    } else if (iy < firstIn) {
                        disty = firstIn - iy;
                        ively = firstIn;
                    } else {
                        disty = 0;
                        ively = iy;
                    }

                    // X direction
                    if (ix > bordLen + nx) {
                        distx = ix - bordLen - nx;
                        ivelx = bordLen + nx;
                    } else if (ix < firstIn) {
                        distx = firstIn - ix;
                        ivelx = firstIn;
                    } else {
                        distx = 0;
                        ivelx = ix;
                    }

                    // max distance to boundary
                    dist = (disty > distz) ? disty : distz;
                    dist = (dist  > distx) ? dist  : distx;

                    bordDist = (float)dist * frac;
                    rfac = (float)rand() / (float)RAND_MAX;

                    A::get(vpz, i) =
                        A::get(vpz, ind(ivelx, ively, ivelz, sx, sy)) *
                        (1.0f - bordDist) +
                        maxP * rfac * bordDist;

                    A::get(vsv, i) =
                        A::get(vsv, ind(ivelx, ively, ivelz, sx, sy)) *
                        (1.0f - bordDist) +
                        maxS * rfac * bordDist;
                }

                // --------------------------------------------------
                // null speed at border
                // --------------------------------------------------
                else {
                    A::get(vpz, i) = 0.0f;
                    A::get(vsv, i) = 0.0f;
                }
            }
        }
    }
}
