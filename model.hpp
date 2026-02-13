#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include "source.hpp"
#include "driver.hpp"
#include "walltime.hpp"
#include "walltime.hpp"
#include <cstdio>
// #include <memory>
#include "utils.hpp"

void Model(const int st, const int iSource, const float dtOutput,
    const int nx,   const int ny, const int nz,
    const int sx,   const int sy,   const int sz,
    const float dx, const float dy, const float dz,
    const float dt, const int it,
    const int bord, const int absorb,
    SlicePtr sPtr,
    int argc, char **argv);
