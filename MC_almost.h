#ifndef MC_ALMOST_H
#define MC_ALMOST_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include "gamma.h"
#include "utils.h"


__global__ void MC_almost(float rho, float v_0, float S_0, float r, float sigma, float k, float theta, float dt, float K, int N,
            curandState* state, float* PayGPU);

#endif // MC_ALMOST_H