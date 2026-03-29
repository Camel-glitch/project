#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include <curand_kernel.h>


__device__ float generate_gamma(float a, curandState *state) {
    if (a < 1.0) {
        float u = curand_uniform_float(state);
        return generate_gamma(a + 1.0, state) * pow(u, 1.0 / a);
    }

    // Algorithme de Marsaglia-Tsang (a >= 1)
    float d = a - 1.0f / 3.0f;
    float c = 1.0f / sqrt(9.0f * d);
    float x, v, u;

    while (true) {
        do {
            x = curand_normal_float(state);//N(0,1)
            v = 1.0f + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_float(state); // U(0,1)

        // Squeeze Test (imply the second test and faster)
        if (u < 1.0f - 0.0331f * (x * x) * (x * x)) {
            return d * v;
        }

      
        if (log(u) < 0.5f * x * x + d * (1.0f - v + log(v))) {
            return d * v;
        }
    }
}