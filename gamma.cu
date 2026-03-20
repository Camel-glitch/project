#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include <curand_kernel.h>


__device__ double generate_gamma(double a, curandState *state) {
    if (a < 1.0) {
        double u = curand_uniform_double(state);
        return generate_gamma(a + 1.0, state) * pow(u, 1.0 / a);
    }

    // Algorithme de Marsaglia-Tsang (a >= 1)
    double d = a - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    double x, v, u;

    while (true) {
        do {
            x = curand_normal_double(state);//N(0,1)
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(state); // U(0,1)

        // Squeeze Test (imply the second test and faster)
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return d * v;
        }

      
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v))) {
            return d * v;
        }
    }