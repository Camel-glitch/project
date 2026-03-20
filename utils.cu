#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <curand_kernel.h>
#include "utils.h"


// Structure pour stocker les paramètres
typedef struct {
    float kappa;
    float theta;
    float sigma;
} HestonParams;


// Fonction utilitaire pour générer un float entre min et max
float rand_float(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

HestonParams generate_valid_params() {
    HestonParams p;
    bool is_valid = false;

    while (!is_valid) {
        p.kappa = rand_float(0.1f, 10.0f);
        p.theta = rand_float(0.01f, 0.5f);
        p.sigma = rand_float(0.1f, 1.0f);

        // Condition de Feller : 2 * kappa * theta > sigma^2
        // Note : J'utilise 2.0f, mais tu peux mettre 20.0f si c'est ta contrainte spécifique
        if ((20.0f * p.kappa * p.theta) > (p.sigma * p.sigma)) {
            is_valid = true;
        }
    }
    return p;
}

// Set the state for each thread
__global__ void init_curand_state_k(curandStateXORWOW *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    /* curand_init parameters:
       1. seed: Use the same one for all threads (e.g., 1234).
       2. subsequence: USE THE THREAD ID. This ensures each thread 
          gets a unique, non-overlapping sequence of numbers.
       3. offset: Usually 0 (starts at the beginning of the sequence).
       4. state: Pointer to this thread's specific memory slot.
    */
    curand_init(seed, idx, 0, &state[idx]);
}


__device__ double step_variance(double vt, double kappa, double theta, double sigma, double dt, curandState *state) {
    // 1. Précalcul des constantes pour ce pas de temps
    double exp_kdt = exp(-kappa * dt);
    double sigma2 = sigma * sigma;
    double scale = (sigma2 * (1.0 - exp_kdt)) / (4.0 * kappa); // Facteur multiplicatif final

    // 2. Calcul des paramètres d et lambda
    // Note : d dans ton énoncé est le "degré de liberté" divisé par 2
    double d = (2.0 * kappa * theta) / sigma2;
    double lambda = (4.0 * kappa * exp_kdt * vt) / (sigma2 * (1.0 - exp_kdt));

    // 3. Simulation de la composante Poisson (N)
    // N ~ Poisson(lambda / 2)
    unsigned int N = curand_poisson(state, lambda / 2.0);

    // 4. Simulation de la Gamma standard G(alpha)
    // alpha = d + N
    double alpha = d + N;
    double G = generate_gamma(alpha, state); // Utilise ta device function Marsaglia-Tsang

    // 5. Calcul de vt+dt
    // La formule vt+dt = scale * 2 * G(d + N) correspond à la loi du Khi-deux non-centrale
    return scale * 2.0 * G;
}
