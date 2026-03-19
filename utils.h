#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// Structure pour stocker les paramètres
typedef struct {
    float kappa;
    float theta;
    float sigma;
} HestonParams;

// Fonction utilitaire pour générer un float entre min et max
float rand_float(float min, float max);

HestonParams generate_valid_params();

void init_curand_state_k(curandState *state, unsigned long seed);   

double step_variance(double v_prev, float kappa, float theta, float sigma, float dt, curandState *state);

#endif // UTILS_H