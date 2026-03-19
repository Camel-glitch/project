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