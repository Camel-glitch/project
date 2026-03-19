int main() {
    // Initialisation du générateur (à faire une seule fois au début du programme)
    srand((unsigned int)time(NULL));

    // Exemple de génération
    HestonParams my_params = generate_valid_params();

    printf("Parametres generes :\n");
    printf("kappa : %.4f\n", my_params.kappa);
    printf("theta : %.4f\n", my_params.theta);
    printf("sigma : %.4f\n", my_params.sigma);
    printf("Feller check (2kt > s^2) : %.4f > %.4f\n", 
            2.0f * my_params.kappa * my_params.theta, 
            my_params.sigma * my_params.sigma);

    return 0;
}