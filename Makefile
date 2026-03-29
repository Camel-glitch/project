# 1. Configuration des outils
NVCC = nvcc
CFLAGS = -O3 -Wall -Wno-deprecated-gpu-targets # Flags d'optimisation
TARGET = main

# 2. La "Magie" : Trouver tous les fichiers sources automatiquement
# On cherche tous les .c et tous les .cu
SRCS = $(wildcard *.c) $(wildcard *.cu)

# On transforme la liste des sources en liste d'objets (.o)
# Exemple : main.cu devient main.o, calculs.c devient calculs.o
OBJS = $(SRCS:.c=.o)
OBJS := $(OBJS:.cu=.o)

# 3. Règle principale (Linker)
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET)
	@echo "--- Compilation terminée avec succès ! ---"

# 4. Règle générique pour les fichiers .c -> .o
%.o: %.c
	$(NVCC) -c $< -o $@
	
# 5. Règle générique pour les fichiers .cu -> .o
%.o: %.cu
	$(NVCC) -c $< -o $@

# 6. Nettoyage
clean:
	rm -f *.o $(TARGET)
	@echo "--- Dossier nettoyé ---"