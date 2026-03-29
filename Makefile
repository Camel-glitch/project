# 1. Configuration des outils
NVCC = nvcc
# On sépare les flags : -rdc=true est indispensable ici
# On ajoute souvent -arch=sm_xx pour spécifier l'architecture (ex: sm_75, sm_80)
NVCCFLAGS = -O3 -rdc=true -Wno-deprecated-gpu-targets 
TARGET = main

# 2. Trouver tous les fichiers sources automatiquement
SRCS_C = $(wildcard *.c)
SRCS_CU = $(wildcard *.cu)

# On transforme la liste des sources en liste d'objets (.o)
OBJS = $(SRCS_C:.c=.o) $(SRCS_CU:.cu=.o)

# 3. Règle principale (Linker)
# ATTENTION : -rdc=true doit aussi être présent à l'étape du linkage
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET)
	@echo "--- Compilation multi-fichiers terminée avec succès ! ---"

# 4. Règle générique pour les fichiers .c -> .o
%.o: %.c
	$(NVCC) -c $< -o $@
    
# 5. Règle générique pour les fichiers .cu -> .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 6. Nettoyage
clean:
	rm -f *.o $(TARGET)
	@echo "--- Dossier nettoyé ---"