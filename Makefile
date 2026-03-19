# Variables pour faciliter les modifications futures
CC = nvcc
CFLAGS = -arch=sm_60 -O2	
# CFLAGS = -Wall -Wextra -g

# La première règle est celle exécutée par défaut quand on tape 'make'
mon_programme: main.o calculs.o
	$(CC) $(CFLAGS) main.o calculs.o -o mon_programme

# Comment créer main.o
main.o: main.c calculs.h
	$(CC) $(CFLAGS) -c main.c

# Comment créer calculs.o
calculs.o: calculs.c calculs.h
	$(CC) $(CFLAGS) -c calculs.c

# Règle de nettoyage pour supprimer les fichiers temporaires
clean:
	rm -f *.o mon_programme