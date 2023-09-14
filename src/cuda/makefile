# Nom de l'exécutable
Executable = ethereum-vanity

# Répertoires source, objet et d'inclusion
SRC_DIR = /home/christophe/Documents/ethereum-vanity/src/cuda/src/
OBJ_DIR = /home/christophe/Documents/ethereum-vanity/src/cuda/src/
INCLUDE_DIR = /home/christophe/Documents/ethereum-vanity/src/cuda/include

# Compilateur CUDA (nvcc)
CC = nvcc

# Options de compilation
CFLAGS = -include include/config.hpp -I . -I /home/christophe/Téléchargements/secp256k1-master/include

# Options de l'éditeur de liens
LDFLAGS = -lsecp256k1 -lcurand -rdc=true

# Autres fichiers sources
SOURCES = $(wildcard $(SRC_DIR)/*.cu)

# Objets
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Cible par défaut
all: $(Executable)

# Règle de construction de l'exécutable
$(Executable): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Lien terminé !"

# Règle de compilation des objets
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compilation de "$<" réussie !"

# Cible pour nettoyer les fichiers générés
clean:
	rm -f $(OBJ_DIR)/*.o $(Executable)
