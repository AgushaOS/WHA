g++ -o code coder/*.cpp \
    -std=c++23 \
    -Ofast \
    -march=native -mtune=native \
    -mavx2 -mfma \
    -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math \
    -funroll-loops -fomit-frame-pointer \
    -flto=auto \
    -fno-semantic-interposition -fno-plt \
    -falign-functions=32 -falign-loops=32 -falign-jumps=32 \
    -DNDEBUG \
    -L/usr/local/lib -I/usr/local/include \
    -lfftw3f -lm

g++ -o decode decoder/*.cpp \
    -std=c++23 \
    -Ofast \
    -march=native -mtune=native \
    -mavx2 -mfma \
    -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math \
    -funroll-loops -fomit-frame-pointer \
    -flto=auto \
    -fno-semantic-interposition -fno-plt \
    -falign-functions=32 -falign-loops=32 -falign-jumps=32 \
    -DNDEBUG \
    -L/usr/local/lib -I/usr/local/include \
    -lfftw3f -lm