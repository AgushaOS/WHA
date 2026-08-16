g++ -o code coder/coder.cpp \
-O3 -march=native -mtune=native \
-ffast-math -funroll-loops -fomit-frame-pointer \
-flto -fno-signed-zeros -fno-trapping-math -fassociative-math \
-fopenmp -DNDEBUG \
-march=native -ffast-math \
-std=c++23 \
 -lfftw3f -lm

g++ -o decode decoder/decoder.cpp \
-O3 -march=native -mtune=native \
-ffast-math -funroll-loops -fomit-frame-pointer \
-flto -fno-signed-zeros -fno-trapping-math -fassociative-math \
-fopenmp -DNDEBUG \
-march=native -ffast-math \
-std=c++23 \
 -lfftw3f -lm \
