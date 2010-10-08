#include <iostream>
#include <iomanip>
#include <cmath>

#define SQ(x) ((x) * (x))

static const float A = 0.0, B = 1.0;  // limites de integracion
static const int N = 1 << 20;         // numero de intervalos = 2^20
static const float H = (B - A) / N;   // taman~o del intervalo de integracion
static const float PI(M_PI);          // pi con precision simple

float f(float x) {
    float d = x - .43f;
    return 2.0f + 9.0f / (1.0f + 600.0f * SQ(d));
}

float g(float x) {
    float c = std::cos(2.0f * PI * f(x) * x);
    return std::exp(-x/4.0f) * SQ(c);
}

int main() {
    float sum = 0.0;
    float x;

//#   pragma omp parallel for private(x) reduction(+: sum)
    for (int i = 1; i < 2 * N; ++i) {
        x = A + i * (H / 2);
        sum += (i % 2 == 0 ? 4 : 2) * g(x);
    }
    sum += g(A) + g(B);

    float integral = sum * H / 6.0f;

    std::cout << std::setprecision(5) << integral << std::endl;
}

