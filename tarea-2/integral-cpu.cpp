#include <iostream>
#include <iomanip>
#include <cmath>

#define SQ(x) ((x) * (x))

static const float A = -4.0, B = 4.0;  // limites de integracion
static const int N = 1 << 22;         // numero de intervalos = 2^20
static const float H = (B - A) / N;   // taman~o del intervalo de integracion
static const float PI(M_PI);          // pi con precision simple

float h(float x) {
    return .5f + 1.5f / (1.0f + 50.0f * SQ(x));
}

float f(float x) {
    int i;
    float sum = 0.0f, x0;
    for (i = 0; i < 10; ++i)
        x0 = -3.3f + i * 0.7f;
        sum += h(x - x0);
    return sum/10.0f;
}

float g(float x) {
    float c = std::cos(2.0f * PI * f(x) * x);
    return std::exp(-x/16.0f) * SQ(c);
}

int main() {
    float sum = 0.0;
    float x;

#   pragma omp parallel for private(x) reduction(+: sum)
    for (int i = 1; i < 2 * N; ++i) {
        x = A + i * (H / 2);
        sum += (i % 2 == 0 ? 4 : 2) * g(x);
    }
    sum += g(A) + g(B);

    float integral = sum * H / 6.0f;

    std::cout << std::setprecision(5) << integral << std::endl;
}

