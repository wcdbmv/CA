#include <vector>
#include <math.h>
#include <iostream>
#include <functional>
#include <iomanip>
#include <chrono>

using std::log;
using std::exp;
using std::pow;
using std::abs;
using std::min;
using std::max;

typedef std::vector<double> Vector;
typedef std::vector<Vector> Matrix;

Matrix Q = {
  {2000, 4000,   6000,   8000,  10000,  12000,  14000,  16000,  18000,  20000,  22000,  24000,  26000},  // T, K
  {   1,    1,      1,  1.001, 1.0025, 1.0198, 1.0895, 1.2827, 1.6973, 2.4616, 3.6552, 5.3749, 7.6838},  // Q1
  {   4,    4, 4.1598, 4.3006, 4.4392, 4.5661, 4.6817, 4.7932, 4.9099, 5.0511, 5.2354, 5.4841, 5.8181},  // Q2
  { 5.5,  5.5, 5.5116, 5.9790, 6.4749, 6.9590, 7.4145, 7.8370, 8.2289, 8.5970, 8.9509, 9.3018, 9.6621},  // Q3
  {  11,   11,     11,     11,     11,     11,     11,     11,     11,     11,      11,    11,     11},  // Q4
  {  15,   15,     15,     15,     15,     15,     15,     15,     15,     15,      15,    15,     15}   // Q5
};

const Vector Z  = {0, 1, 2, 3, 4};
const Vector Z2 = {0, 1, 4, 9, 16};

const Vector E  = {12.13, 20.98, 31.0, 45.0};

const double EPS   = 1e-7;

int FindNearestPoint(Matrix& input_data, double x) {
  int data_len = static_cast<int>(input_data.size());

  int lower = 0;
  int upper = data_len;

  while (abs(upper - lower) > 1) {
    const int middle = (lower + upper) / 2;
    if (input_data[middle][0] > x)
      upper = middle;
    else
      lower = middle;
  }

  return abs(x - upper) > abs(x - lower) ? lower : upper;
}

inline double Div(double numerator, double denominator) {
  return denominator == 0.0 ? numerator / denominator : 0.0;
}

Matrix FindDivDiff(Matrix& input_data, int lower_edge, int upper_edge) {
  int n = upper_edge - lower_edge;
  Matrix div_diff(static_cast<size_t>(n + 1));

  for (int i = lower_edge; i <= upper_edge; ++i)
    div_diff[0].push_back(input_data[i][1]);

  for (int i = 1; i < n + 1; ++i)
    for (int j = 0; j < n + 1 - i; ++j)
      div_diff[i].push_back(Div(
          div_diff[i - 1][j + 1] - div_diff[i - 1][j],
          input_data[i + j + lower_edge][0] - input_data[j + lower_edge][0]
      ));

  return div_diff;
}

double FindValue(double x, Matrix& div_diff, Matrix& input_data) {
  double result = 0;
  int n = static_cast<int>(div_diff.size());

  for (int i = 0; i < n; ++i) {
    double tmp = 1;
    for (int j = 0; j < i; ++j)
      tmp *= x - input_data[j][0];
    tmp *= div_diff[i][0];

    result += tmp;
  }

  return result;
}

double Interpolate(Matrix& data, int n, double x) {
  n = max(static_cast<int>(data.size()) - 1, n);

  int data_len = static_cast<int>(data.size());
  int upper_edge = FindNearestPoint(data, x);
  int lower_edge = upper_edge;

  int n_points = min(n, data_len);

  while (n_points > 0) {
    if (upper_edge < data_len - 1) {
      ++upper_edge;
      --n_points;
    }
    if (n_points > 0 && lower_edge > 0) {
      --lower_edge;
      --n_points;
    }
  }

  Matrix data_slice;
  data_slice.reserve(upper_edge - lower_edge);
  for (int i = lower_edge; i < upper_edge; ++i)
    data_slice.push_back(data[i]);

  Matrix div_diff = FindDivDiff(data, lower_edge, upper_edge);
  return FindValue(x, div_diff, data_slice);
}

template <typename T>
constexpr T Sqr(T value) {
  return value * value;
}

double getGamma(double gamma, double t, Vector& x) {
  gamma /= 2;
  auto result = exp(x[0]) / (1 + gamma);
  for (int i = 1; i < 6; ++i)
    result += exp(x[i]) * Z2[i - 1] / (1 + Z2[i - 1] * gamma);
  result = 4 * Sqr(gamma) - result * 5.87 * 1e10 / pow(t, 3);
  return result;
}

double bisGamma(double start, double end, double t, Vector& x) {
  while (abs(start - end) > EPS) {
    auto middle = (start + end) / 2;

    if (getGamma(middle, t, x) <= 0)
      start = middle;
    else
      end = middle;
  }

  return (start + end) / 2;
}

inline double getAlpha(double gamma, double t) {
  return 0.285 * 1e-11 * pow(gamma * t, 3);
}

Vector getDeltaE(double gamma, double t) {
  Vector result(4);
  gamma /= 2;
  for (int i = 0; i < 4; ++i) {
    const double tmp = 1 + Z2[i + 1] * gamma;
    result[i] = 8.61 * 1e-5 * t * log(tmp * (1 + gamma) / tmp);
  }

  return result;
}

double InterpolateFromVectors(double x, int n, Vector& r1, Vector& r2) {
  Matrix data(r1.size());

  for (size_t i = 0; i < r1.size(); ++i) {
    data[i].push_back(r1[i]);
    data[i].push_back(r2[i]);
  }

  return Interpolate(data, n, x);
}

Vector getK(double t, const Vector& deltaE) {
  Vector k(4);

  for (int i = 0; i < 4; ++i) {
    const double qi1 = InterpolateFromVectors(t, 4, Q[0], Q[i + 2]);
    const double qi = InterpolateFromVectors(t, 4, Q[0], Q[i + 1]);
    k[i] = 0.00483 * qi1 / qi * pow(t, 1.5) * exp(-(E[i] - deltaE[i]) * 11603 / t);
  }

  return k;
}

inline double getT(double z, double t0, double tw, double m) {
  return t0 + (tw - t0) * pow(z, m);
}

double Integrate(double lb, double ub, const std::function<double(double)>& function) {
  const double step = 0.05;
  const double half_step = step / 2;
  double result = 0;

  while (lb <= ub) {
    const double left = function(lb);
    lb += step;
    const double right = function(lb);

    result += half_step * (left + right);
  }

  return result;
}

void SolveSystem(Matrix& left_side, Vector& right_side) {
  const int n = static_cast<int>(left_side.size());
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      const double sep = left_side[j][i] / left_side[i][i];
      for (int k = 0; k < n; ++k)
        left_side[j][k] -= left_side[i][k] * sep;
      right_side[j] -= right_side[i] * sep;
    }

  for (int i = n - 1; i >= 0; --i) {
    for (int k = i + 1; k < n; ++k)
      right_side[i] -= left_side[i][k] * right_side[k];
    right_side[i] /= left_side[i][i];
  }
}

double MaxDelta(const Vector& x, const Vector& deltaX) {
  double result = abs(deltaX[0] / x[0]);
  for (size_t i = 1; i < x.size(); ++i)
    result = max(abs(deltaX[i] / x[i]), result);
  return result;
}

double Nt(double t, double p, Vector& x) {
  while (true) {
    const double gamma = bisGamma(0, 3, t, x);
    const double alpha = getAlpha(gamma, t);
    const Vector deltaE = getDeltaE(gamma, t);
    Vector k = getK(t, deltaE);

    Matrix left_side = {
      {         1,         -1,                 1,                 0,                 0,                0},
      {         1,          0,                -1,                 1,                 0,                0},
      {         1,          0,                 0,                -1,                 1,                0},
      {         1,          0,                 0,                 0,                -1,                1},
      {-exp(x[0]), -exp(x[1]),        -exp(x[2]),        -exp(x[3]),        -exp(x[4]),        -exp(x[5])},
      { exp(x[0]),          0, -Z[1] * exp(x[2]), -Z[2] * exp(x[3]), -Z[3] * exp(x[4]), -Z[4] * exp(x[5])}
    };

    Vector right_side = {
      log(k[0]) + x[1] - x[2] - x[0],
      log(k[1]) + x[2] - x[3] - x[0],
      log(k[2]) + x[3] - x[4] - x[0],
      log(k[3]) + x[4] - x[5] - x[0],
      exp(x[0]) + exp(x[1]) + exp(x[2]) + exp(x[3]) + exp(x[4]) + exp(x[5]) - alpha - p * 7243 / t,
      Z[1] * exp(x[2]) + Z[2] * exp(x[3]) + Z[3] * exp(x[4]) + Z[4] * exp(x[5]) - exp(x[0])
    };

    SolveSystem(left_side, right_side);

    if (MaxDelta(x, right_side) < EPS)
      break;

    for (size_t i = 0; i < x.size(); ++i)
      x[i] += right_side[i];
  }

  double result = 0;
  for (int i = 0; i < 6; ++i)
    result += exp(x[i]);
  return result;
}

double Input(const char* prompt) {
  double res;
  std::cout << "Enter " << prompt << ": ";
  std::cin >> res;
  return res;
}

int main() {
  double p0 = Input("P0"),
         t0 = Input("T0"),
         tw = Input("Tw"),
         m  = Input("m");

  double p_end = 0.0;
  double p_start = 20.0;
  Vector x = {-1, 3, -1, -20, -20, -20};

  auto t1 = std::chrono::high_resolution_clock::now();

  double left_const = 7243 * p0 / 293 / 2;
  while (abs(p_end - p_start) > EPS) {
    const double middle_p = abs(p_end + p_start) / 2.0;
    const double integral = Integrate(0, 1, [&](double z) {
      return Nt(getT(z, t0, tw, m), middle_p, x) * z;
    });
    if (left_const - integral > 0)
      p_start = middle_p;
    else
      p_end = middle_p;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  std::cout << std::endl << "Result: " << std::fixed << std::setprecision(10) << p_end << std::endl;
  std::cout << "Time: " << std::fixed << std::setprecision(3) << duration / 1000000.0 << "s";

  return 0;
}
