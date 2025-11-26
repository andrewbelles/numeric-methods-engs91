/*
 * 
 * quadrature.cpp  Andrew Belles  Oct 30th, 2025 
 *
 * Computes Romberg and Gaussian Quadratures, makes some analytic comparison 
 * on different methods 
 */ 

#include <csignal>
#include <iostream> 
#include <cmath> 
#include <unordered_map>
#include <vector> 
#include <tuple> 
#include <iomanip> 

typedef double (*functor)(const double&);

using LResult = std::tuple<double, double, size_t, size_t>; 
using SResult = std::tuple<double, size_t>;

std::unordered_map<int, std::vector<double>> roots, coeffs; 

static void gauss_root_table(std::unordered_map<int, std::vector<double>>& table);
static void gauss_coeff_table(std::unordered_map<int, std::vector<double>>& table);

static double first(const double& x); 
static double second(const double& x); 
static LResult romberg(const functor& f, const double& a, const double& b); 
static inline bool isSufficient(const double& Rprev, const double& Rcurr); 
static double composite(const functor& f, const double& a, const double& b, const size_t& n);
static SResult search_composite_best(const double& I, const functor& f, const double& a, const double& b);
static SResult gaussian_quad(const functor& f, const double& a, const double& b, const double& n);

static void evaluate(const std::string& label, const functor& f, const double& a, const double& b);

int main(void) 
{
  gauss_root_table(roots); 
  gauss_coeff_table(coeffs);

  evaluate("x^2*e^{-x} on interval [0, 1]", first, 0.0, 1.0);
  evaluate("x^{1/3} on interval [0, 1]", second, 0.0, 1.0);
  evaluate("x^2*e^{-x} on interval [1, 2]", first, 1.0, 2.0);
  evaluate("x^{1/3} on interval [1, 2]", second, 1.0, 2.0);

  return 0; 
}

static void 
evaluate(const std::string& label, const functor& f, const double& a, const double& b)
{
  size_t n = 0; 

  std::cout << std::setprecision(15); 
    
  auto [res, rn0, ext, evals] = romberg(f, a, b); 
  std::cout << "1)\n" << label << ": " << res << '\n'
            << "n: " << ext << " and " << evals << " function evaluations\n";

  std::cout << "trapezoidal approx: " << rn0 << '\n'
            << "Rn0, Rnn diff: " << std::abs(res - rn0) << '\n';

  auto [v, comp_n] = search_composite_best(res, f, a, b);
  std::cout << comp_n << " panels, " << comp_n + 1 << " evaluations required for " 
            << std::abs(v - res) << " difference\n\n";

  std::vector<SResult> gauss_results; 
  for (n = 1; n <= 5; n++) {
    gauss_results.push_back(gaussian_quad(f, a, b, n));
  }

  std::cout << "2)\n" << label << " gaussian quadrature\n"; 
  for (auto& [r, k] : gauss_results) {
    if ( k < 5 ) {
      std::cout << std::setprecision(8);
    } else { 
      std::cout << std::setprecision(15); 
    }
    std::cout << "I(b)=" << r << ", n=" << k << ", romberg diff: " 
              << std::abs(res - r) << '\n';   
  }
  std::cout << '\n';
}

static LResult 
romberg(const functor& f, const double& a, const double& b) 
{
  size_t i = 0, j = 0, n = 1, k = 2; 
  const double h0 = b - a;
  double sum = 0.0, h = 0.0; 
  const double boundary = f(a) + f(b); 
  
  std::vector<double> prow; 
  prow.push_back(0.5 * h0 * boundary);

  // compute Rn,1 
  while ( true ) {

    h = h0 / std::pow(2.0, n);  

    sum = 0.0; 
    size_t evals = (1 << (n - 1));
    for (i = 0; i < evals; i++) {
      double x = a + (2.0 * i + 1) * h; 
      sum += f(x);
    }
    k += evals;

    std::vector<double> crow(n + 1);
    crow[0] = 0.5 * prow[0] + h * sum; 

    // extrapolate upward to Rn,n 
    for (j = 1; j <= n; j++) {
      double den = std::pow(4.0, static_cast<int>(j)) - 1.0; 
      crow[j] = crow[j - 1] + ((crow[j - 1] - prow[j - 1]) / den);
    }
    
    if ( isSufficient(crow.back(), prow.back()) ) {
      return { crow.back(), crow.front(), n, k }; 
    } else {
      prow = std::move(crow); 
      n++; 
    }
  } 

  return {0.0, 0.0, 0, 0};
}

static inline bool 
isSufficient(const double& Rcurr, const double& Rprev)
{
  constexpr double tol = 1e-9; 
  if ( std::abs(Rcurr - Rprev) < tol ) {
    return true; 
  } else {
    return false; 
  }
}

static double 
composite(const functor& f, const double& a, const double& b, const size_t& n)
{
  const double h = (b - a) / n;
  double midsum = 0.0; 
  size_t i = 0; 

  for (i = 1; i < n; i++) {
    double x = a + static_cast<double>(i) * h; 
    midsum += f(x);
  }

  return 0.5 * h * (f(a) + 2.0 * midsum + f(b));
}

static SResult   
search_composite_best(const double& I, const functor& f, const double& a, const double& b)
{
  size_t n = 1; 
  double C = composite(f, a, b, n);
  constexpr size_t max_n = 1e5; 
  
  while ( n < max_n && std::abs(I - C) >= 1e-9 ) {
    C = composite(f, a, b, ++n);
  }

  return {C, n}; 
}

static SResult 
gaussian_quad(const functor& f, const double& a, const double& b, const double& n)
{
  // get points from table 
  std::vector<double> groots = roots[n], gcoeffs = coeffs[n];
  size_t k = groots.size(), i = 0; 
  std::vector<double> evals;
  double sum = 0.0; 
  evals.reserve(k);

  // project points into space 
  for (auto& r : groots) {
    evals.push_back(f((a + b) / 2.0 + (b - a) / 2.0 * r));  
  }

  for (i = 0; i < k; i++) {
    sum += gcoeffs[i] * evals[i]; 
  }
  sum *= 0.5 * (b - a);
  
  return {sum, k};
}

static double 
first(const double& x)
{
  return x * x * exp(-x); 
}

static double 
second(const double& x)
{
  return pow(x, 0.333333);
}

static void 
gauss_root_table(std::unordered_map<int, std::vector<double>>& table)
{
  const double a = std::sqrt(1.0 / 3.0);
  const double b = std::sqrt(3.0 / 5.0); 
  const double c[2] = {
    (1.0 / 35.0) * std::sqrt(525.0 + 70.0 * std::sqrt(30.0)), 
    (1.0 / 35.0) * std::sqrt(525.0 - 70.0 * std::sqrt(30.0))
  }; 
  const double d[2] = {
    (1.0 / 21.0) * std::sqrt(245.0 + 14.0 * std::sqrt(70.0)),
    (1.0 / 21.0) * std::sqrt(245.0 - 14.0 * std::sqrt(70.0))
  };

  table[1] = {0.0}; 
  table[2] = {-a, a};
  table[3] = {-b, 0.0, b}; 
  table[4] = {-c[0], -c[1], c[1], c[0]};
  table[5] = {-d[0], -d[1], 0.0, d[1], d[0]};
}


static void 
gauss_coeff_table(std::unordered_map<int, std::vector<double>>& table)
{
  const double a[2] = {1.0 / 36.0, std::sqrt(30.0)};
  const double b = 13.0 * std::sqrt(70.0);

  table[1] = {2.0};
  table[2] = {1.0, 1.0}; 
  table[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  table[4] = {
    a[0] * (18.0 - a[1]), 
    a[0] * (18.0 + a[1]), 
    a[0] * (18.0 + a[1]), 
    a[0] * (18.0 - a[1])
  }; 
  
  table[5] = {
    (1.0 / 900.0) * (322.0 - b), 
    (1.0 / 900.0) * (322.0 + b), 
    128.0 / 225.0,  
    (1.0 / 900.0) * (322.0 + b),
    (1.0 / 900.0) * (322.0 - b)
  }; 
}
