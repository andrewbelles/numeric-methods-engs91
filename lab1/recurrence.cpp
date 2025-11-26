/*
 * recurrence.cpp  Andrew Belles  Sept 16th, 2025 
 *
 * Defines methods for computing nth bessel functions
 * from initial conditions, as well as computing error 
 *
 */

#include <iostream> 
#include <cmath> 
#include <vector> 
#include <array> 
#include <cstdint>
#include <cstdio> 

/*
 * Cyclindrical bessel values to compare computed values to 
 */ 
std::vector<double> 
besselj(double x, uint32_t N)
{
  std::vector<double> vec(N);
  for (uint32_t i(0); i < N; i++) {
    vec[i] = std::cyl_bessel_j(static_cast<double>(i), x); 
  }
  return vec; 
}

class Bessel {
public: 
  std::vector<std::vector<double>> computed;  // computed from recurrence  
  std::vector<std::vector<double>> error;     // residual error 

  Bessel(
      std::array<double, 3> x_values, 
      std::array<std::pair<double, double>, 3> ic, 
      uint32_t N,
      bool forward
  ) : forward_(forward), initial_conditions_(ic), N_(N), x_values_(x_values) {

      for (auto& x : x_values) {
        real_.push_back(besselj(x, N));
      }
  }

  void 
  run() 
  {
    compute_recurrence_();
    compute_error_(); 
  }

private: 
  std::array<std::pair<double, double>, 3> initial_conditions_; // J_0,  J_1 
  bool forward_; 
  uint32_t N_; 
  std::vector<std::vector<double>> real_; 
  std::array<double, 3> x_values_;

  /* 
   * From recurrence relation: 
   *   J_{n+1}(x) = \frac{2 * n}{x} * J_n(x) - J_{n-1}(x)
   * Or, if backwards: 
   *   J_{n-1}(x) = \frac{2 * n}{x} * J_n(x) - J_{n+1}(x)
   */
  void
  compute_recurrence_() 
  {
    computed.clear(); 
    error.clear();         // set first two to zero ? 
    double C = 0.0, next = 0.0, x= 0.0;  

    computed.resize(3);
    for (auto& arr : computed) {
      arr.resize(N_);
    }

    for (int i(0); i < 3; i++) {
      x = x_values_[i];
      if ( forward_ ) {
        computed[i][0] = initial_conditions_[i].first; 
        computed[i][1] = initial_conditions_[i].second; 

        for (uint32_t j(1); j < N_ - 1; j++) {
          C = (2.0 * j) / x; 
          next = C * computed[i][j] - computed[i][j - 1];
          computed[i][j + 1] = next;  
        }

      } else { // from iter downto 0 
        computed[i][N_ - 2] = initial_conditions_[i].first; 
        computed[i][N_ - 1] = initial_conditions_[i].second; 

        for (uint32_t j(N_ - 2); j > 0; j--) {
          C = (2.0 * j) / x; 
          next = C * computed[i][j] - computed[i][j + 1];
          computed[i][j - 1] = next;  
        }
      }
    }
  }

  void 
  compute_error_()
  {
    error.clear();
    error.resize(3);
    for (auto& arr : error) {
      arr.resize(N_);
    }

    for (int i(0); i < 3; i++) {
      for (uint32_t j(0); j < N_; j++) {
        error[i][j] = computed[i][j] - real_[i][j];
      }
    }
  }
};


int main(int argc, char* argv[]) 
{
  std::pair<double, double> ic(0.0, 0.0);
  std::array<std::pair<double, double>, 3> ics;
  double x = 0.0; 
  int forward = 0; 
  std::array<double, 3> x_values;

  if ( argc != 5) {
    std::cerr << "Expects 3 x values and forward (0,1)\n"; 
    return 1; 
  }

  for (int i(1); i < 4; i++) {
    if ( sscanf(argv[i], "%lf", &x) != 1 ) {
      std::cerr << "Choked converting argument to x value\n";
      return 2; 
    } else {
      x_values[i - 1] = x;
    }
  }

  if ( sscanf(argv[4], "%d", &forward) != 1 ) {
    std::cerr << "Expected T/F for forward\n";
    return 3; 
  }

  // expect initial condition pair off stdin
  for (int i(0); i < 3; i++) {
    if ( fscanf(stdin, "%lf %lf", &ic.first, &ic.second) != 2 ) {
      std::cerr << "Expects initial conditions off stdin\n";
      return 4;
    } else {
      ics[i] = ic;
    }
  }

  Bessel bessel = Bessel(x_values, ics, 51, forward);  
  bessel.run();

  for (auto& x : x_values) {
    std::cout << x << ' ';
  }
  fputc('\n', stdout);

  for (int i(0); i < 3; i++) {
    for (auto& val : bessel.computed[i]) {
      std::cout << val << '\n';
    }

    for (auto& err : bessel.error[i]) {
      std::cout << err << '\n';
    }
  }


  return 0; 
}
