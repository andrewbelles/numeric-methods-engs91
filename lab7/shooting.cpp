/*
 * shooting.cpp  Andrew Belles  Nov 9th, 2025 
 *
 * 4th Order A-B/A-M Predictor Corrector, Newton's aided Shooting Method 
 * numeric solution to ODE describing deflection of beam
 *
 */ 

#include <algorithm>
#include <iostream> 
#include <cmath> 
#include <string>
#include <vector>
#include <format> 
#include <gplot++.h> 

constexpr double EPS{1e-9};
constexpr size_t MAXITER{1000};

struct State {
  double y, yprime; 

};

inline State operator+(const State& a, const State& b) 
{
  return {a.y + b.y, a.yprime + b.yprime};
}

inline State operator-(const State& a, const State& b)
{
  return {a.y - b.y, a.yprime - b.yprime};
}

inline State operator*(const double& c, const State& a)
{
  return {c * a.y, c * a.yprime};
}


class Beam {
public: 
  
  Beam(const double& u, const double& alpha, const double& beta, 
       const double& h = 1e-3) 
    : h_(h)
  {
    bcs_ = {alpha, beta};
    u0_  = u; 

    // Get distance array 
    size_t n{static_cast<size_t>(std::round(L / h))};  
    // Pre-allocate 
    x_.reserve(n);
    z_.reserve(n);
    F_.reserve(n); 
    v_.reserve(n);
    G_.reserve(n);

    for (size_t i = 0; i < n; i++) {
      x_.push_back(static_cast<double>(i) * h);
    }
    
    // Push back IC, get first force  
    z_.push_back({alpha, u0_});
    F_.push_back(system_rate(z_.front(), 0.0)); 
    v_.push_back({0.0, 1.0});
    G_.push_back(newton_rate(v_.front(), z_.front(), 0.0));
  }

  double run(void)
  {
    shots_.clear(); 
    auto [alpha, beta] = bcs_; 
    size_t iter{0};
    double u = u0_; 
    double beta_est = 0.0; 
    
    do {
      predictor_corrector_(z_, F_, v_, G_);
      beta_est = z_.back().y; 

      u -= (z_.back().y - beta) / (v_.back().y);
      shots_.push_back(z_); 
      reinitialize_(u);
      iter++;

    } while ( std::abs(beta_est - beta) > EPS && iter < MAXITER ); 

    u_optimal_ = u; 
    return u; // return best trajectory  
  }

  const std::pair<std::vector<State>&, std::vector<State>&> z() 
  {
    if ( z_.size() != x_.size() ) {
      reinitialize_(u_optimal_);
      predictor_corrector_(z_, F_, v_, G_); 
    }

    return {z_, v_}; 
  }

  const std::vector<double>& x() const  
  {
    return x_; 
  }

  const std::vector<std::vector<State>>& shots() const 
  {
    return shots_; 
  }

private: 

  State bcs_;
  double u0_; // ? 
  double u_optimal_{0.0}; // store optimal angle 
  double L{50.0}, D{8.5e7}, S{100.0}, q{1000.0};
  double h_{1e-3}; 
  std::vector<State> F_{}, z_{}, v_{}, G_{};
  std::vector<double> x_{}; 
  std::vector<std::vector<State>> shots_{};

  void reinitialize_(const double u) 
  {
    z_.front().yprime = u;
    F_.front() = system_rate(z_.front(), 0.0); 
    G_.front() = newton_rate(v_.front(), z_.front(), 0.0);
  }

  State system_rate(const State& z, const double x)
  {
    auto [y, yp] = z; 

    const double a = std::pow((1.0 + yp * yp), 1.5); 
    const double b = (q * x * (x - L) * y) / (2.0 * D); 
    const double c = ((S / D) * yp);

    return {yp, a * (b + c)};
  }

  State newton_rate(const State& v, const State& z, const double x)
  {
    const auto [g, gp] = v; 
    const auto [y, yp] = z; 

    const double a = 1.0 + yp * yp; 
    const double sqrt_a = std::sqrt(a); 
    const double K = std::pow(a, 1.5); 

    const double B = q / (2.0 * D) * (x * x - x * L); 
    const double A = S / D;

    const double coef_g  = B * K; 
    const double coef_gp = A * (K + 3.0 * yp * sqrt_a) + B * 2.0 * yp * y; 

    return {gp, coef_g * g + coef_gp * gp};
  }

  State bashforth_(const State& z, const std::vector<State>& F)
  {
    auto it = F.end(); 
    const State f[4] = {*(it - 4), *(it - 3), *(it - 2), *(it - 1)};

    const State update = 55.0 * f[3] - 59.0 * f[2] + 37.0 * f[1] - 9.0 * f[0];  
    return z + (h_ / 24.0) * update; 
  }

  State moulton_(const State& z, const std::vector<State>& F, const State& fpred)
  {
    auto it = F.end(); 
    const State f[4] = {*(it - 4), *(it - 3), *(it - 2), *(it - 1)};

    const State a = 251.0 * fpred + 646.0 * f[3] - 264.0 * f[2]; 
    const State b = 106.0 * f[1] - 19.0 * f[0]; 
    return z + (h_ / 720.0) * (a + b);
  }

  /*
   * We write to take z and f as arguments, 
   *
   */
  void predictor_corrector_(std::vector<State>& z, std::vector<State>& F, 
                            std::vector<State>& v, std::vector<State>& G)
  {
    // Reset back to 4 values (or generate) for start  
    if ( z.size() != 4 ) {
      z.erase(z.begin() + 1, z.end());
      F.erase(F.begin() + 1, F.end());
      v.erase(v.begin() + 1, v.end());
      G.erase(G.begin() + 1, G.end());
      rk4_(z, F, v, G); // pre-load now 
    }

    // Run A-B/A-M pred/corr starting from 4th element 
    for (auto x_it = x_.begin() + 4; x_it != x_.end(); x_it++) {
      auto x = *x_it; 
      const State zpred = bashforth_(z.back(), F);
      const State fpred = system_rate(zpred, x);
      const State vpred = bashforth_(v.back(), G); 
      const State gpred = newton_rate(vpred, zpred, x);

      const State zcorr = moulton_(z.back(), F, fpred); 
      const State fcorr = system_rate(zcorr, x);  
      const State vcorr = moulton_(v.back(), G, gpred); 
      const State gcorr = newton_rate(vcorr, zcorr, x);

      z.push_back(zcorr); 
      F.push_back(fcorr); 
      v.push_back(vcorr); 
      G.push_back(gcorr);
    }
  }

  /*
   * Computes 3 iterations of Runge-Kutta 4th order to pre-load
   * the predictor corrector for both g and y 
   */ 
  void rk4_(std::vector<State>& z, std::vector<State>& F, 
            std::vector<State>& v, std::vector<State>& G)
  {
    for (size_t i = 0; i < 3; i++) {
      auto x  = x_[i];

      auto z1 = z.back();
      auto v1 = v.back();
      const State kz1 = h_ * system_rate(z1, x); 
      const State kv1 = h_ * newton_rate(v1, z1, x);

      auto z2 = z1 + 0.5 * kz1;  
      auto v2 = v1 + 0.5 * kv1;  
      const State kz2 = h_ * system_rate(z2, x + (0.5 * h_)); 
      const State kv2 = h_ * newton_rate(v2, z2, x + (0.5 * h_));
       
      auto z3 = z1 + 0.5 * kz2; 
      auto v3 = v1 + 0.5 * kv2; 
      const State kz3 = h_ * system_rate(z3, x + (0.5 * h_)); 
      const State kv3 = h_ * newton_rate(v3, z3, x + (0.5 * h_));

      auto z4 = z1 + kz3; 
      auto v4 = v1 + kv3; 
      const State kz4 = h_ * system_rate(z4, x + h_);
      const State kv4 = h_ * newton_rate(v4, z4, x + h_);

      z.push_back(z1 + (1.0 / 6.0) * (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4)); 
      F.push_back(system_rate(z.back(), (x + h_)));  
      v.push_back(v1 + (1.0 / 6.0) * (kv1 + 2.0 * kv2 + 2.0 * kv3 + kv4)); 
      G.push_back(newton_rate(v.back(), z.back(), (x + h_)));
    }
  }
}; 

int main(void)
{
  const double L = 50.0; 
  const double alpha = 0.0, beta = 0.0; 

  // Preturb initial guess because y' = 0 is a trivial answer  
  Beam sol(0.25, alpha, beta, 1e-3);

  auto ustar = sol.run(); 
  std::cout << ustar << '\n'; 

  auto& [z, v] = sol.z();
  const auto& x = sol.x(); 
  const double bu0 = z[0].yprime;
  std::vector<double> y(x.size()), yp(x.size()); 
  for (size_t i = 0; i < x.size(); i++) {
    y[i]  = z[i].y;
    yp[i] = z[i].yprime; 
  }

  {
    Gnuplot plt{}; 

    auto title = std::format("u0={:.4e}", bu0);
    plt.redirect_to_png("deflection.png", "1200,1000");
    plt.set_title("Beam Deflection using Newton's Shooting Method. Best: " + title); 
    plt.set_xlabel("x [dx=1e-3]");
    plt.set_ylabel("y & y' [m & dy/dx]");

    plt.plot(x, y, "y(x)", Gnuplot::LineStyle::LINES);
    plt.plot(x, yp, "y'(x)", Gnuplot::LineStyle::LINES);
    plt.show();
  }

  const auto& shots = sol.shots();  
  // Still using sol.x() 

  {
    Gnuplot plt{}; 

    auto title = std::format("u0={:.4e}", bu0);
    plt.redirect_to_png("traj_error.png", "1200,1000");
    plt.set_title("Global Error of Each Trajectory from best: " + title);
    plt.set_xlabel("x [dx=1e-3]"); 
    plt.set_ylabel("y [m]"); 
    plt.set_logscale(Gnuplot::AxisScale::LOGY);

    for (const auto& shot : shots) { 
      const double u0 = shot[0].yprime;
      if ( u0 == bu0 ) {
        continue; 
      } // skip reference/correct point 

      std::vector<double> y(x.size()), err(x.size()); 
      for (size_t i = 0; i < x.size() - 1; i++) {
        y[i]   = shot[i].y;
        err[i] = y[i] - z[i].y;
      }

      auto label = std::format("u0={:.4e}", u0);
      plt.plot(x, err, label, Gnuplot::LineStyle::LINES);
    }
    plt.show();
  }
  
  std::vector<double> stepsizes(16); 
  for (size_t i = 0; i < stepsizes.size() - 1; i++) {
    size_t n{stepsizes.size() - i};
    stepsizes[i] = (1 << n) * 1e-5; 
  }
  stepsizes.back() = 1e-5; 

  std::vector<double> trailing_y; trailing_y.reserve(stepsizes.size() - 1);
  std::vector<double> trailing_p; trailing_p.reserve(stepsizes.size() - 1);
  std::vector<double> inverse; inverse.reserve(stepsizes.size() - 1);  

  {
    Gnuplot plt{}; 

    plt.redirect_to_png("convergence.png", "1200,1000");
    plt.set_title("Convergence of 4th order A-B/A-M Predictor-Corrector Scheme"); 
    plt.set_xlabel("1/dx [m^-1]"); 
    plt.set_ylabel("relative error at boundary x=L");
    plt.set_logscale(Gnuplot::AxisScale::LOGXY); 

    double exact_y = 0.0, exact_yp;  
    for (const auto& dx : stepsizes) {
      auto model = Beam(0.25, alpha, beta, dx);  
      model.run(); 

      const auto [z, _] = model.z(); 
      const auto [yL, ypL] = z.back(); 

      if ( dx != stepsizes.back() ) {
        trailing_y.push_back(yL);
        trailing_p.push_back(ypL);
        inverse.push_back(1.0/dx);
      } else {
        exact_y  = yL; 
        exact_yp = ypL;  
      } 
    }

    std::transform(trailing_y.begin(), trailing_y.end(), trailing_y.begin(), 
      [&](double el) { 
        return std::abs(el - exact_y) / std::abs(exact_y); 
    }); 

    std::transform(trailing_p.begin(), trailing_p.end(), trailing_p.begin(), 
      [&](double el) { 
        return std::abs(el - exact_yp) / std::abs(exact_yp); 
    }); 

    plt.plot(inverse, trailing_y, "rel error y(x)", Gnuplot::LineStyle::LINES);
    plt.plot(inverse, trailing_p, "rel error y'(x)", Gnuplot::LineStyle::LINES);

    plt.set_xrange(inverse.front(), inverse.back());
    plt.set_yrange(std::min(trailing_y.back(), trailing_p.back()),
                   std::max(trailing_y.front(), trailing_p.front()));
    plt.show();
  } 

  return 0; 
}
