/* 
 * adaptive_multistep.cpp  Andrew Belles  Oct 11th, 2025 
 *
 * Testing idea for adaptive timestep on multistep methods 
 * that ignores recomputating rate functions 
 *
 */ 

#include <functional> 
#include <string>
#include <vector> 
#include <cstdlib> 
#include <cmath> 
#include <algorithm> 
#include <gplot++.h> 

/*
 * Implementation of adaptive time step for multi-step method 
 * using 3rd and 4th order differences 
 *
 * Implements Adams-Bashforth Three and Four-Step Methods 
 *
 */ 
class MultiOde34 {
public: 
  const std::string tag; 
  
  MultiOde34(
      const std::string& tag_str,  
      const std::function<double(double)>& fn, 
      const std::vector<double>& t0, 
      const std::vector<double>& y0, 
      const double& h = 1e-4,
      const bool& adaptive = false
  ) : tag(tag_str), rate_(fn), h_(h), adapt_(adaptive) 
  {
    w_.reserve(4); 
    t_.reserve(4); 
    f_.reserve(4); 
    q_.reserve(4);

    if ( y0.size() != 4 ) {
      exit( 99 ); 
    }

    if ( t0.size() != 2 ) {
      exit( 99 ); 
    }

    // assume that first four values were computed with hceil 
    for (int i = 0; i < 4; i++) {
      double ti = t0[0] + static_cast<double>(i) * h_; 
      w_.push_back(y0[i]); 
      t_.push_back(ti);
      f_.push_back(rate_(y0[i]));
      q_.push_back(h_);
    }
    tf_ = t0.back();
  }

  /*
   * Computes the A-B fourth order approximation of the ODE using 
   * the given rate function 
   */ 
  void run() 
  {
    double ti = t_.back();

    while ( ti < tf_ ) {
      // Slice fi through fi-3 
      const std::vector<double> fs(f_.end() - 4, f_.end()); 
      // Get appropriate timestep 
      const double qh = next_q_(ti, fs); 
      ti += qh; 
      
      // Compute next predicted value, push new values into vectors, get next rate func
      const double N = 55.0 * fs[3] - 59.0 * fs[2] + 37.0 * fs[1] - 9.0 * fs[0]; 
      w_.push_back(w_.back() + (qh / 24.0 * N)); 
      t_.push_back(ti);
      q_.push_back(qh);
      f_.push_back(rate_(w_.back()));
    }
  }

  /*
   * Plots the computed solution against an exact solution on the same time interval
   */ 
  void plot(const std::string& title, const std::vector<double>& y)
  {
    int i = 0; 
    std::vector<double> error(t_.size()); 
    std::vector<double> n(t_.size()); 
    for (i = 0; i < error.size(); i++) {
      error[i] = std::abs(w_[i] - y[i]);
      n[i] = i + 1; 
    }

    {
      Gnuplot plt{}; 

      plt.redirect_to_png(tag + "_computed_vs_exact.png", "1200,700");
      plt.set_title(title);
      plt.set_xlabel("t");
      plt.set_ylabel("y");
      plt.plot(t_, w_, "A-B", Gnuplot::LineStyle::LINES);
      plt.plot(t_, y, "exact", Gnuplot::LineStyle::LINES); 
      plt.show();
    }

    {
      Gnuplot plt{}; 

      plt.redirect_to_png(tag + "_error.png", "1200,700");
      plt.set_title("Error Plot");
      plt.set_xlabel("t");
      plt.set_ylabel("|w - y|");
      plt.set_logscale(Gnuplot::AxisScale::LOGY);
      plt.plot(t_, error, "error", Gnuplot::LineStyle::LINES); 
      plt.show(); 
    }

    {
      Gnuplot plt{}; 

      plt.redirect_to_png(tag + "_qh_over_time.png", "1200,700");
      plt.set_title("qh value over time"); 
      plt.set_xlabel("iter");
      plt.set_ylabel("qh"); 
      plt.set_xrange(1, n.back());
      plt.set_logscale(Gnuplot::AxisScale::LOGY); 
      plt.plot(n, q_, "qh", Gnuplot::LineStyle::LINES);
      plt.show(); 
    }
  }

  const std::vector<double>& w() { return w_; }
  const std::vector<double>& t() { return t_; }
  const std::vector<double>& q() { return q_; }

private: 
  int lock{4};
  std::vector<double> t_{}; // time vector  
  std::vector<double> w_{}; // our solution
  std::vector<double> f_{}; // rate function at i 
  std::vector<double> q_{}; // tracked adaptive q value per frame 
  bool adapt_{false};       // whether to modify h per step  
  const std::function<double(double)> rate_; // some rate function to solve  
  double tf_{0.0};           // final time value  
  double h_{0.0};

/************ private methods ******************************/ 
  /*
   * Computes the next value q to adapt timestep  
   *
   */ 
  inline double next_q_(const double& ti, const std::vector<double>& fs) 
  {
    if ( !adapt_ ) {
      return h_; 
    }

    const double qhprev = q_.back(); 
    const double order_difference = -a * fs[3] + b * fs[2] - b * fs[1] + a * fs[0];  
    double qh = tol * h_ / std::abs(order_difference);   
    qh = std::min(qh, hceil); // clamp to hceil 

    // if we adjusted recently, void the adjustment but decrement the lock  
    if ( lock > 0 ) {
      lock--; 
      qh = qhprev; 
    } else {
      lock = 4;   
    }

    // necessarily ensure the last timestep takes us to tfinal 
    if ( ti + qh > tf_ ) {
      qh = tf_ - ti; // use an even more conservative timestep to complete interval 
    } 

    return qh; 
  }

/************ class constants ******************************/ 
  static constexpr double tol   = 1e-9; // tolerance to consider to 
  static constexpr double hceil = 1e-2; 
  static constexpr double a = 9.0 / 24.0, b = 27.0 / 24.0; 
}; 

constexpr double k = 25.0; 
constexpr double r = 4.0; 
constexpr double C = 1.0; 

static double easy_exact(const double& t);
static double easy_rate(const double& y); 
static double hard_exact(const double& t);
static double hard_rate(const double& y); 

int main(void)
{
  const std::vector<double> t0 = {0.0, 2.5}; 
  const std::vector<double> ey0 = {
    easy_exact(0), easy_exact(1e-4), easy_exact(2.0 * 1e-4), easy_exact(3.0 * 1e-4)
  };

  const std::vector<double> hy0 = {
    hard_exact(0), hard_exact(1e-4), hard_exact(2.0 * 1e-4), hard_exact(3.0 * 1e-4)
  };

  std::vector<double> ey; 
  std::vector<double> hy; 

  MultiOde34 easy_solver("exp", easy_rate, t0, ey0, 1e-4, true);
  easy_solver.run(); 

  auto& et = easy_solver.t(); 
  ey.reserve(et.size());
  for (auto& ti : et) {
    ey.push_back(easy_exact(ti));
  }

  easy_solver.plot("Adam-Bashforth 4-Step with Adaptive Timestep", ey); 

  MultiOde34 hard_solver("logistic", hard_rate, t0, hy0, 1e-4, true);
  hard_solver.run(); 

  auto& ht = hard_solver.t(); 
  hy.reserve(ht.size());
  for (auto& ti : ht) {
    hy.push_back(hard_exact(ti));
  }

  hard_solver.plot("Adam-Bashforth 4-Step with Adaptive Timestep", hy); 

  return 0; 
}

static double easy_exact(const double& t)
{
  return C * std::exp(-k * t);
}

static double easy_rate(const double& y)
{
  return -k * y; 
}

static double hard_exact(const double& t) 
{
  const double A = (k - C) / C; 
  return k / (1.0 + A * std::exp(-r * t));
}

static double hard_rate(const double& y)
{
  return r * y * (1.0 - y / k);
}
