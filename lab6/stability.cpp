/*
 * stability.cpp  Andrew Belles  Nov 6th, 2025 
 *
 * A-B/A-M two-step predictor corrector model to visualize stability range for
 * the given prototype problem. 
 *
 */

#include <cmath>
#include <vector>
#include <functional>
#include <gplot++.h>

template<typename R> 
using Interval = std::pair<R, R>; 

template<typename R> 
using Rate = std::function<R(const R&, const R&)>; 

template<typename R> 
class ABAM {
public: 
  using Data = std::pair<std::vector<R>, std::vector<R>>; 

  // A-B/A-M two-step constructor 
  ABAM(const R a, const R h, Interval<R> ic, Interval<R> time, Rate<R> fn) 
    : a_(a), h_(h), rate_func_(std::move(fn))
  {
    auto [t0, tf] = time;
    auto [y0, y1] = ic;  
    t0_ = t0; 
    tf_ = tf; 
    size_t n = static_cast<size_t>(std::floor((tf_ - t0_) / h)); 
    
    // reserve space for all 
    w.reserve(n + 1); t.reserve(n + 1); f.reserve(n + 1);
    
    // create time vector 
    for (size_t i{0}; i <= n; i++) {
      t.push_back(t0 + i * h);
    }

    // push back initial conditions 
    w.push_back(y0); w.push_back(y1);
    f.push_back(rate_func_(a_, y0)); 
    f.push_back(rate_func_(a_, y1)); 
  }

  void run() 
  {
    // start from cleared arrays 
    f.erase(f.begin() + 2, f.end());
    w.erase(w.begin() + 2, w.end());

    for (auto ti = t.begin() + 2; ti != t.end(); ti++) {
      auto wpred = ab_();                 // get predicted w value  
      auto fpred = rate_func_(a_, wpred); // get predicted rate function 
      auto wcorr = am_(fpred);            // get corrected w value  
      auto fcorr = rate_func_(a_, wcorr); // update rate function at w_{i+1}
      
      // save new values 
      f.push_back(fcorr); 
      w.push_back(wcorr); 
    } 
  }

  // Return Copy of Data or Time arrays
  std::vector<R> data() const { return w; }
  std::vector<R> time() const { return t; } 
  Data both() const { return {w, t}; }

private: 
  std::vector<R> w;
  std::vector<R> f; 
  std::vector<R> t; 

  R a_{0.0}, h_{0.0};
  R t0_{0.0}, tf_{0.0}; 
  Rate<R> rate_func_{}; 

  inline R ab_() const 
  {
    // deref back two rate function evaluations 
    auto f1 = *f.rbegin(); 
    auto f2 = *(f.rbegin() + 1);  
    return w.back() + 0.5 * h_ * (3.0 * f1 - f2);
  }

  inline R am_(const R& f3) const 
  {
    auto f1 = *f.rbegin(); 
    auto f2 = *(f.rbegin() + 1);  
    return w.back() + (h_ / 12.0) * (5.0 * f3 + 8.0 * f1 - f2);
  }
}; 

template <typename R> 
void plot(const std::vector<R>& t, const std::vector<R>& w, 
          const std::string& title, const std::string& label, const std::string& png)
{
  if ( t.size() != w.size() ) {
    return; 
  }

  Gnuplot plt{}; 
  plt.redirect_to_png(png, "1200,1000"); 
  plt.set_title(title);
  plt.set_xlabel("t");
  plt.set_ylabel("w");
  plt.plot(t, w, label, Gnuplot::LineStyle::LINES);
  plt.show();
}

double rate(const double& a, const double& w);

int main(void)
{
  double a{1.0};
  auto e = [&a](const double& y) -> double {
    return 50.0 * std::exp(-a * y);
  };

  // Certainly stable 
  Interval<double> ic_stable{e(0.0), e(1e-3)}, t{0.0, 100.0};
  auto stable   = ABAM<double>(a, 1e-3, ic_stable, t, rate); 
  stable.run(); 
  
  // Very close to being unstable 
  Interval<double> ic_boundary{e(0.0), e(2.5)};
  auto semi     = ABAM<double>(a, 1.0, ic_boundary, t, rate); 
  semi.run(); 

  Interval<double> ic_unstable{e(0.0), e(5.0)};
  auto unstable = ABAM<double>(a, 5.0, ic_unstable, t, rate); 
  unstable.run();

  // copy results 
  auto [w_stable, t_stable]     = stable.both();
  auto [w_semi, t_semi]         = semi.both(); 
  auto [w_unstable, t_unstable] = unstable.both(); 

  plot<double>(t_stable, w_stable,
               "Stable solution, h = 1e-3", "stable", "stable.png");
  plot<double>(t_semi, w_semi,
               "Semi-stable solution, h = 1.0", "semi-stable", "semistable.png");
  plot<double>(t_unstable, w_unstable,
               "Unstable solution, h = 5.0", "unstable", "unstable.png");

  return 0; 
}

inline double rate(const double& a, const double& w)
{
  return -a * w; 
}
