/*
 * approx.cpp  Andrew Belles  Oct 9th, 2025 
 *
 * Linear, Cubic polynomial, and Log Linear fits from least squares  
 * Additional errors plots are included. Usage: ./run.sh [args]
 *
 */ 

#include <iostream> 
#include <cmath> 
#include <fstream> 
#include <cstdint> 
#include <string_view>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include <gplot++.h>
#include <lapacke.h> 

static inline std::vector<double> linspace(double s, double e, int n);

/************ polynomial evaluation overloads *************/ 
static inline double evaluate(const std::vector<double>& coeffs, const double& x);
static inline std::vector<double> evaluate(const std::vector<double>& coeffs, 
                                           const std::vector<double>& x);
static inline std::vector<double> evaluate_loglinear(const std::vector<double>& coeffs,
                                                     const std::vector<double>& x);

class DataSet {
public: 

  enum FitType : int8_t {
    Linear, 
    Cubic,
    LogLinear,
    NonLinear, 
    All 
  };

  struct FitCurve {
    FitType type;
    std::vector<double> coeffs; 
  };
  
  DataSet(const char* path) : path_(path) {
    fptr_.open(path);  
    if ( !fptr_.is_open() ) {
      throw std::runtime_error("choked on invalid file");
    }
  }

  ~DataSet() { fptr_.close(); }

  /********** DataSet::read() *****************************/
  /* 
   * Reads data from the file pointer opened at construction of object 
   * If method is recalled, clear data and rewind ptr before reading again 
   */
  void 
  read() 
  {
    x_.clear(); 
    y_.clear(); 
    fptr_.clear(); 
    fptr_.seekg(0, std::ios::beg);

    double x = 0.0, y = 0.0; 
    // live running totals 
    double lxsq = 0.0, lxy = 0.0, lsx = 0.0, lsy = 0.0; 

    while ( fptr_ >> x >> y ) {
      x_.push_back(x);
      y_.push_back(y);
      
      lxsq += x * x; 
      lxy  += x * y; 
      lsx  += x; 
      lsy  += y; 
    }

    // save collected totals 
    xsq_ = lxsq; 
    xy_  = lxy; 
    sx_  = lsx; 
    sy_  = lsy; 
  }

  /********** DataSet::fit() ******************************/ 
  /* 
   * Wraps over private methods for computing fits 
   * Utilizes fit enum, wraps outputs with the method that was used 
   *
   */
  std::vector<FitCurve> 
  fit(FitType fit_enum)
  {
    switch (fit_enum) {
      case Linear: 
        return { FitCurve{Linear, linear_()} };
      case Cubic: 
        return { FitCurve{Cubic, cubic_()} }; 
      case LogLinear: 
        return { FitCurve{LogLinear, log_linear_()} };
      case All:
        return { 
          FitCurve{Linear, linear_()}, 
            FitCurve{Cubic, cubic_()}, 
            FitCurve{LogLinear, log_linear_()},
        };
      default: 
        return {}; 
    }
  }

  static constexpr std::string_view 
  to_string(FitType v) 
  {
    switch (v) {
      case Linear: 
        return "Linear";
      case Cubic: 
        return "Cubic";
      case LogLinear:
        return "LogLinear";
      case NonLinear: 
        return "NonLinear";
      case All:
        return "All";
    }
    return "invalid";
  }

  /********** DataSet::plot() ******************************/
  /*
   * Plots all fits provided to it, utilizes enum methods to 
   * properly label fits to their approximation method
   *
   */ 
  void 
  plot(const std::vector<FitCurve>& coeff_table, 
       std::string& png, std::string& title)
  {
    double xmin = *std::min_element(x_.begin(), x_.end()); 
    double xmax = *std::max_element(x_.begin(), x_.end()); 
    auto xarr = linspace(xmin, xmax, (xmax - xmin) * 1000.0 );
    int i = 0, m = y_.size(); 
    std::vector<std::vector<double>> ytable, err_table; 
    
    for (auto& [type, coeff] : coeff_table) {
      std::vector<double> err(m); 

      std::vector<double> yhat = (type == LogLinear || type == NonLinear)
        ? evaluate_loglinear(coeff, x_)
        : evaluate(coeff, x_);

      std::vector<double> yarr = (type == LogLinear || type == NonLinear)
        ? evaluate_loglinear(coeff, xarr)
        : evaluate(coeff, xarr);

      std::transform(
        y_.begin(), y_.end(), yhat.begin(), err.begin(),
        [](double y, double yh) {
          return (std::abs(y - yh) / std::abs(y));
        }
      );

      ytable.push_back(yarr);
      err_table.push_back(err);
    }

    {
      Gnuplot plt{};
      
      plt.redirect_to_png(png, "1200,800");

      plt.set_title(title);
      plt.set_xlabel("x");
      plt.set_ylabel("y");
      plt.set_xrange(xmin, xmax); 
      
      for (i = 0; i < coeff_table.size(); i++) {
        plt.plot(xarr, ytable[i], to_string(coeff_table[i].type).data(), 
                 Gnuplot::LineStyle::LINES);
      }
      plt.plot(x_, y_, "Data", Gnuplot::LineStyle::LINESPOINTS);
      plt.show();
    } // end plot 1 scope 

    {
      Gnuplot plt{};
      
      plt.redirect_to_png("log_"+png, "1200,800");

      plt.set_title("LogLinear and NonLinear Fits on Logscale");
      plt.set_xlabel("x");
      plt.set_ylabel("y [logscale]");
      plt.set_xrange(xmin, xmax);
      plt.set_logscale(Gnuplot::AxisScale::LOGY);
      
      for (i = 0; i < coeff_table.size(); i++) {
        // only plot loglinear approximations on logscale 
        if ( coeff_table[i].type == LogLinear || coeff_table[i].type == NonLinear ) {
          plt.plot(xarr, ytable[i], to_string(coeff_table[i].type).data(), 
                   Gnuplot::LineStyle::LINES);
        }
      }
      plt.plot(x_, y_, "Data", Gnuplot::LineStyle::LINESPOINTS);
      plt.show();
    }

    {
      Gnuplot plt{}; 
      plt.redirect_to_png("errors_" + png, "1200,800");

      plt.set_title("Log-Scale Error: " + title); 
      plt.set_xlabel("x");
      plt.set_ylabel("relative error [log-scale]");
      plt.set_logscale(Gnuplot::AxisScale::LOGY);
      plt.set_xrange(xmin, xmax); 

      for (i = 0; i < err_table.size(); i++) {
        plt.plot(x_, err_table[i], to_string(coeff_table[i].type).data(),
                 Gnuplot::LineStyle::LINES);
      }
      plt.show(); 
    } // end plot 3 scope 
  }

  // returns errors for newtons and loglinear at x[0]
  std::pair<std::vector<double>, std::vector<double>>  
  compare()
  {
    auto ab0 = log_linear_(); 
    double a = ab0[1], b = ab0[0];
    size_t k = 0, m = x_.size(); 
    double C[7]{0.0};

    auto set_sums = [&](double a, double b) {
      for (auto& c : C) {
        c = 0.0; 
      }

      size_t i = 0;
      for (i = 0; i < m; i++) {
        double c = std::exp(a * x_[i]);
        double c2 = c * c; 
        C[0] += c2; 
        C[1] += y_[i] * c; 
        C[2] += x_[i] * c2; 
        C[3] += x_[i] * y_[i] * c; 
        C[4] += 2.0 * x_[i] * c2; 
        C[5] += 2.0 * x_[i] * x_[i] * c2; 
        C[6] += x_[i] * x_[i] * y_[i] * c; 
      }
    }; 

    // iterate starting from log linear coefficients 10 iterations to check improvement
    for (k = 0; k < 10; k++) {
      set_sums(a, b);

      // using closed for expression where b is solved for and substituted, making 
      // this a 1d newtons instead of a 2d system 
      b = C[1] / C[0];
      double F  = ( C[1] * C[2] ) / C[0] - C[3];
      double dF = ((C[3]*C[2] + C[1]*C[5] - (C[1]*C[2]/C[0])*C[4]) / C[0]) - C[6];

      a += -F / dF; 
    }

    set_sums(a, b);
    b = C[1] / C[0];

    auto yhat = evaluate_loglinear({b, a}, x_);
    auto y    = evaluate_loglinear(log_linear_(), x_);
    return {{b, a}, {std::abs(y_[0] - y[0]), std::abs(y_[0] - yhat[0])}};
  }

  // getters 
  std::vector<double> x() const { return x_; }
  std::vector<double> y() const { return y_; }
  double sum_x_sq() const { return xsq_; }
  double sum_xy() const { return xy_; }
  double sum_x() const { return sx_; }
  double sum_y() const { return sy_; }

private: 
  const char* path_; 
  std::ifstream fptr_;
  std::vector<double> x_{}; 
  std::vector<double> y_{}; 
  double xsq_{0.0}, xy_{0.0}, sx_{0.0}, sy_{0.0}; 

  /********** fit functions *******************************/
  /* wrapped over by exposed fit api */ 
  std::vector<double> 
  linear_()
  {
    std::vector<double> coeff(2, 0);  
    size_t m(x_.size());

    coeff[0] = ((xsq_ * sy_) - (xy_ * sx_)) / (m * xsq_ - sx_ * sx_);  
    coeff[1] = ((m * xy_) - (sx_ * sy_)) / (m * xsq_ - sx_ * sx_); 
    return coeff; 
  }

  std::vector<double> 
  cubic_() 
  {
    size_t m = x_.size(), i = 0, k = 0;
    std::vector<double> A(16, 0.0); 
    std::vector<double> b(4, 0.0);

    double sums[7]  = {0.0}; 
    double cross[4] = {0.0};
    
    // compute all sums into static arrays 
    for (i = 0; i < m; i++) {
      double x = x_[i], y = y_[i], sum = 1.0; 

      sums[0]  += 1.0; 
      cross[0] += y;

      // compute exponential terms and crossing terms 
      for (k = 1; k <= 6; k++) {
        sum *= x; 
        sums[k] += sum; 
        if ( k <= 3 ) {
          cross[k] += sum * y; 
        }
      }
    }

    // set value of A and cross terms  
    for (i = 0; i < 4; i++) {
      for (k = 0; k < 4; k++) {
        A[i * 4 + k] = sums[i + k]; 
      }
      b[i] = cross[i];
    }
    
    int ipiv[4]; 

    // solve 4x4, 4x1 system of linear equations 
    if ( LAPACKE_dgesv(LAPACK_ROW_MAJOR, 4, 1, A.data(), 4, ipiv, b.data(), 1) != 0 ) {
      std::cerr << "lapacke general solve failure\n";
      exit( 99 ); // constitutes major failure 
    }
    return b; // contains coefficients      
  }

  std::vector<double> 
  log_linear_()
  {
    size_t m = x_.size(), i = 0;
    std::vector<double> logy(y_.size(), 0.0);
    std::vector<double> coeff(2, 0);  
    double sumx = 0.0, sxx = 0.0, sxy = 0.0, sumy = 0.0;  

    // compute running sums, get coefficients
    for (i = 0; i < m; i++) {
      // if any value of y is negative assume malformed and throw exception 
      if ( y_[i] <= 0.0 ) {
        throw std::runtime_error("negative data point in log linear fit");
      }
      // take log of each element from y 
      double x = x_[i], logy = std::log(y_[i]);
      sumx += x; 
      sumy += logy; 
      sxx  += x * x; 
      sxy  += x * logy; 
    }

    const double den = (m * sxx) - (sumx * sumx);
    coeff[0] = std::exp((sxx * sumy - sxy * sumx) / den); 
    coeff[1] = (m * sxy - sumx * sumy) / den;  
    return coeff; 
  }
};

int main(int argc, char* argv[]) {

  if ( argc != 4 ) {
    std::cerr << "invalid usage: ./approx [lab4-data.txt] [fit enum] [fit.png]\n";
    exit( 1 ); 
  }

  DataSet::FitType ft = static_cast<DataSet::FitType>(std::stoi(argv[2]));
  DataSet ds(argv[1]);

  ds.read();

  std::vector<DataSet::FitCurve> coeff_table = ds.fit(ft);
  auto [coeff, rel] = ds.compare();
  coeff_table.push_back({DataSet::FitType::NonLinear, coeff});

  std::cout << "LogLinear Error at x[0]: " << rel[0] << '\n'
            << "Nonlinear Newton's at x[0]: " << rel[1] << '\n';

  std::string title = std::string(DataSet::to_string(ft));
  std::string png   = std::string(argv[3]);

  title += " Methods of Approximating Data";
  ds.plot(coeff_table, png, title);

  exit( 0 ); 
}

/************ plotting helper function implementations ****/
static inline std::vector<double> 
linspace(double s, double e, int n)
{
  if ( n < 2 ) {
    throw std::invalid_argument("input size must be greater than 2");
  }

  int i = 0; 
  double h = 0.0;
  std::vector<double> v(n); 

  // step size 
  h = (e - s) / (n - 1);

  for (i = 0; i < n; i++) {
    v[i] = s + i * h; 
  }
  return v; 
}

static inline double 
evaluate(const std::vector<double>& coeffs, const double& x)
{
  double y = 0.0; 
  int i = 0; 

  for (i = coeffs.size() - 1; i >= 0; i--) {
    y = y * x + coeffs[i];
  }
  
  return y; 
}

static inline std::vector<double> 
evaluate(const std::vector<double>& coeffs, const std::vector<double>& xarr)
{
  std::vector<double> yarr(xarr.size());
  int i = 0; 
  for (i = 0; i < xarr.size(); i++) {
    yarr[i] = evaluate(coeffs, xarr[i]);
  }

  return yarr;
}

static inline std::vector<double> 
evaluate_loglinear(const std::vector<double>& coeffs, const std::vector<double>& xarr)
{
  std::vector<double> yarr(xarr.size()); 
  int i = 0; 
  for (i = 0; i < xarr.size(); i++) {
    yarr[i] = coeffs[0] * std::exp(coeffs[1] * xarr[i]);
  }
  return yarr; 
}
