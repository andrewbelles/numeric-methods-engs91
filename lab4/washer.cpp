/*
 * washer.cpp  Andrew Belles  Oct 14th, 2025 
 *
 * Solution to washer problem on lab 4 
 *
 *
 */ 

#include <cmath> 
#include <vector> 
#include <lapacke.h>
#include <gplot++.h>

typedef double (*f)(const double*, const double*, const double&);
typedef void (*Jacobian)(const double*, double*, const double*);

constexpr size_t MAXITER = 500; 
constexpr double TOL = 1e-9;
constexpr double pi = M_PI;
constexpr double offset = (149.0 * pi) / 180.0; 
constexpr double stepsize = (1.0 * pi) / 180.0;

double recontinuous(double x0, double x1);
inline double wrap(double angle);
double linkage_one(const double r[4], const double T[2], const double& t4);
double linkage_two(const double r[4], const double T[2], const double& t4);
void linkage_jacobian(const double r[4], double J[4], const double T[2]);
double error(const double r[4], const std::vector<f>& funcs, 
             const double T[2], const double t4);
void update(const double J[4], double F[2], double X[2]);
std::vector<double> newton_system(const double r[4], Jacobian junc, 
                                  const std::vector<f>& funcs, const double X0[2], 
                                  const double& t4);
std::vector<double> forward_difference(const std::vector<double>& angles, double h);
std::vector<double> centered_difference(const std::vector<double>& angles, double h);
std::vector<double> second_centered(const std::vector<double>& angles, double h);
std::vector<double> second_forward(const std::vector<double>& angles, double h);

int main(void) 
{
  int i = 0; 
  std::vector<std::vector<double>> first(361);
  std::vector<std::vector<double>> second(361); 

  std::vector<double> theta(361); 
  std::vector<double> phi(361); 
  std::vector<double> alpha(361);
  std::vector<double> beta(361); 

  std::vector<f> funcs(2);
  double X[2] = {0.0, 1.5 * pi}, t = 0.0;

  funcs[0] = linkage_one;
  funcs[1] = linkage_two;

  // compute first linkage 
  double r1[4] = {7.1, 2.36, 6.68, 1.94};
  for (i = 1; i <= 360; i++) {

    t = (static_cast<double>(i) * stepsize) + pi;
    first[i] = newton_system(r1, linkage_jacobian, funcs, X, t);
    X[0] = first[i][0];
    X[1] = first[i][1];

    theta[i] = t - pi;
    phi[i] = first[i][0];
    alpha[i] = phi[i] + offset;
  }

  X[0] = first.back()[0];
  X[1] = first.back()[1];
  first[0] = newton_system(r1, linkage_jacobian, funcs, X, pi);
  phi[0] = first[0][0];
  alpha[0] = first[0][0] + offset;
  theta[0] = 0.0;

  // solve for second system 
  double r2[4] = {1.23, 1.26, 1.82, 2.35}; 
  X[0] = 0.0; 
  X[1] = 1.5 * pi;
  for (i = 0; i < 361; i++) {
    auto a = alpha[i];
    auto v = newton_system(r2, linkage_jacobian, funcs, X, a + pi);
    second[i] = v;

    // update initial guess; 
    X[0] = v[0];
    X[1] = v[1];

    beta[i] = v[0];
  }

  // forward differences 
  auto delta_phi_forward   = forward_difference(phi, stepsize);
  auto delta_alpha_forward = forward_difference(alpha, stepsize); 
  auto delta_beta_forward  = forward_difference(beta, stepsize);

  // centered differences
  auto delta_phi_center   = centered_difference(phi, stepsize);
  auto delta_alpha_center = centered_difference(alpha, stepsize); 
  auto delta_beta_center  = centered_difference(beta, stepsize);
  auto d2beta_f = forward_difference(delta_beta_center, stepsize);
  auto d2beta_c = centered_difference(delta_beta_center, stepsize);

  std::vector<double> c_beta_dt(361), c_beta_d2t(361), f_beta_dt(361), f_beta_d2t(361);
  for (i = 0; i < 361; i++) {
    c_beta_dt[i] = (550.0 / 60.0) * delta_beta_center[i];
    c_beta_d2t[i] = (550.0 / 60.0) * (550.0 / 60.0) * d2beta_c[i];
    f_beta_dt[i] = (550.0 / 60.0) * delta_beta_forward[i];
    f_beta_d2t[i] = (550.0 / 60.0) * (550.0 / 60.0) * d2beta_f[i];
  }

  // wrap values within half-open interval [0, 2pi)
  for (i = 0; i < 361; i++) {
    phi[i] = wrap(phi[i]);
    alpha[i] = wrap(alpha[i]);
    beta[i] = wrap(beta[i]);
  }

  std::vector<double> dt_diff(361), d2t_diff(361); 
  std::vector<double> phi_diff(361);
  for (i = 0; i < 361; i++) {
    phi_diff[i] = std::abs(delta_phi_center[i] - delta_phi_forward[i]);
    dt_diff[i] = std::abs(c_beta_dt[i] - f_beta_dt[i]); 
    d2t_diff[i] = std::abs(c_beta_d2t[i] - f_beta_d2t[i]); 
  }

  // scoped plots
  {
    Gnuplot plt{};
    plt.redirect_to_png("washer_angles.png", "1200,800");
    plt.set_title("Washer Angles: Phi, Alpha, and Beta");
    plt.set_xlabel("Theta [rads]");
    
    // plot angles 
    plt.set_title("Angles: phi, alpha, beta");
    plt.set_ylabel("Angle [rads]");
    plt.set_xrange(0.0, 2.0 * pi);
    plt.plot(theta, phi, "phi", Gnuplot::LineStyle::LINES);
    plt.plot(theta, alpha, "alpha", Gnuplot::LineStyle::LINES);
    plt.plot(theta, beta, "beta", Gnuplot::LineStyle::LINES);
    plt.show(); 
  } 

  {
    Gnuplot plt{};
    plt.redirect_to_png("washer_derivatives.png", "1200,800");
    plt.set_title("Phi Derivatives (Forward and Centered)");
    plt.set_xlabel("Theta [rads]");
    plt.set_ylabel("Change in Angle");
    plt.set_xrange(0.0, 2.0 * pi);
    plt.plot(theta, delta_phi_forward, "forward", Gnuplot::LineStyle::LINES);
    plt.plot(theta, delta_phi_center, "centered", Gnuplot::LineStyle::LINES);
    plt.show();
  }

  {
    Gnuplot plt{};
    plt.redirect_to_png("washer_angular.png", "1200,800");
    plt.multiplot(1,2, "Beta Angular Velocity and Acceleration");

    // plot beta's derivatives/kinematics 
    plt.set_xrange(0.0, 2.0 * pi);
    plt.set_xlabel("Theta [rads]");
    plt.set_ylabel("Angular Velocity [rads/sec]");
    plt.plot(theta, f_beta_dt, "forward", Gnuplot::LineStyle::LINES);
    plt.plot(theta, c_beta_dt, "centered", Gnuplot::LineStyle::LINES);
    plt.show(); 

    plt.set_xrange(0.0, 2.0 * pi);
    plt.set_ylabel("Angular Acceleration [rads/sec^2]");
    plt.plot(theta, f_beta_d2t, "forward", Gnuplot::LineStyle::LINES);
    plt.plot(theta, c_beta_d2t, "centered", Gnuplot::LineStyle::LINES);
    plt.show(); 
  }

  {
    Gnuplot plt{};

    plt.redirect_to_png("washer_phi_differences.png", "1200,800");
    plt.set_logscale(Gnuplot::AxisScale::LOGY);

    plt.set_xrange(0.0, 2.0 * pi); 
    plt.set_xlabel("Theta [rads]");
    plt.set_ylabel("First Derivative of Phi Difference [log]");
    plt.plot(theta, phi_diff, "diff", Gnuplot::LineStyle::LINES);
    plt.show(); 
  }

  // differnece plots 
  {
    Gnuplot plt{};
    plt.redirect_to_png("washer_beta_differences.png", "1200,800");
    plt.multiplot(1,2, "Differences in forward and centered approximations");
    plt.set_logscale(Gnuplot::AxisScale::LOGY);

    plt.set_xrange(0.0, 2.0 * pi); 
    plt.set_xlabel("Theta [rads]");

    plt.set_ylabel("First Derivative of Beta Difference [log]"); 
    plt.plot(theta, dt_diff, "diff", Gnuplot::LineStyle::LINES);    
    plt.show(); 

    plt.set_ylabel("Second Derivative of Beta Difference [log]"); 
    plt.plot(theta, d2t_diff, "diff", Gnuplot::LineStyle::LINES);    
    plt.show(); 
  }

  return 0; 
}

// helper function to renormalize angles back to same [0, 2pi) to avoid discts vals
double 
recontinuous(double x0, double x1)
{
  double diff = x1 - x0; 
  if ( diff > pi ) {
    x1 -= 2.0 * pi; 
  } else if ( diff < -pi ) {
    x1 += 2.0 * pi; 
  }
  return x1; 
}

inline double 
wrap(double angle) 
{
  angle = std::fmod(angle, 2.0 * pi);
  if ( angle < 0 ) {
    angle += 2.0 * pi; 
  }
  return angle; 
}

/************ functions to optimize by newtons *************/

double 
linkage_one(const double r[4], const double T[2], const double& t4)
{
  return r[1] * std::cos(T[0]) + r[2] * std::cos(T[1]) + 
         r[3] * std::cos(t4) - r[0];
}

double 
linkage_two(const double r[4], const double T[2], const double& t4)
{
  return r[1] * std::sin(T[0]) + r[2] * std::sin(T[1]) + 
         r[3] * std::sin(t4);
}

// raw ptr to 2x2 (flattened) array of jacobian 
void 
linkage_jacobian(const double r[4], double J[4], const double T[2])
{
  J[0] = -r[1] * std::sin(T[0]); 
  J[1] = -r[2] * std::sin(T[1]);
  J[2] = r[1] * std::cos(T[0]);
  J[3] = r[2] * std::cos(T[1]);
}

double 
error(const double r[4], const std::vector<f>& funcs, 
      const double T[2], const double t4)
{
  size_t i = 0, n = funcs.size();  
  double v = 0.0, sum = 0.0; 
  for (auto& fn : funcs) {
    v = fn(r, T, t4); 
    sum += v * v;
  }
  return std::sqrt(sum);
} 

void 
update(const double J[4], double F[2], double X[2])
{
  X[0] = X[0] - J[0] * F[0] + J[1] * F[1];
  X[1] = X[1] - J[2] * F[0] + J[3] * F[1]; 
}

/************ newton's method *****************************/ 
/*
 * Defines a generic newton's method for a nonlinear system of linkage equations 
 * that takes some t0 (theta 4) and initial guesses \vec{x0} (theta 2 and 3) and 
 * computes the exact roots of the linkage system 
 *
 * Caller Provides: 
 *   x0: initial guesses for roots 
 *   t0: set value of theta 4 
 *   junc: function for jacobian
 *   funcs: vector of linkage equations
 */ 
std::vector<double> 
newton_system(const double r[4], Jacobian junc, const std::vector<f>& funcs, 
              const double X0[2], const double& t4)
{
  double J[4] = {0.0}, eps = error(r, funcs, X0, t4);
  double X[2] = {X0[0], X0[1]}, F[2] = {0.0}; 
  std::vector<double> x(2); 
  size_t iter = 0, i = 0, n = funcs.size(); 

  while ( iter < MAXITER && eps > TOL ) {

    // evaluate for iteration 
    junc(r, J, X);
    for (i = 0; i < n; i++) {
      F[i] = funcs[i](r, X, t4);
    }

    // compute offset to next term 
    int ipiv[2]; 
    double dX[2] = { -F[0], -F[1] };
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2, 1, J, 2, ipiv, dX, 1);
    if ( info != 0 ) {
      std::cerr << "singular point\n";
      exit( 99 ); 
    }

    X[0] += dX[0];
    X[1] += dX[1];

    eps = error(r, funcs, X, t4);
    iter++; 
  }

  x[0] = X[0];
  x[1] = X[1];
  return x; 
}

std::vector<double> 
forward_difference(const std::vector<double>& angles, double h)
{
  size_t i = 0, n = angles.size(); 
  std::vector<double> derivative(n);

  // compute all forward difference but right boundary 
  for (i = 0; i < n - 1; i++) {
    derivative[i] = (angles[i + 1] - angles[i]) / h;  
  }

  derivative[n - 1] = (-angles[n - 2] + angles[n - 1]) / h; 

  return derivative;
}

std::vector<double> 
centered_difference(const std::vector<double>& angles, double h)
{
  size_t i = 0, n = angles.size(); 
  std::vector<double> derivative(n);

  // use more precise forward and backward differences for left and right bounds 
  derivative[0] = (-3.0*angles[0] + 4.0*angles[1] - angles[2]) / (2.0*h);
  derivative[n-1] = (angles[n-3] - 4.0*angles[n-2] + 3.0*angles[n-1]) / (2.0*h); 

  // compute all centered differences except at boundaries 
  for (i = 1; i < n - 1; i++) {
    derivative[i] = (-angles[i - 1] + angles[i + 1]) / (2.0 * h);
  }
  return derivative;
}

std::vector<double> 
second_centered(const std::vector<double>& angles, double h)
{
  size_t i = 0, n = angles.size(); 
  std::vector<double> derivative(n);

  derivative[0] = (2.0*angles[0] - 5.0*angles[1] + 4.0*angles[2] - angles[3]);
  derivative[0] /= (h * h);

  derivative[n-1] = (2.0*angles[n-1] - 5.0*angles[n-2] + 4.0*angles[n-3] - angles[n-4]);
  derivative[n-1] /= (h * h);

  for (i = 1; i < n - 1; i++) {
    derivative[i] = (angles[i - 1] - 2.0 * angles[i] + angles[i + 1]) / (h * h);
  }
  return derivative; 
}

std::vector<double> 
second_forward(const std::vector<double>& angles, double h)
{
  size_t i = 0, n = angles.size(); 
  std::vector<double> derivative(n);

  // backwards at right bound
  derivative[n-1] = (angles[n-1] - 2.0*angles[n-2] + angles[n-3]) / (h * h);
  derivative[n-2] = (angles[n-2] - 2.0*angles[n-3] + angles[n-4]) / (h * h);
  
  for (i = 0; i < n - 2; i++) {
    derivative[i] = (angles[i] - 2.0*angles[i+1] + angles[i+2]) / (h * h); 
  }
  return derivative; 
}
