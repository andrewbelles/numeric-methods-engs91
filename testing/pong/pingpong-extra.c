/*
 * pingpong-extra.c  Andrew Belles  Nov 21th, 2025 
 *
 * Better version of pingpong question using AB-AM pred-corr
 * and RK4 bootstrap. Uses Newton's Shooting Method 
 *
 */ 

#include <math.h> 
#include <stddef.h>
#include <stdio.h> 
#include <stdbool.h> 
#include <stdlib.h> 

#define G 9.81 
#define ONE_DEG_RAD M_PI / 180.0 
#define RESOLUTION 89
#define SOLUTIONS 4 
#define TRAJECTORY_SIZE 1024 
#define MAXITER 256

typedef struct {
  double x;
  double y; 
} point_t; 

#define POINT(x_, y_) \
  (point_t){ .x = x_, .y = y_ }

inline point_t add(point_t a, point_t b) { return (point_t){a.x + b.x, a.y + b.y}; }
inline point_t sub(point_t a, point_t b) { return (point_t){a.x - b.x, a.y - b.y}; }
inline point_t scale(point_t a, double c) { return (point_t){c * a.x, c * a.y}; }
inline double mag(point_t a) { return sqrt(a.x * a.x + a.y * a.y); }

inline point_t 
add_list(const point_t* __restrict__ pts, size_t count) 
{
  point_t p = POINT(0.0, 0.0); 
  for (size_t i = 0; i < count; i++) {
    p.x += pts[i].x; 
    p.y += pts[i].y; 
  }
  return p; 
}

typedef struct {
  point_t* pos;     // all (x,z) positions 
  point_t vel[4];   // current and last three velocities 
  point_t force[4]; // current and last three function evaluations 
  size_t size;      // index to last inserted position 
  size_t capacity;  // total positions that could be store  
} trajectory_t; 

typedef enum hit {
  NONE=0,
  FLOOR=1, 
  STEP_WALL=2, 
  STEP_FLOOR=3,
  BACK_WALL=4
} hit_e; 

typedef enum dir {
  X=0, 
  Y=1
} dir_e;

#define NOT(dir) \
  (dir_e)(dir == X)? Y : X 

typedef struct quadInfo { 
  point_t pos; 
  point_t vel; 
  double dt_rem; 
  bool success; 
} quadInfo_t; 

typedef struct {
  double m; 
  double k; 
  double v0; 
  double ds; 
  double hs; 
  double d; 
  double dw; 
  double w; 
  double dt; 
  double eps; 
} params_t;

typedef void (*history)(point_t*, point_t); 
typedef void (*step)(const params_t* __restrict__ params, size_t offset, 
                     trajectory_t* traj, double dt, history);
typedef point_t (*pred_corr)(const params_t* __restrict__, point_t, point_t*,
                             point_t*, double);

typedef struct solver {
  pred_corr pc; 
  step single; 
  history push_history;
  size_t evals; 
} solver_t;

static void 
resize(trajectory_t* traj)
{
  if ( !traj ) {
    return; 
  }
  traj->capacity *= 1.6;  
  traj->pos = realloc(traj->pos, traj->capacity * sizeof(point_t));
}

static trajectory_t* 
new_trajectory(size_t size)
{
  if ( size == 0 ) {
    return NULL; 
  } else {
    trajectory_t* traj = (trajectory_t*)malloc(sizeof(trajectory_t)); 
    traj->capacity = size; 
    traj->size     = 0;  
    traj->pos = (point_t*)malloc(traj->capacity * sizeof(point_t));
    return traj; 
  }
}

static void 
rk4_history(point_t* history, point_t newest) 
{
  for (size_t i = 3; i > 0; --i) {
    history[i] = history[i - 1]; 
  }
  history[0] = newest; 
}

trajectory_t* sim(const params_t* __restrict__ params, const double angle, 
                  solver_t* config);
point_t rate(const params_t* __restrict__ params, point_t vel);
void bootstrap(const params_t* __restrict__ params, size_t pos_ptr, size_t step_count, 
               trajectory_t* traj, step single, const double dt, history push_history);
void rk4_step(const params_t* __restrict__ params, size_t offset, 
              trajectory_t* traj, double dt, history push_history);
point_t ab_am_four(const params_t* __restrict__ params, point_t pos, point_t* vels, 
                   point_t* forces, double dt); 
hit_e hit(const params_t* __restrict__ params, point_t a, point_t b);
quadInfo_t quadratic_interpolate(const params_t* __restrict__ params, point_t prev, 
                                 point_t curr, point_t next, point_t vprev, 
                                 point_t vcurr, point_t vnext, double dt, 
                                 dir_e axis, double boundary);


trajectory_t* 
sim(const params_t* __restrict__ params, const double angle, solver_t* config)
{
  const double dt = params->dt; // Possibly adjustment to a variable step method  

  trajectory_t* traj = new_trajectory(TRAJECTORY_SIZE);
  point_t p = POINT(0.0, 0.0);
  point_t v = POINT(params->v0 * cos(angle), params->v0 * sin(angle)); 
  bool done = false; 

  traj->pos[0]   = p;  
  traj->vel[0]   = v; 
  traj->force[0] = rate(params, v);
  bootstrap(params, 0, config->evals, traj, config->single, dt, config->push_history);
  const pred_corr pc = config->pc; 

  // Main simulation loop for single trajectory 
  while ( !done ) {

    if ( traj->size >= traj->capacity ) {
      resize(traj);
    } 

    point_t pnew = pc(params, p, traj->vel, traj->force, dt);

    double boundary = 0.0; 
    dir_e dir; 
    const hit_e status = hit(params, p, pnew);

    switch (status) {
      case FLOOR: 
      case STEP_FLOOR: 
        boundary = (status == FLOOR)? 0.0 : params->hs; 
        dir = X; 
        done = true; 
        break; 
      case STEP_WALL: 
      case BACK_WALL: 
        // velocity quadratic interpolate 
        boundary = (status == STEP_WALL)? params->ds : params->dw; 
        dir = Y; 
        break;
      case NONE: 
      default: 
        break; 
    }

    if ( status != NONE ) {
      size_t l = traj->size - 1; 
      quadInfo_t res = quadratic_interpolate(params, traj->pos[l - 1], traj->pos[l], 
                                             pnew, traj->vel[2], traj->vel[1], 
                                             traj->vel[0], dt, dir, boundary);
      if ( !res.success ) {
        fprintf(stderr, "Failed to interpolate. prior status: %d\n", status);
        exit(99);
      }

      
    }

    p = pnew; 
    traj->pos[traj->size++] = p;
  }


  return traj;
}

void 
bootstrap(const params_t* __restrict__ params, size_t pos_ptr, size_t step_count, 
          trajectory_t* traj, step single, const double dt, history push_history)
{
  if ( !params || !traj ) {
    return; 
  }

  for (int i = 1; i < step_count; i++) {
    (*single)(params, pos_ptr + i, traj, dt, push_history); 
  }
  traj->size = pos_ptr + step_count; 
}

/*
 * ar must be guaranteed 4 elements, anything else is UB 
 */
static inline point_t
bashforth(const point_t* __restrict__ ar, const double dt)
{
  point_t loc[4]; 
  loc[0] = scale(ar[0], 55.0); 
  loc[1] = scale(ar[1], -59.0); 
  loc[2] = scale(ar[2], 37.0); 
  loc[3] = scale(ar[3], -9.0); 
  return scale(add_list(loc, 4), dt / 24.0);
}

/*
 * we guarantee we only use the front 3 elements of ar but it must have 4 
 */
static inline point_t 
moulton(const point_t* __restrict__ ar, const point_t pred, const double dt)
{
  point_t loc[4]; 
  loc[0] = scale(pred, 9.0); 
  loc[1] = scale(ar[0], 19.0);
  loc[2] = scale(ar[1], -5.0); 
  loc[3] = ar[2]; 
  return scale(add_list(loc, 4), dt / 24.0);
}

point_t 
ab_am_four(const params_t* __restrict__ params, point_t pos, point_t* vels, 
           point_t* forces, double dt) 
{
  if ( !params ) {
    return POINT(0.0, 0.0);
  }  

  point_t vnew = vels[0]; 

  // Predict w_{i+1}
  point_t vpred = add(vels[0], bashforth(forces, dt));
  point_t apred = rate(params, vpred); 
  
  // Correct velocity and acceleration 
  point_t vcorr = add(vels[0], moulton(forces, apred, dt));
  point_t acorr = rate(params, vcorr);

  // Correct position 
  point_t ppred = add(pos, bashforth(vels, dt)); 
  point_t pcorr = add(pos, moulton(vels, vcorr, dt));

  // Fix evaluation history 
  for (size_t i = 3; i > 0; i--) {
    vels[i] = vels[i-1]; 
    forces[i] = forces[i-1];
  }

  // Update  
  vels[0]   = vcorr; 
  forces[0] = acorr; 
  return pcorr; 
}

void 
rk4_step(const params_t* __restrict__ params, size_t ptr, 
         trajectory_t* traj, double dt, history push_history)
{
  // Compute function evaluations  
  const point_t k1 = scale(rate(params, traj->vel[ptr-1]), dt); 
  const point_t v1 = add(traj->vel[ptr-1], scale(k1, 0.5)); 

  const point_t k2 = scale(rate(params, v1), dt);
  const point_t v2 = add(traj->vel[ptr-1], scale(k2, 0.5));

  const point_t k3 = scale(rate(params, v2), dt);
  const point_t v3 = add(traj->vel[ptr-1], k3); 

  const point_t k4 = scale(rate(params, v3), dt); 
  const point_t v4 = add(traj->vel[ptr-1], k4);   

  // Get updated velocity, position 
  const point_t ks[4] = {k1, scale(k2, 2.0), scale(k3, 2.0), k4};
  const point_t vel_weight = scale(add_list(ks, 4), 1.0/6.0); 
  const point_t vnew = add(traj->vel[ptr-1], vel_weight); 
  const point_t anew = rate(params, vnew); 

  // Ensure clean history for predictor corrector 
  push_history(traj->vel, vnew); 
  push_history(traj->force, anew); 

  // Update position from velocities 
  const point_t vs[4] = {traj->vel[ptr-1], scale(v1, 2.0), scale(v2, 2.0), v3};
  const point_t pos_weight = scale(add_list(vs, 4), dt/6.0);  
  traj->pos[ptr] = add(traj->pos[ptr-1], pos_weight);
}

point_t 
rate(const params_t* __restrict__ params, point_t vel)
{
  if ( !params ) {
    return POINT(0.0, 0.0);
  }

  const double C = params->k / params->m; 
  point_t s = POINT(vel.x - params->w, vel.y); 
  double speed = mag(s);
  point_t drag = scale(s, -C * speed);
  return POINT(drag.x, drag.y - G);
}

hit_e 
hit(const params_t* __restrict__ params, point_t a, point_t b)
{
  int status = NONE; 
  double ds = params->ds, hs = params->hs, dw = params->dw;  

  // Hit front of step
  if ( (a.x < ds && b.x >= ds) && b.y < hs ) {
    status = STEP_WALL; 
  } 

  // Hit floor before step
  if ( (b.x < ds && a.x < ds) && (b.y <= 0.0 && a.y > 0.0) ) {
    status = FLOOR; 
  } 

  // Hit back wall 
  if ( (a.x < dw && b.x > dw) ) {
    status = BACK_WALL; 
  }

  // Hit on top of step 
  if ( (b.x >= ds && b.x < dw ) 
    && (a.y > hs && b.y <= hs) ) {
    status = STEP_FLOOR; 
  } 

  return status;
}

/************ neville_eval() ******************************/ 
/* Evaluates an interpolated point at x using n points from t, y
 *
 * Caller Provides: 
 *   Valid pointers to t, y of equal size that represent points in R^2 
 *   The degree of interpolation 
 *   The x coordinate to interpolate at 
 *  
 * We return: 
 *   The y coordinate at x using the interpolated function 
 */ 
static double 
neville_eval(const double* t, const double* y, size_t n, double x)
{
  double q[3];  // Stores previous two points needed for next   
  for (size_t i = 0; i < n; i++) {
    q[i] = y[i]; 
  }

  for (size_t k = 1; k < n; k++) {
    for (size_t i = 0; i < n - k; i++) {
      const double C = t[i] - t[i + k]; 
      q[i] = ((x - t[i + k]) * q[i] + (t[i] - x) * q[i -1]) / C; 
    }
  }
  return q[0];
}

/************ poly_eval() *********************************/ 
/* Evaluates degree deg polynomial given by coefficients at c at x 
 * using Horner's Method 
 *
 * c is structured: 
 *   [c^n, c^{n-1}, ..., c^1, c^0]
 *
 * Caller Provides: 
 *   degree of polynomial 
 *   c of size deg+1 
 *   x to evaluate at 
 */ 
static double 
poly_eval(const double* __restrict__ c, const size_t deg, const double x)
{
  if ( !c ) {
    return 0.0; 
  } 

  if ( deg == 0 ) {
    return c[deg]; 
  }

  double result = c[deg]; // add constant term   
  for (size_t i = deg - 1; i >= 0; i--) {
    result = (result * x) + c[i];
  }

  return result; 
}

static void
components(double* __restrict__ r, point_t p1, point_t p2, point_t p3, 
           double boundary, dir_e axis)
{
  bool xdir = axis == X; 

  r[0] = (xdir? p1.x : p1.y) - boundary;
  r[1] = (xdir? p2.x : p2.y) - boundary; 
  r[2] = (xdir? p3.x : p3.y) - boundary; 
}

static double
dir(const point_t p, dir_e axis) 
{
  return (axis == X)? p.x : p.y; 
}



/************ quadratic_interpolate() *********************/ 
/* Interpolates position and velocity in either x or y direction using 
 * Neville's method. Packages new interpolated points and effective 
 * remaining dt 
 *
 * Caller provides: 
 *   Current and next positions and velocities 
 *   Direction of intersection 
 * 
 * We return: 
 *   quadInfo_t (see struct defn). 
 */ 
quadInfo_t 
quadratic_interpolate(const params_t* __restrict__ params, point_t prev, point_t curr, 
                      point_t next, point_t vprev, point_t vcurr, point_t vnext, 
                      double dt, dir_e axis, double boundary)
{
  quadInfo_t result = {
    .pos     = curr, 
    .vel     = vcurr, 
    .dt_rem  = 0.0, 
    .success = false  
  };

  // Guard division by zero 
  if (dt <= 0.0) {
    return result; 
  }
  
  const double t[3] = {-dt, 0.0, dt};

  // Save positions and velocities componentwise for interpolation 
  const double p[3] = {
    dir(prev, NOT(axis)), 
    dir(curr, NOT(axis)), 
    dir(next, NOT(axis))
  };
  const double vx[3] = {vprev.x, vcurr.x, vnext.x}; 
  const double vy[3] = {vprev.y, vcurr.y, vnext.y}; 

  // Get components in interpolate direction to form unique quadratic 
  double r[3]; 
  components(r, prev, curr, next, boundary, axis);

  // Unique quadratic through the three samples at each ti  
  const double a = (r[0] - 2.0 * r[1] + r[2]) / (2.0 * dt * dt); 
  const double b = (r[2] - r[0]) / (2.0 * dt); 
  const double c = r[1]; 

  // Form polynomial and its derivative as static array
  const double coeff[3] = {a, b, c}; 
  const double derv[2]  = {a, b}; 
  
  // Newton's method to compute point, initial guess is line through endpoints 
  double ti  = dt / 2.0; 
  size_t iter = 0; 

  // Get value t when intersected boundary  
  while ( iter < MAXITER ) { 
    double err = poly_eval(coeff, 2, ti); 
    if ( fabs(err) < params->eps ) {
      break; 
    }

    // Update guess 
    ti = ti - err / poly_eval(derv, 1, ti);
    iter++; 
  }

  if ( iter == MAXITER ) {
    return result; 
  } // Failed to converge to root (?)

  result.dt_rem = dt - ti; // Remaining timestep to be made  
  
  // Interpolate at found ti 
  const double vx_int = neville_eval(t, vx, 3, ti); 
  const double vy_int = neville_eval(t, vy, 3, ti); 
  const double x_int  = axis == X? boundary : neville_eval(t, p, 3, ti); 
  const double y_int  = axis == X? neville_eval(t, p, 3, ti) : boundary - params->eps;

  // Set output 
  result.pos = POINT(x_int, y_int); 
  result.vel = POINT(vx_int, vy_int); 
  result.success = true; 

  return result; 
}
