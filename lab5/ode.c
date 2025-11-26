/* 
*
*
*
*
*
*/ 

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 

typedef struct {
  double* y; 
  size_t size, max; 
} vec_t; 

typedef double (*functor)(const double, const double);
typedef void (*ode)(functor rate, vec_t*, const vec_t*, const double);

static vec_t* linspace(const double lo, const double hi, const double step);
static inline double three_rate(const double y, const double t); 
static inline vec_t* three_exact(const vec_t* t);
static vec_t* solve(ode method, functor rate, const double y0, const double dt, const vec_t* t);
static void euler(functor rate, vec_t* vec, const vec_t* t, double dt);
static void midpoint(functor rate, vec_t* vec, const vec_t* t, double dt);
static void modified_euler(functor rate, vec_t* vec, const vec_t* t, double dt);
static void rk4(functor rate, vec_t* vec, const vec_t* t, double dt); 
static void abam_pred_corrector(functor rate, vec_t* vec, const vec_t* t, double dt);

static vec_t* vec_new(size_t n, double* ar);
static void vec_delete(vec_t* v);

// VLA const stepsize array 
const int num_sizes = 6; 
const int method_count = 5;

const double stepsizes[6] = {
  1.0 / (float)(2 << 3),
  1.0 / (float)(2 << 4),
  1.0 / (float)(2 << 5),
  1.0 / (float)(2 << 6),
  1.0 / (float)(2 << 7),
  1.0 / (float)(2 << 8),
};

const ode methods[5] = { euler, midpoint, modified_euler, rk4, abam_pred_corrector };
const char method_enum[5][32] = { "euler", "midpoint", "mod_euler", "rk4", "abam" };

int main(void) 
{
  vec_t* t = NULL, *v = NULL, *truth = NULL; 
  vec_t *error[num_sizes][method_count];
  int i = 0, j = 0;
  size_t k = 0; 
  double y0 = exp(-1);
  char bufr[256]; 
  FILE* fp = NULL; 

  for (j = 0; j < num_sizes; j++) {
    t = linspace(1.0, 2.0, stepsizes[j]); 
    truth = three_exact(t); 

    for (i = 0; i < method_count; i++) {
      error[j][i] = vec_new(t->size, NULL);
      error[j][i]->size = t->size; 

      snprintf(bufr, sizeof(bufr), "%s_traj_%d.csv", method_enum[i], j + 3); 
      if ( (fp = fopen(bufr, "w")) == NULL ) {
        exit( 99 ); 
      }
      fprintf(fp, "%.4e\n", stepsizes[j]);
      v = solve(methods[i], three_rate, y0, stepsizes[j], t);

      for (k = 0; k < t->size; k++) {
        error[j][i]->y[k] = fabs(truth->y[k] - v->y[k]);
        fprintf(fp, "%.15f, %.15f, %.15f\n", t->y[k], v->y[k], truth->y[k]);
      }
      fclose(fp);

      if ( v ) {
        vec_delete(v); 
        v = NULL; 
      }
    }

    // build vector of analytic solution 
    if ( t ) {
      vec_delete(t);
      t = NULL; 
    }

    if ( truth ) {
      vec_delete(truth); 
      truth = NULL; 
    }
  }

  /********** save errors **********************************/ 

  for (j = 0; j < num_sizes; j++) {
    t = linspace(1.0, 2.0, stepsizes[j]); 
    truth = three_exact(t); 

    for (i = 0; i < method_count; i++) {
      snprintf(bufr, sizeof(bufr), "%s_error_%d.csv", method_enum[i], j + 3);
      if ( (fp = fopen(bufr, "w")) == NULL ) {
        exit( 99 ); 
      }
      fprintf(fp, "%.4e\n", stepsizes[j]);

      for (k = 0; k < t->size; k++) {
        // save relative error over time 
        fprintf(fp, "%.15f, %.15f\n", t->y[k], error[j][i]->y[k]);
      }

      fclose(fp);
    }

    if ( t ) {
      vec_delete(t);
      t = NULL; 
    }

    if ( truth ) {
      vec_delete(truth); 
      truth = NULL; 
    }
  }

  for (i = 0; i < method_count; i++) {
    snprintf(bufr, sizeof(bufr), "%s_abs_error.csv", method_enum[i]);
    if ( (fp = fopen(bufr, "w")) == NULL ) {
      exit( 99 ); 
    }

    for (j = 0; j < num_sizes; j++) {
      k = error[j][i]->size - 1; 

      // get absolute error 
      fprintf(fp, "%.15f, %.15f\n", 1.0 / stepsizes[j], error[j][i]->y[k]);
      if ( error[j][i] ) { 
        vec_delete(error[j][i]);
        error[j][i] = NULL; 
      }
    }

    fclose(fp);
  }

  exit(0);
}

static inline double 
three_rate(const double y, const double t)
{
  return (2.0 * y) * (( 1.0 / t ) - t);   
}

static inline vec_t* 
three_exact(const vec_t* t)
{
  size_t i = 0, n = t->size; 
  vec_t* v = vec_new(n, NULL);
  v->size = n;

  for (i = 0; i < n; i++) {
    const double t2 = t->y[i] * t->y[i]; 
    v->y[i] = t2 * exp(-t2);
  }

  return v; 
}

/************ wrapper over ode solvers *********************/

static vec_t*
solve(ode method, functor rate, const double y0, const double dt, const vec_t* t)
{
  const size_t n = t->size; 
  vec_t* vec = vec_new(n, NULL); // prefix sum 
  vec->y[0] = y0; // set ic  
 
  method(rate, vec, t, dt);
  vec->size = n; 
  return vec; 
}

/************ explicit single step methods ****************/

static void  
euler(functor rate, vec_t* vec, const vec_t* t, double dt)
{
  size_t n = t->size, i = 0; 
  double r = 0.0; 

  for (i = 1; i < n; i++) {
    r = rate(vec->y[i - 1], t->y[i - 1]); 
    vec->y[i] = vec->y[i - 1] + (dt * r); 
  }
}

static void 
midpoint(functor rate, vec_t* vec, const vec_t* t, double dt)
{
  size_t n = t->size, i = 0; 
  double r = 0.0, ymid = 0.0, tmid = 0.0, half = 0.0; 

  for (i = 1; i < n; i++) {
    // compute half step 
    r = rate(vec->y[i - 1], t->y[i - 1]);
    ymid = vec->y[i - 1] + (0.5 * dt * r);
    tmid = t->y[i - 1] + (0.5 * dt);
    half = rate(ymid, tmid);

    vec->y[i] = vec->y[i - 1] + (dt * half);
  }
}

static void 
modified_euler(functor rate, vec_t* vec, const vec_t* t, double dt)
{
  size_t n = t->size, i = 0; 
  double r = 0.0, yfull = 0.0, full = 0.0; 

  for (i = 1; i < n; i++) {
    // compute half step 
    r = rate(vec->y[i - 1], t->y[i - 1]);
    yfull = vec->y[i - 1] + dt * r;
    full = rate(yfull, t->y[i]);

    vec->y[i] = vec->y[i - 1] + (0.5 * dt * (full + r));
  }
}

static void 
rk4(functor rate, vec_t* vec, const vec_t* t, double dt)
{
  size_t n = t->size, i = 0; 
  double r = 0.0, half1 = 0.0, half2 = 0.0, full = 0.0; 
  double yhalf1 = 0.0, yhalf2 = 0.0, yfull = 0.0, thalf = 0.0; 

  for (i = 1; i < n; i++) {
    r      = rate(vec->y[i - 1], t->y[i - 1]); 
    thalf  = t->y[i - 1] + 0.5 * dt; 
    yhalf1 = vec->y[i - 1] + (0.5 * dt) * r; 
    half1  = rate(yhalf1, thalf);
    yhalf2 = vec->y[i - 1] + (0.5 * dt) * half1; 
    half2  = rate(yhalf2, thalf); 
    yfull  = vec->y[i - 1] + dt * half2;
    full   = rate(yfull, t->y[i]); 
    
    vec->y[i] = vec->y[i - 1] + (dt / 6.0) * (r + 2.0 * (half1 + half2) + full);
  }
}

/************ multi-step predictor corrector scheme *******/ 

static void 
abam_pred_corrector(functor rate, vec_t* vec, const vec_t* t, double dt)
{
  size_t n = t->size, i = 0; 
  double fi = 0.0, fprev = 0.0, ynext = 0.0, next = 0.0; 

  // compute a single midpoint (~O(h^2)) step 
  {
    double f1 = rate(vec->y[0], t->y[0]);
    double ymid = vec->y[0] + (0.5 * dt * f1);
    double tmid = t->y[0] + (0.5 * dt);
    double f2 = rate(ymid, tmid);
    vec->y[1] = vec->y[0] + (dt * f2);
    fi = f1;  
  } // end scope midpoint local values 


  for (i = 2; i < n; i++) {
    // compute two-step AB 
    fprev = fi; 
    fi = rate(vec->y[i - 1], t->y[i - 1]);
    ynext = vec->y[i - 1] + (0.5 * dt) * (3.0 * fi - fprev);
    // prediction of y_{i+1}
    next  = rate(ynext, t->y[i]);
    
    // correct prediction 
    vec->y[i] = vec->y[i - 1] + (dt / 12.0) * (5.0 * next + 8.0 * fi - fprev); 
  }
}

/************ helper functions ****************************/

static vec_t* 
linspace(const double lo, const double hi, const double step)
{
  if ( step <= 0.0 ) {
    return NULL; 
  }

  vec_t* v = NULL; 
  size_t n = (size_t)floor((hi - lo) / step) + 1;

  double* x = malloc(n * sizeof(double));
  for (size_t i = 0; i < n; i++) {
      x[i] = lo + step * (double)i;
  }

  v = vec_new(n, x);
  return v;
}

static vec_t* 
vec_new(size_t n, double* ar)
{
  if ( n == 0 ) {
    return NULL; 
  } 

  vec_t* new = malloc(sizeof(vec_t));
  if ( !new ) {
    exit( 99 );
  }

  if ( ar ) {
    // assume that n corresponds to size of ar 
    new->y = ar; 
    new->size = n; 
    new->max = n; 
  } else {
    new->y = malloc(n * sizeof(double));
    new->size = 0; 
    new->max  = n; 
  }

  return new; 
}

static void 
vec_delete(vec_t* v)
{
  if ( v ) {
    if ( v->y ) {
      free(v->y);
    }
    free(v);
  }
}
