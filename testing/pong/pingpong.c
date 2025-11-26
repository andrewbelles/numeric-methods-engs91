
#include <math.h> 
#include <stdio.h> 
#include <stdlib.h> 

typedef struct {
  double x, y;
} point_t; 

typedef struct {
  double hs; 
  double dw; 
  double d; 
  double ds; 
  double v0; 
  double k; 
  double m; 
  double w; 
  double eps; 
  double dt; 
} params_t; 

typedef struct {
  point_t* array; 
  int size, max_size; 
} data_t; 

#define g 9.81 
#define rad_one_deg M_PI / 180.0 
#define TRAJCOUNT 179
#define SOLCOUNT 4

// Trajectory Functions 
int hit(params_t* params, point_t a, point_t b);
point_t linear_interpolate_known_y(point_t a,  point_t b, double y);
double linear_interpolate_known_x(params_t* params, point_t a, point_t b, point_t avel, 
                                point_t bvel, point_t* pos_new, point_t* vel_new, 
                                double x); 
data_t eulers(params_t* params, double theta);
point_t step(params_t* params, point_t a, point_t b, double dt, int flag);
double force(params_t* params, point_t vel, int direction);

// Bisection Functions 
void map_trajectories(params_t* params, point_t* trajectories);
void bisection(params_t* params, point_t* trajectories, 
               double* solutions);

// File handling, etc. Functions 
int parse_args(params_t* params, int argc, char* argv[]);

int main(int argc, char* argv[]) {
  int i = 0, j = 0; 
  double solutions[SOLCOUNT];
  point_t trajectories[TRAJCOUNT];
  params_t params; 
  data_t traj; 
  FILE* gp = NULL; 

  switch (parse_args(&params, argc, argv)) {
    case 0:
      printf("Successfully loaded parameters from file: %s\n", argv[1]); 
      break; 
    case 1: 
      printf("Pass a valid pointer to parameters to parse_args\n");
      return 1; 
    case 2: 
      printf("Too many arguments. Usage: ./pingpong [parameters.txt]\n");
      return 2;  
    case 3: 
      printf("Default parameters loaded\n"); 
      break; 
    case 4: 
    case 5: 
      printf("Invalid parameter file: %s\n", argv[1]);
      return 3;  
    default: 
      break; 
  }

  // Clear solution array to zeros 
  for (i = 0; i < SOLCOUNT; i++) {
    solutions[i] = 0.0; 
  }

  map_trajectories(&params, trajectories);
  bisection(&params, trajectories, solutions);

  gp = popen("gnuplot", "w"); 
  if (gp == NULL) {
    printf("Failed to open pipe to gnuplot\n"); 
    return 4; 
  }

  fprintf(gp, "set terminal pngcairo\n");
  fprintf(gp, "set output 'solutions.png'\n");
  fprintf(gp, "set xlabel 'x [m]'\n");
  fprintf(gp, "set ylabel 'z [m]'\n");
  fprintf(gp, "set title 'Solution Trajectories'\n");

  fprintf(gp, "plot ");
  for (i = 0; i < SOLCOUNT; i++) {
    if (solutions[i] != 0.0) {
        if ( i > 0 ) {
          fprintf(gp, ", ");
        }
        fprintf(gp, "'-' with lines lw 1 title 'angle=%.4lf'", 
                solutions[i] * 180.0 / M_PI);
    }
  }
  fprintf(gp, "\n");

  for (i = 0; i < SOLCOUNT; i++) {
    if (solutions[i] != 0.0) {
      traj = eulers(&params, solutions[i]);
      for (j = 0; j < traj.size; j++) {
        fprintf(gp, "%lf %lf\n", traj.array[j].x, traj.array[j].y);
      }
      fprintf(gp, "e\n");
      free(traj.array);
    }
  }

  fprintf(gp, "unset output\n");
  pclose(gp);

  return 0; 
}


int parse_args(params_t* params, int argc, char* argv[]) {
  FILE* param_file = NULL; 
  const params_t defaults = (params_t){
    .m   = 2.7e-3, 
    .k   = 5e-4, 
    .v0  = 30.0,
    .ds  = 6.0, 
    .hs  = 1.0, 
    .d   = 8.0, 
    .dw  = 9.0, 
    .w   = 1.5, 
    .dt  = 1e-3, 
    .eps = 1e-3 
  };

  if ( params == NULL ) {
    return 1; 
  }
  
  if (argc > 2) {
    return 2; 
  }

  // default parameters 
  if (argc == 1) {
    (*params) = defaults;  
    return 3; 
  }

  param_file = fopen(argv[1], "r"); 
  if (param_file == NULL) {
    return 4; 
  }

  if (fscanf(param_file, "%lf", &params->m) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->k) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->v0) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->ds) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->hs) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->d) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->dw) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->w) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->dt) == EOF) {
    return 5; 
  }
  if (fscanf(param_file, "%lf", &params->eps) == EOF) {
    return 5; 
  }

  fclose(param_file);
  return 0; 
}

data_t eulers(params_t* params, double theta) {
  data_t trajectory;
  int done = 0, status = 0; 
  double t_remaining = 0.0, wall_distance = 0.0, floor_height = 0.0;
  point_t p = {0.0, 0.0}, pnext;
  point_t v = {params->v0 * cos(theta), params->v0 * sin(theta)}, vnext; 

  trajectory.array = (point_t*)malloc(1000 * sizeof(point_t));
  trajectory.max_size = 1000; 
  trajectory.size = 0; 

  while (!done) {
    // Ensure data array isn't at Capacity 
    if (trajectory.size == trajectory.max_size) {
      trajectory.max_size *= 2; 
      trajectory.array = (point_t*)realloc(trajectory.array, 
                                           trajectory.max_size * sizeof(point_t));
    }
    
    // Get next velocity and Position 
    vnext = step(params, p, v, params->dt, 1); 
    pnext = step(params, p, v, params->dt, 0);

    // Handle hit against wall or floor 
    status = hit(params, p, pnext);
    switch (status) {
      case 1: 
      case 3: 
        // Get x distance to interpolate position and velocity into  
        wall_distance = (status == 1)? params->ds : params->dw; 
        t_remaining = linear_interpolate_known_x(params, p, pnext, v, vnext, 
                                                 &pnext, &vnext, wall_distance);
        // Remaining timestep after moving point to interpolated position at wall 
        t_remaining = params->dt - t_remaining * params->dt; 
        pnext = step(params, pnext, vnext, t_remaining, 0);
        vnext = step(params, pnext, vnext, t_remaining, 1); 
        break; 
      case 2: 
      case 4: 
        // Interpolate with ground at either 0 or the step, flag that we are done
        floor_height = (status == 2)? 0.0 : params->hs; 
        pnext = linear_interpolate_known_y(p, pnext, floor_height); 
        done = 1; 
        break; 
      case 0:
      default: 
        // Do nothing 
        break; 
    } 

    // Update position and velocity, store values 
    p = pnext; 
    v = vnext; 
    trajectory.array[trajectory.size] = p; 
    trajectory.size++; 
  }

  return trajectory;
}

point_t step(params_t* params, point_t pos, point_t vel, double dt, int flag) {
  point_t new; 

  switch (flag) {
    // Update position, b is the velocity 
    case 0: 
      new.x = pos.x + dt * vel.x; 
      new.y = pos.y + dt * vel.y; 
      break; 
    // Update velocity 
    case 1: 
      new.x = vel.x + dt * force(params, vel, 0);
      new.y = vel.y + dt * force(params, vel, 1); 
      break; 
    default: 
      break;
  }

  return new; 
}

double force(params_t* params, point_t vel, int direction) {
  double magnitude = sqrt(vel.x * vel.x + vel.y * vel.y); 
  double v = 0.0, result = 0.0; 
  
  if (!direction) {
    v = vel.x - params->w; 
  } else {
    v = vel.y; 
  }

  result = -(params->k / params->m) * v * magnitude; 

  if ( !direction ) {
    return result; 
  } else {
    return result - g; 
  }
}

int hit(params_t* params, point_t a, point_t b) {
  int status = 0; 
  double ds = params->ds, hs = params->hs, dw = params->dw;  

  // Hit front of step
  if ( (a.x < ds && b.x >= ds) && b.y < hs ) {
    status = 1; 
  } 

  // Hit floor before step
  if ( (b.x < ds && a.x < ds) && (b.y <= 0.0 && a.y > 0.0) ) {
    status = 2; 
  } 

  // Hit back wall 
  if ( (a.x < dw && b.x > dw) ) {
    status = 3; 
  }

  // Hit on top of step 
  if ( (b.x >= ds && b.x < dw ) 
    && (a.y > hs && b.y <= hs) ) {
    status = 4; 
  } 

  return status;
}

point_t linear_interpolate_known_y(point_t a,  point_t b, double y) {
  double dy = b.y - a.y;
  double dx = b.x - a.x; 
  double t = 0.0; 

  // Prevent division by zero 
  if (fabs(dy) < 1e-12) {
    return (point_t){b.x, y}; 
  }

  // Similar triangle ratio, return interpolated point 
  t = (y - a.y) / dy; 
  return (point_t){a.x + t * dx, y};
}

double linear_interpolate_known_x(params_t* params, point_t a, point_t b, point_t avel, 
                                  point_t bvel, point_t* pos_new, point_t* vel_new, 
                                  double x) {
  double dy = b.y - a.y; 
  double dx = b.x - a.x; 
  double t = (x - a.x) / dx; 
  point_t p = (point_t){x - params->eps, a.y + t * dy};
  point_t v = (point_t){
    .x = -(avel.x + t * (bvel.x - avel.x)), 
    .y = avel.y + t * (bvel.y - avel.y)
  };

  // Update values 
  (*pos_new) = p; 
  (*vel_new) = v; 
  return t; 
}

void map_trajectories(params_t* params, point_t* trajectories) {
  int i = 0, last = 0; 
  double angle = 0.0, xfinal = 0.0, error = 0.0;
  data_t traj; 

  for (i = 0; i < TRAJCOUNT; i++) {
    angle = rad_one_deg * (double)(i + 1);  
    traj = eulers(params, angle);  

    last = traj.size - 1; 
    xfinal = traj.array[last].x; 
    error = xfinal - params->d; 
    trajectories[i] = (point_t){angle, error};
    
    free(traj.array);
  }
}

void bisection(params_t* params, point_t* trajectories, 
               double* solutions) {
  int L = 0, R = 0, S = 0, done = 0; 
  double error = 0.0, mid_angle = 0.0, left_angle = 0.0, right_angle = 0.0; 
  double left_final = 0.0, right_final = 0.0; 
  data_t traj; 

  while (!done && L < TRAJCOUNT - 1 && S < SOLCOUNT) {
    R = L + 1; 

    // Set up starting bracket for current Bisection 
    while (R < TRAJCOUNT && (trajectories[L].y * trajectories[R].y) > 0.0) {
      R++; 
    }

    // If R made it to end, we do not have any more solutions 
    if (R >= TRAJCOUNT) {
      done = 1; 
    } else {
      left_angle = trajectories[L].x; 
      right_angle = trajectories[R].x; 
      left_final = trajectories[L].y; 
      right_final = trajectories[R].y; 

      // Initialize Error, and First angle to left bracket in case it is a solution 
      error = left_final; 
      mid_angle = left_angle; 

      while (fabs(error) > params->eps) {
        // Simulate trajectory with mid angle 
        mid_angle = 0.5 * (left_angle + right_angle); 
        traj = eulers(params, mid_angle); 
        error = traj.array[traj.size - 1].x - params->d; 

        // If mid angle is on same side as left, replace L else replace R 
        if (left_final * error <= 0.0) {
          right_angle = mid_angle; 
          right_final = error; 
        } else {
          left_angle = mid_angle; 
          left_final = error; 
        }

        free(traj.array);
      } 

      // Add solution safely 
      if (L < TRAJCOUNT) {
        solutions[S] = mid_angle; 
        S++; 
      }

      // Update left brack to previous right bracket; 
      L = R; 
    }
  }
}
