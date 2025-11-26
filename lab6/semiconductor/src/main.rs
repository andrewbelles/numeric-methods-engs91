//!
//! semiconductor.rs  Andrew Belles  Nov 6th, 2025 
//!
//! Computes solution to the semiconductor, nonlinear ODE 
//! using a 4-step A-B/A-M predictor corrector. Plots all outputs
//! on same plot 
//!

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

use plotters::prelude::*; 
use plotters_bitmap::BitMapBackend;
use plotters::style::Palette99;

// ODE: y'' = ay' - (y')^3 - y 
// System of first order 
// z' = [z2; az2 - (z2)^3 - z1]

///
/// Rate function for nonlinear system of 1st Order ODE's 
///
pub fn rate(alpha: f64, z: &[f64; 2], dz: &mut [f64; 2]) {
    dz[0] = z[1]; 
    dz[1] = alpha * z[1] - z[1].powi(3) - z[0];  
}

///
/// RK4 rk4r for 2 state linear system 
///
fn rk4(alpha: f64, dt: f64, t0: f64, tf: f64) -> (Vec<f64>, Vec<[f64; 2]>) {
    let el = ((tf - t0) / dt).floor() as usize;  
    let mut t: Vec<f64> = Vec::with_capacity(el + 1); 
    let mut y: Vec<[f64; 2]> = Vec::with_capacity(el + 1);

    t.push(t0);
    y.push([0.0, 0.1]);

    // initialize local per step states
    let mut k1: [f64; 2] = [0.0, 0.0];
    let mut k2: [f64; 2] = [0.0, 0.0];
    let mut k3: [f64; 2] = [0.0, 0.0];
    let mut k4: [f64; 2] = [0.0, 0.0];

    let mut w1: [f64; 2];
    let mut w2: [f64; 2] = [0.0, 0.0];
    let mut w3: [f64; 2] = [0.0, 0.0];
    let mut w4: [f64; 2] = [0.0, 0.0];

    let update = |w: &[f64; 2], k: &[f64; 2], u: &mut [f64; 2], h: f64| {
         u[0] = w[0] + h * k[0]; 
         u[1] = w[1] + h * k[1]; 
    }; 

    let next = |w1: &[f64; 2], ks: &[[f64; 2]; 4]| -> [f64; 2] {
        let mut wnext: [f64; 2] = [0.0, 0.0]; 
        let pool0 = (ks[0][0] + 2.0 * ks[1][0] + 2.0 * ks[2][0] + ks[3][0]);
        let pool1 = (ks[0][1] + 2.0 * ks[1][1] + 2.0 * ks[2][1] + ks[3][1]);

        wnext[0] = w1[0] + (dt / 6.0) * pool0; 
        wnext[1] = w1[1] + (dt / 6.0) * pool1; 
        wnext 
    };

    // main computation 
    for i in 1..=el {
        w1 = *y.last().unwrap(); 
        rate(alpha, &w1, &mut k1);
        update(&w1, &k1, &mut w2, 0.5_f64 * dt); 
        rate(alpha, &w2, &mut k2); 
        update(&w2, &k2, &mut w3, 0.5_f64 * dt); 
        rate(alpha, &w3, &mut k3); 
        update(&w3, &k3, &mut w4, dt);
        rate(alpha, &w4, &mut k4); 
        y.push(next(&w1, &[k1, k2, k3, k4]));

        let ti = t0 + (i as f64) * dt; 
        t.push(ti);
    }

    (t, y)
}

fn abam4_pred_corr(alpha: f64, dt: f64, t0: f64, tf: f64) 
    -> (Vec<f64>, Vec<[f64; 2]>) {
    // get first 3 values 
    let el = ((tf - t0) / dt).floor() as usize;  
    let (_, y0) = rk4(alpha, dt, t0, t0 + 3.0 * dt);
    
    let mut t: Vec<f64> = Vec::with_capacity(el + 1); 
    let mut y: Vec<[f64; 2]> = Vec::with_capacity(el + 1);
    let mut f: [[f64; 2]; 4] = [[0.0, 0.0]; 4]; 

    // initialize array 
    for (i, y0i) in y0.iter().enumerate() {
        t.push(t0 + (i as f64) * dt);
        y.push(*y0i);
        rate(alpha, y0i, &mut f[i]); // get first rate functions  
    }

    let predict = |w: &[f64; 2], f: &[[f64; 2]; 4], wpred: &mut [f64; 2]| {
        let pool0 = 55.0 * f[3][0] - 59.0 * f[2][0] + 37.0 * f[1][0] - 9.0 * f[0][0];
        let pool1 = 55.0 * f[3][1] - 59.0 * f[2][1] + 37.0 * f[1][1] - 9.0 * f[0][1];
        wpred[0] = w[0] + (dt / 24.0) * pool0;  
        wpred[1] = w[1] + (dt / 24.0) * pool1; 
    };

    // Ensure that we shift our rate functions before applying 
    let correct = |w: &mut [f64; 2], f: &[[f64; 2]; 4], fpred: &[f64; 2]| {
        let pool0 = 9.0 * fpred[0] + 19.0 * f[3][0] - 5.0 * f[2][0] + f[1][0]; 
        let pool1 = 9.0 * fpred[1] + 19.0 * f[3][1] - 5.0 * f[2][1] + f[1][1]; 
        w[0] += (dt / 24.0) * pool0; 
        w[1] += (dt / 24.0) * pool1; 
    };

    for i in 4..=el {
        // get current approximated value of y 
        let mut w: [f64; 2] = *y.last().unwrap(); 
        let mut wpred: [f64; 2] = [0.0, 0.0];

        // we have the four rate functions we need  
        predict(&w, &f, &mut wpred);
        let mut fpred = [0.0, 0.0]; 
        rate(alpha, &wpred, &mut fpred);
        
        // correct predicted value 
        correct(&mut w, &f, &fpred);
        let mut fcorr = [0.0, 0.0];
        rate(alpha, &w, &mut fcorr);

        // update rate functions 
        f.rotate_left(1);
        f[3] = fcorr; 

        // update arrays 
        let ti = t0 + (i as f64) * dt; 
        y.push(w); 
        t.push(ti);
    }

    (t, y)
}

///
/// Plot each element of solution from rk4 against time vector 
///
fn plot(t: &[f64], solutions: &Vec<Vec<[f64; 2]>>, alphas: &[f64; 5], path: &str, title: &str)
    -> Result<(), Box<dyn std::error::Error>> {

    let n = t.len(); 
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);

    for sol in solutions {
        for yi in sol {
            ymin = ymin.min(yi[0]).min(yi[1]);
            ymax = ymax.max(yi[0]).max(yi[1]); 
        }
    }

    let pad = (ymax - ymin) * 0.05; 
    ymax += pad; 
    ymin -= pad; 

    let (tmin, tmax) = (t[0], t[n - 1]); 

    let root = BitMapBackend::new(path, (1200,700)).into_drawing_area();
    root.fill(&WHITE)?; 
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 55)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .build_cartesian_2d(tmin..tmax, ymin..ymax)?; 

    chart.configure_mesh().x_desc("t").y_desc("voltage").draw()?; 

    for (i, (alpha, sol)) in alphas.iter().zip(solutions.iter()).enumerate() {
        let color0 = Palette99::pick(i); 
        let color1 = color0.mix(0.55);

        chart.draw_series(LineSeries::new(
            (0..n).map(|j| (t[j], sol[j][0])),
                ShapeStyle {
                    color: color0.to_rgba(), 
                    filled: false, 
                    stroke_width: 2 
                },
            ))?
            .label(format!("N1: {}", alpha))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], 
                    ShapeStyle{ 
                        color: color0.to_rgba(),
                        filled: false, 
                        stroke_width: 2 
                    })});

        chart.draw_series(LineSeries::new(
            (0..n).map(|j| (t[j], sol[j][1])),
                ShapeStyle {
                    color: color1.to_rgba(), 
                    filled: false, 
                    stroke_width: 2 
                },
            ))?
            .label(format!("N2: {}", alpha))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], 
                    ShapeStyle{ 
                        color: color1.to_rgba(),
                        filled: false, 
                        stroke_width: 2 
                    })});
    }

    chart.configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .draw()?; 

    root.present()?; 
    Ok(())
} 

fn solve<F>(func: &F, dt: f64, path: &str, title: &str)
where F: Fn(f64, f64, f64, f64) -> (Vec<f64>, Vec<[f64; 2]>) {
    let (t0, tf) = (0.0, 100.0);
    let n = ((tf - t0) / dt).floor() as usize; 
    let alphas = [0.5, 1.5, 2.5, 3.5, 4.5];
    let mut t = Vec::with_capacity(n + 1);
    let mut solutions = Vec::with_capacity(alphas.len());

    for i in 0..=n {
        t.push(t0 + (i as f64) * dt); 
    } 

    for a in alphas {
        let (_, y) = func(a, dt, t0, tf); 
        solutions.push(y);
    }
    
    let _ = plot(&t, &solutions, &alphas, path, title); 
}


fn main() {

    solve(
        &abam4_pred_corr, 
        1e-3, 
        "abam4_semiconductor.png", 
        "AB/AM 4th Order Semiconductor System for Range of (a), dt = 1e-3"
    );
    solve(
        &rk4, 
        1e-3,
        "rk4_semiconductor.png", 
        "Runge-Kutta 4th Order, Semiconductor System for Range of (a), dt = 1e-3"
    );

    solve(
        &abam4_pred_corr, 
        1e-1, 
        "bad_timestep.png", 
        "AB/AM 4th Order, Semiconductor System for Range of (a), dt = 1e-1"
    );
}
