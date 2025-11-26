//!
//! ecosystem.rs  Andrew Belles  Nov 6th, 2025  
//!
//! Runge-Kutta 4th Order solver for system of  
//! linear coupled differential equations. Plots result 
//!
//!

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

use plotters::prelude::*; 
use plotters_bitmap::BitMapBackend;

/// 
/// Metadata for ecosystem stored in static arrays 
///
struct Ecosystem {
    ic: [f64; 2],    
    a:  [f64; 2], 
    b:  [f64; 2], 
    c:  [f64; 2],
    ts: [f64; 2] 
}

impl Ecosystem {
    pub fn new(
        ic: [f64; 2], 
        a:  [f64; 2], 
        b:  [f64; 2], 
        c:  [f64; 2], 
        ts: [f64; 2]) -> Ecosystem 
    { 
        Ecosystem { ic, a, b, c, ts }
    }

    ///
    /// Rate function as a system of first order ODE's 
    ///
    fn rate(&self, pop: &[f64; 2], d_pop: &mut [f64; 2]) {
        d_pop[0] = pop[0] * (self.a[0] - self.b[0] * pop[0] - self.c[0] * pop[1]);  
        d_pop[1] = pop[1] * (self.a[1] - self.b[1] * pop[1] - self.c[1] * pop[0]);  
    }
 
    ///
    /// Solve the Ecosystem at the given initial conditions given some step size 
    /// using Runge-Kutta 4th Order 
    ///
    pub fn solve(&self, dt: f64) -> (Vec<f64>, Vec<[f64; 2]>) {
        let n = ((self.ts[1] - self.ts[0]) / dt).floor() as usize;  
        let mut t: Vec<f64> = Vec::with_capacity(n + 1); 
        let mut y: Vec<[f64; 2]> = Vec::with_capacity(n + 1);

        t.push(self.ts[0]);
        y.push(self.ic);

        // initialize local per step states
        let mut k1: [f64; 2] = [0.0, 0.0];
        let mut k2: [f64; 2] = [0.0, 0.0];
        let mut k3: [f64; 2] = [0.0, 0.0];
        let mut k4: [f64; 2] = [0.0, 0.0];

        let mut w1: [f64; 2];
        let mut w2: [f64; 2] = [0.0, 0.0];
        let mut w3: [f64; 2] = [0.0, 0.0];
        let mut w4: [f64; 2] = [0.0, 0.0];

        // update helper 
        let update = |w: &[f64; 2], k: &[f64; 2], u: &mut [f64; 2], h: f64| {
             u[0] = w[0] + h * k[0]; 
             u[1] = w[1] + h * k[1]; 
        }; 

        // compute next values 
        let next = |w1: &[f64; 2], ks: &[[f64; 2]; 4]| -> [f64; 2] {
            let mut wnext: [f64; 2] = [0.0, 0.0]; 
            let pool0 = ks[0][0] + 2.0 * ks[1][0] + 2.0 * ks[2][0] + ks[3][0];
            let pool1 = ks[0][1] + 2.0 * ks[1][1] + 2.0 * ks[2][1] + ks[3][1];

            wnext[0] = w1[0] + (dt / 6.0) * pool0; 
            wnext[1] = w1[1] + (dt / 6.0) * pool1; 
            wnext 
        };

        // main loop
        for i in 1..=n {
            w1 = *y.last().unwrap(); 
            self.rate(&w1, &mut k1);
            update(&w1, &k1, &mut w2, 0.5_f64 * dt); 
            self.rate(&w2, &mut k2); 
            update(&w2, &k2, &mut w3, 0.5_f64 * dt); 
            self.rate(&w3, &mut k3); 
            update(&w3, &k3, &mut w4, dt);
            self.rate(&w4, &mut k4); 
            y.push(next(&w1, &[k1, k2, k3, k4]));

            let ti = self.ts[0] + (i as f64) * dt; 
            t.push(ti);
        }

        // return time vector and solution 
        (t, y)
    }
}

///
/// Plot each element of solution from rk4 against time vector 
///
pub fn plot(t: &[f64], y: &Vec<[f64; 2]>, path: &str, title: &str)
    -> Result<(), Box<dyn std::error::Error>> {

    let n = t.len(); 
    let (tmin, tmax) = (t[0], t[n - 1]); 
    
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for yi in y {
        ymin = ymin.min(yi[0]).min(yi[1]);
        ymax = ymax.max(yi[0]).max(yi[1]); 
    }
    let pad = (ymax - ymin) * 0.05; 
    ymax += pad; 

    let root = BitMapBackend::new(path, (1200,700)).into_drawing_area();
    root.fill(&WHITE)?; 
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 55)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .build_cartesian_2d(tmin..tmax, 0.0..ymax)?; 

    chart.configure_mesh().x_desc("t").y_desc("population").draw()?; 

    chart.draw_series(LineSeries::new(
        (0..n).map(|i| (t[i], y[i][0])),
            &RED,
        ))?
        .label("N1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart.draw_series(LineSeries::new(
        (0..n).map(|i| (t[i], y[i][1])),
            &BLUE,
        ))?
        .label("N2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .draw()?; 

    root.present()?; 
    Ok(())
} 

///
/// Compares larger timesteps to dt = 1e-4 (which I've qualitatively determined 
/// to be exact
/// Plots on semilogy, outputs to hardcoded, error.png 
///
pub fn compare(dt: f64) -> Result<(), Box<dyn std::error::Error>> {
    let dtarr = [dt, 2.0 * dt, 4.0 * dt, 8.0 * dt, 16.0 * dt]; 
    let eco = Ecosystem::new(
        [1e5, 1e5], 
        [0.1, 0.1], 
        [8e-7, 8e-7], 
        [1e-6, 1e-7], 
        [0.0, 10.0]
    ); 
    let mut solutions = Vec::with_capacity(dtarr.len()); 
    
    for dti in dtarr {
        let (_, yi) = eco.solve(dti); 
        solutions.push(*yi.last().unwrap()); 
    }  
    
    // plot inverse timestep value against difference from exact 
    let inv_dt: Vec<f64> = dtarr[1..].iter().rev().map(|&dti| 1.0 / dti ).collect();
    let exact = solutions.first().unwrap(); 
    let rel_err0: Vec<f64> = solutions[1..]
        .iter()
        .rev()
        .map(|s| {
            (s[0] - exact[0]).abs() / exact[0].abs()
        })
        .collect(); 
    let rel_err1: Vec<f64> = solutions[1..]
        .iter()
        .rev()
        .map(|s| {
        (s[1] - exact[1]).abs() / exact[1].abs()
        })
        .collect(); 

    let logerr0: Vec<f64> = rel_err0
        .iter()
        .map(|&er| (er.max(1e-16)).log10())
        .collect(); 
    let logerr1: Vec<f64> = rel_err1
        .iter()
        .map(|&er| (er.max(1e-16)).log10())
        .collect(); 

    let mut ymin = logerr0
        .iter()
        .chain(logerr1.iter())
        .copied()
        .fold(f64::INFINITY, f64::min);
    let mut ymax = logerr0
        .iter() 
        .chain(logerr1.iter())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max); 

    ymin = ymin.floor(); 
    ymax = ymax.ceil(); 

    let root = BitMapBackend::new("errors.png", (1200,700)).into_drawing_area();
    root.fill(&WHITE)?; 
    let mut chart = ChartBuilder::on(&root)
        .caption("Relative Error vs 1/dt [semilogy-scale]", ("sans-serif", 22))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 70)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .build_cartesian_2d(
            (*inv_dt.first().unwrap())..(*inv_dt.last().unwrap()), 
            ymin..ymax)?; 

    chart.configure_mesh()
        .x_desc("1/dt")
        .y_desc("relative error")
        .y_label_formatter(&|v| format!("1e{:.0}", v))
        .draw()?; 

    chart.draw_series(LineSeries::new(
        (0..inv_dt.len()).map(|i| (inv_dt[i], logerr0[i])),
        &RED, 
    ))? 
    .label("N1")
    .legend(|(x,y)| PathElement::new(vec![(x,y), (x + 20, y)], RED));

    chart.draw_series(LineSeries::new(
        (0..inv_dt.len()).map(|i| (inv_dt[i], logerr1[i])),
        &BLUE, 
    ))? 
    .label("N2")
    .legend(|(x,y)| PathElement::new(vec![(x,y), (x + 20, y)], BLUE));

    chart.configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.85))
        .draw()?;

    root.present()?; 
    Ok(())
}

///
/// Runs for a given timestep, calling correct constructor, solving system 
/// Then passing solution to be plotted. 
///
pub fn run(dt: f64, path: &str, title: &str) {
    let eco = Ecosystem::new(
        [1e5, 1e5], 
        [0.1, 0.1],
        [8e-7, 8e-7],
        [1e-6, 1e-7], 
        [0.0, 10.0]
    ); 
    let (t, y) = eco.solve(dt);
    let _ = plot(&t, &y, path, title);
    let _ = compare(dt);
}

fn main() {
    run(1e-4, "rk4_ecosystem.png", "Ecosystem over Time, h=1e-4");
}
