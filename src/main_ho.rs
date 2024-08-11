use ndarray::{Array1, Array2};
use num::complex::Complex;
use plotters::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

use rustfft::FftPlanner;

use spgpe::types::*;
use spgpe::constants::*;
use spgpe::utils::*;
use spgpe::rkf45;

fn potential(x: &Array1<f64>, y: &Array1<f64>, gridpoints: &(usize, usize), trap: &Trap) -> Array2<f64> {

    // Calculate the squared values
    let xx = x.mapv(|x| x.powi(2));
    let yy = y.mapv(|y| y.powi(2));

    // Compute the potential using broadcasting
    let potential = 0.5 * (&xx.into_shape((gridpoints.0, 1)).unwrap() + &yy.into_shape((1, gridpoints.1)).unwrap());

    potential
}

fn laplacian_2d(phi: &Array2<Complex<f64>>, dx: &f64, dy: &f64, gridpoints: &(usize, usize)) -> Array2<Complex<f64>> {
    let mut laplacian = Array2::zeros(f.raw_dim());

    for i in 1..gridpoints.0-1 {
        for j in 1..gridpoints.1-1 {
            let laplacian_x = (f[[i+1,j]] - 2.0*f[[i,j]] + f[[i-1,j]]) / dx.powi(2);
            let laplacian_y = (f[[i,j+1]] - 2.0*f[[i,j]] + f[[i,j-1]]) / dy.powi(2);
            laplacian[[i,j]] = laplacian_x + laplacian_y;
        }
    }

    laplacian
//    let mut planner = FftPlanner::new();
//    let fft = planner.plan_fft_forward(gridpoints.0);
//    let ifft = planner.plan_fft_inverse(gridpoints.0);
//
//    // 1. Compute FFT
//    let mut phi_k = phi.clone();
//    fft.process(phi_k.as_slice_mut().unwrap());
//
//    // 2. Compute Laplacian in Fourier space
//    for i in 0..gridpoints.0 {
//        let kx = if i <= gridpoints.0 / 2 {
//            (i as f64) * 2.0 * std::f64::consts::PI / dx
//        } else {
//            (i as f64 - gridpoints.0 as f64) * 2.0 * std::f64::consts::PI / dx
//        };
//
//        for j in 0..gridpoints.1 {
//            let ky = if j <= gridpoints.1 / 2 {
//                (j as f64) * 2.0 * std::f64::consts::PI / dy
//            } else {
//                (j as f64 - gridpoints.1 as f64) * 2.0 * std::f64::consts::PI / dy
//            };
//            phi_k[[i, j]] *= -kx*kx - ky*ky;
//        }
//    }
//
//    // 3. Compute inverse FFT
//    let mut laplacian = phi_k;
//    ifft.process(laplacian.as_slice_mut().unwrap());
//
//    // Scale the result after IFFT
//    let scale_factor = (gridpoints.0 * gridpoints.1) as f64;
//    laplacian.mapv_inplace(|val| val / Complex::new(scale_factor, 0.0));
//
//    laplacian

}

fn gross_pitaevskii_hamiltonian(phi: &Array2<Complex<f64>>, potential: &Array2<f64>, interaction_strength: &f64, mass: &f64, laplacian: &Array2<Complex<f64>>) -> (Array2<Complex<f64>>, Array2<Complex<f64>>) {

   write_phi(&phi, "./output/debug/gpe_before_ke.txt").expect("Failed to write to file");

   // let kinetic_energy: Array2<Complex<f64>> = laplacian.mapv(|x| x * (-REDUCED_PLANCK_CONSTANT * REDUCED_PLANCK_CONSTANT / (2.0 * mass)));

   let kinetic_energy: Array2<Complex<f64>> = laplacian.mapv(|x| x / 2.0);
   let potential_energy: Array2<Complex<f64>> = potential.mapv(|p| Complex::new(p, 0.0));

   write_phi(&phi, "./output/debug/gpe_phi.txt").expect("Failed to write to file");

   let interaction_energy = phi.mapv(|phi| interaction_strength * phi.norm_sqr() * Complex::new(1.0, 0.0));

   write_phi(&kinetic_energy, "./output/debug/energy_kinetic.txt").expect("Failed to write to file");
   write_potential(&potential, "./output/debug/energy_potential.txt").expect("Failed to write to file");
   write_phi(&interaction_energy, "./output/debug/energy_interaction.txt").expect("Failed to write to file");

   (kinetic_energy, potential_energy + interaction_energy)
}

fn sgpe(phi: &Array2<Complex<f64>>, gamma: &f64, mu: &f64, gp_hamiltonian: &(Array2<Complex<f64>>, Array2<Complex<f64>>)) -> Array2<Complex<f64>> {
   let i = Complex::new(0.0, 1.0);

   let rhs =  -(i + gamma) * ((&gp_hamiltonian.1 - *mu) * phi - &gp_hamiltonian.0);

   write_phi(&rhs, "./output/debug/rhs.txt").expect("Failed to write to file");

   rhs
}

fn main() {

   let rb87 = Species {
     atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
   };

   let rb85 = Species {
     atomic_mass: 84.9117 * ATOMIC_MASS_UNIT,
   };

   let atomic_species = &rb87;

   let mut trap = Trap {
     frequency_x: 2.0*PI*30.0,
     frequency_y: 2.0*PI*30.0,
     frequency_z: 2.0*PI*300.0,
     depth: 0.0,
   };

   let scalings = Scalings{
     temperature: BOLTZMANN_CONSTANT/(REDUCED_PLANCK_CONSTANT*trap.frequency_x),
     length_x: (REDUCED_PLANCK_CONSTANT/(atomic_species.atomic_mass * trap.frequency_x)).sqrt(),
     length_y: (REDUCED_PLANCK_CONSTANT/(atomic_species.atomic_mass * trap.frequency_x)).sqrt(),
     length_z: (REDUCED_PLANCK_CONSTANT/(atomic_species.atomic_mass * trap.frequency_z)).sqrt(),
     time: trap.frequency_x,
   };

   trap.depth = 30e-9*scalings.temperature; // define trap depth in nK after energy scalings are introduced. ensure trap.depth > condensate.chemical_potential.

   let condensate = Condensate {
     temperature: 0.0,
     gamma: 1.0,
     scattering_length: 100.0 * BOHR_RADIUS,
     chemical_potential: 20e-9*scalings.temperature // Kelvin
   };

   // assert!((condensate.chemical_potential/scalings.temperature).abs() <  condensate.temperature, "|μ|<< T is not satisfied (see Castin, et al., JMO 47 2000 pp. 2671-2695.");

   let interaction_strength = condensate.interaction_strength(atomic_species.atomic_mass, scalings.length_x, trap.frequency_x);

   let simulation = Simulation {
     grid_size: 10.0e-6/scalings.length_x,
     gridpoints: (256,256),
     timesteps: 10_000,
     step_size: 1.0e-3,
     runs: 1_000,
     noise_realisations: 1
   };

   println!("{:e}, {:e}", simulation.step_size, 0.5*(simulation.grid_size/simulation.gridpoints.0 as f64));
   assert!(simulation.step_size < 0.5*(simulation.grid_size/simulation.gridpoints.0 as f64), "CFL condition violated. Check that dx < 0.5 (dt)^2.");

   let x = Array1::linspace(-simulation.grid_size, simulation.grid_size, simulation.gridpoints.0);

   let y = Array1::linspace(-simulation.grid_size, simulation.grid_size, simulation.gridpoints.1);

   let field = Field {
     phi: Array2::from_elem(simulation.gridpoints, Complex::new(0.0, 0.0)),
   };

   println!("{:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
       condensate.gamma,
       BOLTZMANN_CONSTANT,
       condensate.temperature,
       simulation.step_size,
       REDUCED_PLANCK_CONSTANT,
       trap.frequency_x,
       simulation.grid_size,
       simulation.gridpoints.0,
       simulation.gridpoints.1);

   let noise_magnitude: f64 =
   (2.0 * condensate.gamma * BOLTZMANN_CONSTANT * condensate.temperature * simulation.step_size
     / (REDUCED_PLANCK_CONSTANT * trap.frequency_x
         * (simulation.grid_size / simulation.gridpoints.0 as f64)
         * (simulation.grid_size / simulation.gridpoints.1 as f64)))
     .sqrt(); // the magnitude of the noise is chosen as a square root such that <\eta^* \eta> gives the right magnitude. The dt and 1/dx term comes from the delta functions.

     println!("Noise mag: {:e}", noise_magnitude);

   // let noise_magnitude: f64 =
   // (2.0 * condensate.gamma * condensate.temperature * simulation.step_size
   //   / (REDUCED_PLANCK_CONSTANT * trap.frequency_x
   //       * (simulation.grid_size / simulation.gridpoints.0 as f64)
   //       * (simulation.grid_size / simulation.gridpoints.1 as f64)))
   //   .sqrt(); // the magnitude of the noise is chosen as a square root such that <\eta^* \eta> gives the right magnitude. The dt and 1/dx term comes from the delta functions.

   // Write parameters to file
   let result = write_params(&rb87, &rb85, &atomic_species, &trap, &scalings, &condensate, &simulation, &interaction_strength, &noise_magnitude, "./output/params.txt");


   if let Err(e) = result {
     eprintln!("Error writing parameters to file: {}", e);
   }

   let result = write_coords(&x, "./output/x.txt");
   if let Err(e) = result {
     eprintln!("Error writing x to file: {}", e);
   }

   let result = write_coords(&y, "./output/y.txt");
   if let Err(e) = result {
     eprintln!("Error writing y to file: {}", e);
   }

   // Calculate the 2d potential
   let potential = potential(&x, &y, &simulation.gridpoints, &trap);

   let mut phi: Array2<Complex<f64>> = Array2::ones((simulation.gridpoints.0, simulation.gridpoints.1));

   write_phi(&phi, "./output/debug/phi0.txt").expect("Failed to write to file");;

   // Calculate Laplacian of current phi
   let laplacian = laplacian_2d(&phi, &simulation.step_size, &simulation.step_size, &simulation.gridpoints);

   write_phi(&laplacian, "./output/debug/laplacian.txt").expect("Failed to write Laplacian to file");

   for run_id in 0..simulation.noise_realisations {
       let hamiltonian = gross_pitaevskii_hamiltonian(&phi, &potential, &interaction_strength, &rb87.atomic_mass, &laplacian);

       phi = rkf45::runge_kutta_2d(
           0.0,
           phi,
           &simulation.step_size,
           &(simulation.timesteps as isize),
           &(run_id as isize),
           &|_t, phi| sgpe(
               &phi,
               &condensate.gamma,
               &condensate.chemical_potential,
               &hamiltonian,
           ),
           &simulation.gridpoints,
           &noise_magnitude,
           (simulation.grid_size / simulation.gridpoints.0 as f64)
       );
   }


}

// pub fn runge_kutta_2d(t0: f64, y0: Array2<Complex<f64>>, h: f64, n_steps: usize, run_id: usize, f: &dyn Fn(f64, Array2<Complex<f64>>) -> Array2<Complex<f64>>) -> Array2<Complex<f64>> {
