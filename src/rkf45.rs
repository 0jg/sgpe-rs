//! Implements the Runge-Kutta fourth-order method for SGPE simulations.

use super::constants::*;
use super::types::*;
use super::utils::*;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::E;

/// Calculates the harmonic potential.
///
/// \[V(x,y) = \frac{1}{2}m(\omega_x^2 x^2 + \omega_y^2 y^2)\]
pub fn harmonic_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    // Calculate aspect ratios
    let aspect_ratio_x = trap.frequency_x / trap.frequency_z;
    let aspect_ratio_y = trap.frequency_y / trap.frequency_z;

    // Compute the potential using broadcasting
    let potential_x =
        0.5 * aspect_ratio_x.powi(2) * x.mapv(|x| x.powi(2)).into_shape((x.len(), 1)).unwrap();
    let potential_y =
        0.5 * aspect_ratio_y.powi(2) * y.mapv(|y| y.powi(2)).into_shape((1, y.len())).unwrap();

    // Combine x and y potentials
    let potential = potential_x + potential_y;

    // Convert the combined potential to Complex<f64>
    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Calculates the toroidal potential.
///
/// \[V(r) = V_0\left(1 - e^{-\frac{(r-R)^2}{2\sigma^2}}\right)\]
pub fn toroidal_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    // Unwrap the toroidal trap parameters
    let depth = trap.depth.expect("Depth is required for a toroidal trap");
    let ring_radius = trap
        .ring_radius
        .expect("Ring radius is required for a toroidal trap");
    let trap_radius = trap
        .trap_radius
        .expect("Trap radius is required for a toroidal trap");

    // Create a grid of x and y values
    let x_grid = x.broadcast((y.len(), x.len())).unwrap();
    let y_grid = y.broadcast((x.len(), y.len())).unwrap().reversed_axes();

    // Compute rho for each pair in the grid
    let rho = (&x_grid * &x_grid + &y_grid * &y_grid).mapv(f64::sqrt);

    // Compute the potential using the given equation
    let potential = depth
        * (1.0
            - (-1.0 / trap_radius.powi(2) * (&rho - ring_radius).mapv(|rho_r| rho_r.powi(2)))
                .mapv(f64::exp));

    // Convert the potential to Complex<f64>
    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Calculates the cigar-shaped potential.
pub fn cigar_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    // Calculate aspect ratios
    let aspect_ratio_x = trap.frequency_x / trap.frequency_z;
    let aspect_ratio_y = trap.frequency_y / trap.frequency_z;

    // Compute the potential using broadcasting, similar to the harmonic potential
    let potential_x =
        0.5 * aspect_ratio_x.powi(2) * x.mapv(|x| x.powi(2)).into_shape((x.len(), 1)).unwrap();
    let potential_y =
        0.5 * aspect_ratio_y.powi(2) * y.mapv(|y| y.powi(2)).into_shape((1, y.len())).unwrap();

    // Combine x and y potentials
    let potential = potential_x + potential_y;

    // Convert the combined potential to Complex<f64>
    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Selects and calculates the appropriate potential based on trap type.
pub fn calculate_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    match trap.trap_type {
        TrapType::Harmonic => harmonic_potential(x, y, trap),
        TrapType::Toroidal => toroidal_potential(x, y, trap),
        TrapType::Cigar => cigar_potential(x, y, trap),
    }
}

/// Computes the Laplacian using a second-order central difference method.
pub fn laplacian(f: &Array2<Complex<f64>>, simulation: &Simulation) -> Array2<Complex<f64>> {
    let mut laplacian = Array2::zeros(f.raw_dim());

    // Central differences formula (2nd-order) for the Laplacian
    for i in 0..simulation.gridpoints.0 {
        for j in 0..simulation.gridpoints.1 {
            let ip1 = (i + 1) % simulation.gridpoints.0;
            let im1 = (i + simulation.gridpoints.0 - 1) % simulation.gridpoints.0;
            let jp1 = (j + 1) % simulation.gridpoints.1;
            let jm1 = (j + simulation.gridpoints.1 - 1) % simulation.gridpoints.1;

            laplacian[[i, j]] = (f[[ip1, j]] + f[[im1, j]] + f[[i, jp1]] + f[[i, jm1]]
                - 4.0 * f[[i, j]])
                / simulation.step_size.0.powi(2);
        }
    }

    laplacian
}

/// Calculates the Gross-Pitaevskii Hamiltonian.
///
/// \[H = -\frac{\hbar^2}{2m}\nabla^2 + V(r) + g|\psi|^2\]
pub fn gross_pitaevskii_hamiltonian(
    phi: &Array2<Complex<f64>>,
    interaction_strength: &f64,
    simulation: &Simulation,
    trap: &Trap,
    x: &Array1<f64>,
    y: &Array1<f64>,
) -> (
    Array2<Complex<f64>>,
    Array2<Complex<f64>>,
    Array2<Complex<f64>>,
) {
    write_phi(&phi, "./src/debug/gpe_before_ke.txt").expect("Failed to write to file");

    let kinetic_energy: Array2<Complex<f64>> = laplacian(phi, simulation) * (-0.5);
    let potential_energy: Array2<Complex<f64>> = calculate_potential(&x, &y, &trap);

    let interaction_energy =
        phi.mapv(|phi| interaction_strength * phi.norm_sqr() * Complex::new(1.0, 0.0));

    write_phi(&kinetic_energy, "./src/debug/energy_kinetic.txt").expect("Failed to write to file");
    write_potential(&potential_energy, "./src/debug/energy_potential.txt")
        .expect("Failed to write to file");
    write_phi(&interaction_energy, "./src/debug/energy_interaction.txt")
        .expect("Failed to write to file");

    (kinetic_energy, potential_energy, interaction_energy)
}

/// Computes the right-hand side of the SGPE.
///
/// \[i\hbar\frac{\partial\psi}{\partial t} = (1-i\gamma)(H\psi - \mu\psi)\]
pub fn sgpe(
    phi: &Array2<Complex<f64>>,
    hamiltonian: &(
        Array2<Complex<f64>>,
        Array2<Complex<f64>>,
        Array2<Complex<f64>>,
    ),
    condensate: &Condensate,
) -> Array2<Complex<f64>> {
    let i = Complex::new(0.0, 1.0);

    let rhs = (1.0 / i)
        * (Complex::new(1.0, 0.0) - i * condensate.gamma)
        * ((&hamiltonian.1 + &hamiltonian.2 - condensate.chemical_potential) * phi
            + &hamiltonian.0);

    write_phi(&rhs, "./src/debug/rhs.txt").expect("Failed to write to file");

    rhs
}

/// Generates Wiener noise for the stochastic term.
pub fn generate_wiener_noise(gridpoints: &(usize, usize)) -> Array2<f64> {
    // Create a random number generator
    let mut rng = rand::thread_rng();

    // Generate the Wiener noise matrix
    let wiener_noise = Array2::from_shape_fn(*gridpoints, |_| StandardNormal.sample(&mut rng));

    write_real_2d(&wiener_noise, "./src/debug/wiener_noise.txt").unwrap();

    wiener_noise
}

/// Generates phase noise for the stochastic term.
pub fn generate_phase_noise(gridpoints: &(usize, usize)) -> Array2<f64> {
    // Create a random number generator
    let mut rng = rand::thread_rng();

    // Draw random numbers from a uniform distribution between 0 and 1
    let uniform_dist = rand::distributions::Uniform::new(0.0, 1.0);
    let phase_noise = Array2::from_shape_fn(*gridpoints, |_| uniform_dist.sample(&mut rng));

    write_real_2d(&phase_noise, "./src/debug/phase_noise.txt").unwrap();

    phase_noise
}

/// Calculates the final noise term for the SGPE.
pub fn calculate_noise(
    noise_magnitude: f64,
    wiener_noise: &Array2<f64>,
    phase_noise: &Array2<f64>,
) -> Array2<Complex<f64>> {
    // Convert wiener_noise to Complex<f64>
    let wiener_noise_complex = wiener_noise.mapv(|x| Complex::new(x, 0.0));

    // Convert phase_noise to Complex<f64>
    let phase_noise_complex = phase_noise.mapv(|theta| Complex::new(0.0, 2.0 * PI * theta).exp());

    // Calculate the final noise array using the provided formula
    let noise = Complex::new(noise_magnitude, 0.0) * &wiener_noise_complex * &phase_noise_complex;

    write_phi(&noise, "./src/noise.txt").expect("Failed to write to file");

    noise
}

/// Performs a single Runge-Kutta step for the SGPE.
pub fn runge_kutta_step_2d(
    y: &Array2<Complex<f64>>,
    h: &f64,
    gridpoints: &(usize, usize),
    noise_magnitude: f64,
    interaction_strength: &f64,
    simulation: &Simulation,
    trap: &Trap,
    condensate: &Condensate,
    x_pos: &Array1<f64>,
    y_pos: &Array1<f64>,
) -> Array2<Complex<f64>> {
    // Generate Wiener noise from a normal distribution
    let wiener_noise: Array2<f64> = generate_wiener_noise(gridpoints);

    // Generate phase noise from a uniform distribution
    let phase_noise: Array2<f64> = generate_phase_noise(gridpoints);

    // Calculate the final noise array
    let noise: Array2<Complex<f64>> = calculate_noise(noise_magnitude, &wiener_noise, &phase_noise);

    // Calculate Gross-Pitaevskii Hamiltonian at every step
    let hamiltonian = gross_pitaevskii_hamiltonian(
        &y,
        &interaction_strength,
        &simulation,
        &trap,
        &x_pos,
        &y_pos,
    );

    let k1 = sgpe(y, &hamiltonian, &condensate);
    let k2 = sgpe(
        &(y + Complex::new(h / 2.0, 0.0) * &k1),
        &hamiltonian,
        &condensate,
    );
    let k3 = sgpe(
        &(y + Complex::new(h / 2.0, 0.0) * &k2),
        &hamiltonian,
        &condensate,
    );
    let k4 = sgpe(
        &(y + Complex::new(h / 1.0, 0.0) * &k3),
        &hamiltonian,
        &condensate,
    );

    y + Complex::new(h / 6.0, 0.0) * (k1 + Complex::new(2.0, 0.0) * (k2 + k3) + k4) + noise
}

/// Performs the full Runge-Kutta time evolution for the SGPE.
pub fn runge_kutta_2d(
    t0: f64,
    y0: Array2<Complex<f64>>,
    run_id: &isize,
    noise_magnitude: &f64,
    interaction_strength: &f64,
    simulation: &Simulation,
    trap: &Trap,
    condensate: &Condensate,
    x_pos: &Array1<f64>,
    y_pos: &Array1<f64>,
    unscaled_chemical_potential: f64,
    unscaled_temperature: f64,
    save_every_step: bool,
    dir: &String,
) -> Array2<Complex<f64>> {
    let mut _t = t0;
    let mut y = y0;
    let mut norm: f64;
    let mut previous_norm = y.iter().map(|&c| c.norm_sqr()).sum::<f64>()
        * simulation.step_size.0
        * simulation.step_size.1;
    
    let path = std::path::Path::new(&dir);

    let mut consecutive_small_changes = 0;
    let mut previous_norm = 1.0;

    for i in 0..simulation.timesteps {
        y = runge_kutta_step_2d(
            &y,
            &simulation.timestep,
            &simulation.gridpoints,
            *noise_magnitude,
            interaction_strength,
            simulation,
            trap,
            condensate,
            x_pos,
            y_pos,
        );
        _t += simulation.timestep;
        norm = y.iter().map(|&c| c.norm_sqr()).sum::<f64>()
            * simulation.step_size.0
            * simulation.step_size.1;

        // Conditional saving logic
        if (save_every_step && i % 100 == 0) || i == simulation.timesteps - 1 {
            let filename = format!("{}/{}.txt", dir, i);
            write_phi(&y, &filename).expect("Failed to write phi to file");
        }

        let relative_difference = (norm - previous_norm).abs() / previous_norm;
        
        if relative_difference < 1e-4 {
            consecutive_small_changes += 1; // Increment if relative difference is small enough
        } else {
            consecutive_small_changes = 0; // Reset if it's not
        }

        // Early stopping condition
        if consecutive_small_changes >= 5 {
            if !save_every_step {
                // Save the final state only if we are not saving every step
                let filename = format!("{}/{}.txt", dir, i);
                write_phi(&y, &filename).expect("Failed to write phi to file");
            }
            break;
        }

        previous_norm = norm; // Update the previous norm value for the next iteration
    }

    y
}