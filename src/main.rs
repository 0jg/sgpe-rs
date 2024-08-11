//! Main entry point for the SGPE (Stochastic Gross–Pitaevskii Equation) simulation.
//!
//! This module implements the SGPE, which describes the dynamics of a Bose-Einstein condensate
//! at finite temperature. The SGPE is given by:
//!
//! \[
//! i\hbar \frac{\partial \psi}{\partial t} = (1-i\gamma)\left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}) + g|\psi|^2 - \mu\right]\psi + \eta(\mathbf{r},t)
//! \]
//!
//! where \(\psi\) is the condensate wavefunction, \(\gamma\) is the damping parameter,
//! \(V(\mathbf{r})\) is the trapping potential, \(g\) is the interaction strength,
//! \(\mu\) is the chemical potential, and \(\eta(\mathbf{r},t)\) is a complex Gaussian noise term.

use ndarray::{Array1, Array2};
use num::complex::Complex;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::env;
use std::path::PathBuf;

use spgpe::types::*;
use spgpe::constants::*;
use spgpe::utils::*;
use spgpe::rkf45;

use rand::Rng;

/// Main function for the SGPE simulation
///
/// This function performs the following steps:
/// 1. Parse command-line arguments
/// 2. Set up simulation parameters
/// 3. Initialize the atomic species and trap configuration
/// 4. Calculate scaling factors
/// 5. Set up the condensate parameters
/// 6. Initialize the simulation grid
/// 7. Run the simulation for multiple noise realisations
fn main() {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <chemical_potential> <temperature> <save_every_step> <noise_realisations>", args[0]);
        std::process::exit(1);
    }
    
    /// Parse the save_every_step argument
    /// This determines whether the simulation state is saved at every time step or only at the end
    let save_every_step = env::args().nth(3)
        .expect("Expected a third argument for save behaviour (true for saving every step, false for only saving the thermalised state)")
        .parse::<bool>()
        .expect("Expected a boolean value for save behaviour argument");
      
    /// Parse the number of noise realisations
    /// This determines how many independent simulations are run to account for stochastic effects
    let noise_realisations: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Invalid number for noise realisations");
        std::process::exit(1);
    });

    /// Initialize atomic species
    /// Here we define the properties of Rubidium-87 and Rubidium-85
    let rb87 = Species {
        atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
    };

    let rb85 = Species {
        atomic_mass: 84.9117 * ATOMIC_MASS_UNIT,
    };

    /// Select the atomic species for the simulation (Rb-87 in this case)
    let atomic_species = &rb87;
   
    /// Create a random generator for trap frequency offsets
    /// This allows for slight variations in the trapping potential between simulations
    let mut rng = rand::thread_rng();
    let offset_x: f64 = rng.gen_range(-2.0..=2.0);
    let offset_y: f64 = rng.gen_range(-2.0..=2.0);

    /// Define a harmonic trap
    /// The potential is given by V(r) = 1/2 * m * (ωx^2 * x^2 + ωy^2 * y^2 + ωz^2 * z^2)
    let mut trap = Trap {
        trap_type: TrapType::Harmonic,
        frequency_x: 2.0 * PI * 25.0,  // ωx = 2π * 25 Hz
        frequency_y: 2.0 * PI * 25.0,  // ωy = 2π * 25 Hz
        frequency_z: 2.0 * PI * 600.0, // ωz = 2π * 600 Hz (tight confinement in z-direction)
        depth: None,
        ring_radius: None,
        trap_radius: None,
    };

    /// Calculate scaling factors
    /// These factors are used to non-dimensionalize the SGPE
    let scalings = Scalings {
        temperature: BOLTZMANN_CONSTANT / (REDUCED_PLANCK_CONSTANT * trap.frequency_x),
        length_x: (REDUCED_PLANCK_CONSTANT / (atomic_species.atomic_mass * trap.frequency_x)).sqrt(),
        length_y: (REDUCED_PLANCK_CONSTANT / (atomic_species.atomic_mass * trap.frequency_x)).sqrt(),
        length_z: (REDUCED_PLANCK_CONSTANT / (atomic_species.atomic_mass * trap.frequency_z)).sqrt(),
        time: trap.frequency_x,
        chemical_potential: 1.0 / (REDUCED_PLANCK_CONSTANT * trap.frequency_x)
    };
   
    /// Set trap parameters
    /// These parameters are used for more complex trap geometries (e.g., toroidal traps)
    trap.depth = Some(60e-9 * scalings.temperature);
    trap.ring_radius = Some(40e-6 / scalings.length_x);
    trap.trap_radius = Some(20e-6 / scalings.length_x);

    /// Set up condensate parameters
    /// These parameters define the properties of the Bose-Einstein condensate
    let condensate = Condensate {
        temperature: args[2].parse::<f64>().unwrap() * 1e-9 * scalings.temperature,
        gamma: 0.1,  // Dimensionless damping parameter
        scattering_length: 100.0 * BOHR_RADIUS,  // s-wave scattering length
        chemical_potential: args[1].parse::<f64>().unwrap() * REDUCED_PLANCK_CONSTANT * trap.frequency_x * scalings.chemical_potential
    };

    /// Calculate interaction strength
    /// g = 4πℏ^2a_s/m, where a_s is the s-wave scattering length
    let interaction_strength = condensate.interaction_strength(atomic_species.atomic_mass, scalings.length_z, trap.frequency_x);

    /// Set up simulation parameters
    let mut simulation = Simulation {
        grid_size: 80e-6 / scalings.length_x,  // Size of the simulation box in scaled units
        gridpoints: (128, 128),  // Number of grid points in each dimension
        step_size: (0.0, 0.0),  // Will be calculated later
        timesteps: 100_000,  // Number of time steps in the simulation
        timestep: 1.0e-3,  // Size of each time step in scaled units
        runs: 1_000,  // Number of independent runs (not used in this script)
        noise_realisations: noise_realisations as i64  // Number of stochastic realisations
    };

    /// Calculate the spatial step size
    simulation.step_size = (simulation.grid_size / simulation.gridpoints.0 as f64, simulation.grid_size / simulation.gridpoints.0 as f64);

    /// Check CFL (Courant-Friedrichs-Lewy) condition
    /// This condition ensures numerical stability of the simulation
    assert!(simulation.timestep < 0.5 * simulation.step_size.0, "CFL condition violated. Check that dx < 0.5 (dt)^2.");

    /// Initialize simulation grid
    let x = Array1::linspace(-simulation.grid_size, simulation.grid_size, simulation.gridpoints.0);
    let y = Array1::linspace(-simulation.grid_size, simulation.grid_size, simulation.gridpoints.1);

    /// Calculate noise magnitude
    /// The noise magnitude is related to the temperature and damping of the system
    let noise_magnitude: f64 = (2.0 * condensate.gamma * condensate.temperature * simulation.timestep / (simulation.step_size.0 * simulation.step_size.1)).sqrt();

    /// Write coordinates to file
    if let Err(e) = write_coords(&x, "./src/x.txt") {
        eprintln!("Error writing x to file: {}", e);
    }
    if let Err(e) = write_coords(&y, "./src/y.txt") {
        eprintln!("Error writing y to file: {}", e);
    }

    /// Print estimated peak density
    /// This is calculated as n_peak ≈ μ / g, where μ is the chemical potential and g is the interaction strength
    println!("Estimated peak density is of the order of {:.3e}", condensate.chemical_potential / interaction_strength);

    /// Determine if averaging should occur
    let is_averaging = noise_realisations > 1;
   
    /// Convert trap type to string for file naming
    let trap_type_str = match trap.trap_type {
        TrapType::Harmonic => "harmonic",
        TrapType::Toroidal => "toroidal",
        TrapType::Cigar => "cigar",
    };

    /// Run simulation for multiple noise realisations
    /// This is parallelized using Rayon
    (0..simulation.noise_realisations).into_par_iter().for_each(|run_id| {
        /// Set up output directory
        let parent_dir = if is_averaging {
            format!(
                "./output.tmp/ensembles/{}/{}_{}/",
                trap_type_str,
                args[1].parse::<f64>().unwrap(), // mu
                args[2].parse::<f64>().unwrap()  // T
            )
        } else {
            format!(
                "./output.tmp/{}/{}_{}",
                trap_type_str,
                args[1].parse::<f64>().unwrap(), // mu
                args[2].parse::<f64>().unwrap()  // T
            )
        };
       
        let dir = format!("{}/{}", parent_dir, run_id);
       
        let path = std::path::Path::new(&dir);
        if !path.exists() {
            std::fs::create_dir_all(&path).expect("Failed to create directory");
        }
       
        /// Write simulation parameters to file
        let params_path: PathBuf = [parent_dir.clone(), "params.txt".to_string()].iter().collect();
        let result = write_params(
            &rb87, &rb85, &atomic_species, &trap, &scalings, &condensate, 
            &simulation, &interaction_strength, &noise_magnitude, &params_path
        );
       
        if let Err(e) = result {
            eprintln!("Error writing parameters to file: {}", e);
        }
   
        /// Run the SGPE simulation using the Runge-Kutta method
        let phi = rkf45::runge_kutta_2d(
            0.0,  // Initial time
            Array2::zeros((simulation.gridpoints.0, simulation.gridpoints.1)),  // Initial wavefunction (all zeros)
            &(run_id as isize),
            &noise_magnitude,
            &interaction_strength,
            &simulation,
            &trap,
            &condensate,
            &x,
            &y,
            args[1].parse::<f64>().unwrap(), // mu
            args[2].parse::<f64>().unwrap(), // T
            args[3].parse::<bool>().unwrap(), // save_every_step
            &dir,
        );
    });
}