use ndarray::{Array1, Array2};
use num::complex::Complex;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::env;
use std::path::PathBuf;

use sgpe::types::*;
use sgpe::constants::*;
use sgpe::utils::*;
use sgpe::rkf45;

use rand::Rng;

fn main() {
   
   // ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();

   // println!("Current dir: {:?}", std::env::current_dir());

   let args: Vec<String> = env::args().collect();
   if args.len() < 4 {
     eprintln!("Usage: {} <chemical_potential> <temperature> <save_every_step> <noise_realisations>", args[0]);
     std::process::exit(1);
   }
    
   let save_every_step = env::args().nth(3)
      .expect("Expected a third argument for save behaviour (true for saving every step, false for only saving the thermalised state)")
      .parse::<bool>()
      .expect("Expected a boolean value for save behaviour argument");
      
   let noise_realisations: usize = args[4].parse().unwrap_or_else(|_| {
     eprintln!("Invalid number for noise realisations");
     std::process::exit(1);
   });

   let rb87 = Species {
     atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
   };

   let rb85 = Species {
     atomic_mass: 84.9117 * ATOMIC_MASS_UNIT,
   };

   let atomic_species = &rb87;
   
   // // Assuming args[4] and args[5] are your frequency inputs in Hz, without the 2 * PI factor
   // let base_freq_x: f64 = args[4].parse().unwrap();
   // let base_freq_y: f64 = args[5].parse().unwrap();
   
   // Create a random generator
   let mut rng = rand::thread_rng();
   
   // Generate random offsets within the range of [-2, 2] for each frequency
   let offset_x: f64 = rng.gen_range(-2.0..=2.0);
   let offset_y: f64 = rng.gen_range(-2.0..=2.0);

   // Define a harmonic trap
   let mut trap = Trap {
     trap_type: TrapType::Harmonic,
     frequency_x: 2.0 * PI * 25.0, // Replace with actual frequency
     frequency_y: 2.0 * PI * 25.0, // Replace with actual frequency
     frequency_z: 2.0 * PI * 600.0, // Replace with actual frequency
     depth: None,
     ring_radius: None,
     trap_radius: None,
   };

   // // Define a toroidal trap
   // let mut trap = Trap {
   //     trap_type: TrapType::Toroidal,
   //     frequency_y: 2.0 * PI * (25.0 + offset_x), // Replace with actual frequency
   //     frequency_x: 2.0 * PI * (25.0 + offset_y), // Replace with actual frequency
   //     frequency_z: 2.0 * PI * 100.0, // Replace with actual frequency
   //     depth: Some(1.0e-4),      // Replace with actual depth value
   //     ring_radius: Some(1.0), // Replace with actual ring radius value
   //     trap_radius: Some(0.5), // Replace with actual trap radius value (usually 0.5 * ring_radius)
   // };
   
   // // Define a cigar trap
   // let mut trap = Trap {
   //     trap_type: TrapType::Cigar,
   //     frequency_y: 2.0 * std::f64::consts::PI * 50.0, // Tighter confinement in y than x
   //     frequency_x: 2.0 * std::f64::consts::PI * 25.0, // Less tight confinement in x
   //     frequency_z: 2.0 * std::f64::consts::PI * 100.0, // Not used directly in 2D but indicative of very tight confinement along z
   //     depth: None, 
   //     ring_radius: None,
   //     trap_radius: None,
   // };


   let scalings = Scalings{
     temperature: BOLTZMANN_CONSTANT/(REDUCED_PLANCK_CONSTANT*trap.frequency_x),
     length_x: (REDUCED_PLANCK_CONSTANT/(atomic_species.atomic_mass * trap.frequency_x)).sqrt(),
     length_y: (REDUCED_PLANCK_CONSTANT/(atomic_species.atomic_mass * trap.frequency_x)).sqrt(),
     length_z: (REDUCED_PLANCK_CONSTANT/(atomic_species.atomic_mass * trap.frequency_z)).sqrt(),
     time: trap.frequency_x,
     chemical_potential: 1.0/(REDUCED_PLANCK_CONSTANT*trap.frequency_x)
   };
   
   trap.depth = Some(60e-9 * scalings.temperature);
   trap.ring_radius = Some(40e-6 / scalings.length_x);
   trap.trap_radius = Some(20e-6 / scalings.length_x); // these values need to be carefully chosen to avoid the condensate from exploding

   // trap.depth = 30e-9*scalings.temperature; // define trap depth in nK after energy scalings are introduced. ensure trap.depth > condensate.chemical_potential.

   let condensate = Condensate {
     temperature: args[2].parse::<f64>().unwrap()*1e-9*scalings.temperature,
     gamma: 0.1,
     scattering_length: 100.0 * BOHR_RADIUS,
     chemical_potential: args[1].parse::<f64>().unwrap()*REDUCED_PLANCK_CONSTANT*trap.frequency_x * scalings.chemical_potential
   };

   // assert!((condensate.chemical_potential/scalings.temperature).abs() <  condensate.temperature, "|μ|<< T is not satisfied (see Y. Castin , R. Dum , E. Mandonnet , A. Minguzzi & I. Carusotto (2000) Coherence properties of a continuous atom laser, Journal of Modern Optics, 47:14-15, 2671-2695, DOI: 10.1080/09500340008232189");

   let interaction_strength = condensate.interaction_strength(atomic_species.atomic_mass, scalings.length_z, trap.frequency_x);

   let mut simulation = Simulation {
     grid_size: 80e-6/scalings.length_x, // TODO: What is the relation between the size of the box and the chemical potential?
     gridpoints: (128,128),
     step_size: (0.0, 0.0),
     timesteps: 100_000, // the thermalisation procedure ends either at convergence
                         // of atom number or after a certain number of time steps have passed.
                         // `timesteps` should be sufficiently high to give the simulation time
                         // for convergence but not too high to avoid never-ending simulations.
     timestep: 1.0e-3,
     runs: 1_000,
     noise_realisations: noise_realisations as i64
   };

   simulation.step_size = (simulation.grid_size / simulation.gridpoints.0 as f64, simulation.grid_size / simulation.gridpoints.0 as f64);

   assert!(simulation.timestep < 0.5*simulation.step_size.0, "CFL condition violated. Check that dx < 0.5 (dt)^2.");

   let x = Array1::linspace(-simulation.grid_size, simulation.grid_size, simulation.gridpoints.0);

   let y = Array1::linspace(-simulation.grid_size, simulation.grid_size, simulation.gridpoints.1);

   let noise_magnitude: f64 =
   (2.0 * condensate.gamma * condensate.temperature * simulation.timestep / (simulation.step_size.0 * simulation.step_size.1)).sqrt(); // the magnitude of the noise is chosen as a square root such that <\eta^* \eta> gives the right magnitude. The dt and 1/dx term comes from the delta functions.

   // println!("Noise mag: {:e}", noise_magnitude);

   // let noise_magnitude: f64 =
   // (2.0 * condensate.gamma * condensate.temperature * simulation.timestep
   //   / (REDUCED_PLANCK_CONSTANT * trap.frequency_x
   //       * (simulation.grid_size / simulation.gridpoints.0 as f64)
   //       * (simulation.grid_size / simulation.gridpoints.1 as f64)))
   //   .sqrt(); // the magnitude of the noise is chosen as a square root such that <\eta^* \eta> gives the right magnitude. The dt and 1/dx term comes from the delta functions.


   let result = write_coords(&x, "./src/x.txt");
   if let Err(e) = result {
     eprintln!("Error writing x to file: {}", e);
   }

   let result = write_coords(&y, "./src/y.txt");
   if let Err(e) = result {
     eprintln!("Error writing y to file: {}", e);
   }

   println!("Estimated peak density is of the order of {:.3e}", condensate.chemical_potential/interaction_strength);

   // let chemical_potentials: Vec<f64> = (0..1).map(|x| condensate.chemical_potential * (x as f64 + 1.0)).collect();
   // let temperatures: Vec<f64> = (0..1).map(|x| condensate.temperature * (x as f64 + 1.0)).collect();
   
   // Determine if averaging should occur based on number of noise realisations and save behavior
   let is_averaging = noise_realisations > 1;
   
   let trap_type_str = match trap.trap_type {
      TrapType::Harmonic => "harmonic",
      TrapType::Toroidal => "toroidal",
      TrapType::Cigar => "cigar",
   };
   // TODO: trap_type_str appears in both main.rs and rkf45.rs, it needs brought into one function which is shared between the two.

   let mut phi: Array2<Complex<f64>>;

   (0..simulation.noise_realisations).into_par_iter().for_each(|run_id| {
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
       
        let params_path: PathBuf = [parent_dir.clone(), "params.txt".to_string()].iter().collect();
        let result = write_params(
            &rb87, &rb85, &atomic_species, &trap, &scalings, &condensate, 
            &simulation, &interaction_strength, &noise_magnitude, &params_path
        );
       
       if let Err(e) = result {
         eprintln!("Error writing parameters to file: {}", e);
       }
   
       let phi = rkf45::runge_kutta_2d(
           0.0,
           Array2::zeros((simulation.gridpoints.0, simulation.gridpoints.1)), // Initialize to zeros for each run
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