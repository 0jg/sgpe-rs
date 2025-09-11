//! Utility functions for file I/O operations in the SGPE simulation.
//!
//! This module provides functions for writing various types of data to files,
//! including potential energy, wave function data, coordinates, and simulation parameters.
//! These functions save the simulation results and allowing for
//! post-processing and analysis.

use ndarray::{Array1, Array2};
use num::complex::Complex;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::types::{Condensate, Scalings, Simulation, Species, Trap};

/// Writes the potential energy data to a file.
///
/// This function is used to save the potential energy landscape of the trap.
///
/// # Arguments
///
/// * `data` - A 2D array of complex numbers representing the potential energy.
/// * `filename` - The name of the file to write the data to.
///
/// # Returns
///
/// A `Result` indicating success or an I/O error.
pub fn write_potential(data: &Array2<Complex<f64>>, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for row in data.genrows() {
        for (i, &val) in row.iter().enumerate() {
            // Extract the real part of the complex number
            let real_val = val.re;

            if i > 0 {
                write!(writer, ",")?;
            }
            write!(writer, "{}", real_val)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Writes a 2D array of real numbers to a file.
///
/// This function is useful for saving real-valued data, such as the density
/// distribution of the condensate or other physical observables.
///
/// # Arguments
///
/// * `array` - A 2D array of real numbers.
/// * `filename` - The name of the file to write the data to.
///
/// # Returns
///
/// A `Result` indicating success or an I/O error.
pub fn write_real_2d(array: &Array2<f64>, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for row in array.genrows() {
        for (i, value) in row.iter().enumerate() {
            if i > 0 {
                write!(writer, ",")?;
            }
            write!(writer, "{}", value)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Writes the complex wavefunction data to a file.
///
/// This function saves the wavefunction ψ(r,t) of the condensate, which is the
/// primary object of study in the SGPE simulation. The wavefunction contains
/// information about both the density and phase of the condensate.
///
/// # Arguments
///
/// * `noise` - A 2D array of complex numbers representing the wavefunction.
/// * `filename` - The name of the file to write the data to.
///
/// # Returns
///
/// A `Result` indicating success or an I/O error.
pub fn write_phi(noise: &Array2<Complex<f64>>, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for row in noise.genrows() {
        for (i, complex_num) in row.iter().enumerate() {
            if i > 0 {
                write!(writer, ",")?;
            }
            write!(writer, "{},{}", complex_num.re, complex_num.im)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Writes coordinate data to a file.
///
/// This function saves the spatial coordinates used in the simulation grid.
/// These coordinates are essential for interpreting the spatial distribution
/// of the condensate and other physical quantities.
///
/// # Arguments
///
/// * `data` - A 1D array of real numbers representing coordinates.
/// * `filename` - The name of the file to write the data to.
///
/// # Returns
///
/// A `Result` indicating success or an I/O error.
pub fn write_coords(data: &Array1<f64>, filename: &str) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for (i, &val) in data.iter().enumerate() {
        if i > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    Ok(())
}

/// Writes simulation parameters to a file.
///
/// This function saves all relevant parameters of the SGPE simulation, including
/// atomic species properties, trap characteristics, scaling factors, condensate
/// properties, and numerical simulation settings.
///
/// # Arguments
///
/// * `rb87` - Properties of Rubidium-87.
/// * `rb85` - Properties of Rubidium-85.
/// * `atomic_species` - The chosen atomic species for the simulation.
/// * `trap` - Trap parameters.
/// * `scalings` - Scaling factors used in the simulation.
/// * `condensate` - Condensate properties.
/// * `simulation` - Numerical simulation settings.
/// * `interaction_strength` - The interaction strength g.
/// * `noise_magnitude` - The magnitude of the noise term in the SGPE.
/// * `filename` - The name of the file to write the parameters to.
///
/// # Returns
///
/// A `Result` indicating success or an I/O error.
pub fn write_params(
    rb87: &Species,
    rb85: &Species,
    atomic_species: &Species,
    trap: &Trap,
    scalings: &Scalings,
    condensate: &Condensate,
    simulation: &Simulation,
    interaction_strength: &f64,
    noise_magnitude: &f64,
    filename: &Path,
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "rb87 atomic_mass: {:.4e}", rb87.atomic_mass)?;
    writeln!(writer, "rb85 atomic_mass: {:.4e}", rb85.atomic_mass)?;
    writeln!(
        writer,
        "chosen species mass: {:.4e}",
        atomic_species.atomic_mass
    )?;

    writeln!(writer, "trap frequency_x: {:.4e}", trap.frequency_x)?;
    writeln!(writer, "trap frequency_y: {:.4e}", trap.frequency_y)?;
    writeln!(writer, "trap frequency_z: {:.4e}", trap.frequency_z)?;
    writeln!(writer, "trap depth: {:.4e}", trap.depth.unwrap_or(0.0))?;

    writeln!(writer, "scalings temperature: {:.4e}", scalings.temperature)?;
    writeln!(writer, "scalings length_x: {:.4e}", scalings.length_x)?;
    writeln!(writer, "scalings length_y: {:.4e}", scalings.length_y)?;
    writeln!(writer, "scalings length_z: {:.4e}", scalings.length_z)?;
    writeln!(writer, "scalings time: {:.4e}", scalings.time)?;
    writeln!(
        writer,
        "scalings chemical potential: {:.4e}",
        scalings.chemical_potential
    )?;

    writeln!(
        writer,
        "condensate temperature: {:.4e}",
        condensate.temperature
    )?;
    writeln!(writer, "condensate gamma: {:.4e}", condensate.gamma)?;
    writeln!(
        writer,
        "condensate scattering_length: {:.4e}",
        condensate.scattering_length
    )?;
    writeln!(
        writer,
        "condensate chemical_potential: {:.4e}",
        condensate.chemical_potential
    )?;

    writeln!(writer, "simulation grid_size: {:.4e}", simulation.grid_size)?;
    writeln!(writer, "simulation gridpoints: {:?}", simulation.gridpoints)?;
    writeln!(
        writer,
        "simulation step_size: ({:.4e}, {:.4e})",
        simulation.step_size.0, simulation.step_size.1
    )?;
    writeln!(writer, "simulation timesteps: {:.4e}", simulation.timesteps)?;
    writeln!(writer, "simulation timestep: {:.4e}", simulation.timestep)?;
    writeln!(writer, "simulation runs: {:.4e}", simulation.runs)?;

    writeln!(writer, "interaction strength: {:.4e}", interaction_strength)?;
    writeln!(writer, "noise magnitude: {:?}", noise_magnitude)?;

    writeln!(writer, "EOF")?;

    Ok(())
}