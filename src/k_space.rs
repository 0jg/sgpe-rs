
//! Generates the wave vectors for the Fourier-space representation of the grid.

use ndarray::{Array1, Array2};
use rustfft::FftPlanner;
use num_complex::Complex;
use crate::Simulation;

/// Generates the wave vectors (k-space) for the grid.
///
/// # Arguments
///
/// * `simulation` - The simulation parameters, including grid size and grid points.
///
/// # Returns
///
/// A tuple containing:
/// - `kx`: A 1D array of wave vectors in the x-direction.
/// - `ky`: A 1D array of wave vectors in the y-direction.
/// - `k_sq`: A 2D array of the squared magnitude of the wave vectors.
pub fn generate_k_space(simulation: &Simulation) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let grid_points_x = simulation.gridpoints.0;
    let grid_points_y = simulation.gridpoints.1;
    let step_size_x = simulation.step_size.0;
    let step_size_y = simulation.step_size.1;

    // Generate wave vectors for x and y directions
    let kx = Array1::from_shape_fn(grid_points_x, |i| {
        let freq = i as f64 / (grid_points_x as f64 * step_size_x);
        if i > grid_points_x / 2 {
            freq - 1.0 / step_size_x
        } else {
            freq
        }
    }) * 2.0 * std::f64::consts::PI;

    let ky = Array1::from_shape_fn(grid_points_y, |i| {
        let freq = i as f64 / (grid_points_y as f64 * step_size_y);
        if i > grid_points_y / 2 {
            freq - 1.0 / step_size_y
        } else {
            freq
        }
    }) * 2.0 * std::f64::consts::PI;

    // Create k_sq grid
    let kx_sq = kx.mapv(|v| v.powi(2));
    let ky_sq = ky.mapv(|v| v.powi(2));
    let k_sq = kx_sq.into_shape((grid_points_x, 1)).unwrap() + ky_sq;

    (kx, ky, k_sq)
}
