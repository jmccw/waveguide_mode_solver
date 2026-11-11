// Semivectorial finite-difference mode solver, quasi-TE (E_x representation)
// Ref: Kawano & Kitoh, "Introduction to Optical Waveguide Analysis", Ch. 4.
#ifndef WG_SOLVER_H
#define WG_SOLVER_H
 
#include <complex>
#include <string>
#include <utility>
#include <vector>
 
#include <Eigen/Sparse>
 
// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using Grid1D = std::vector<double>;
using Grid2D = std::vector<std::vector<double>>;
 
// Result of the eigen solve, sorted by n_eff^2 (descending) via `order`.
struct ModeResult {
    // bool               converged = false;
    Eigen::VectorXcd   eigenvalues;
    Eigen::MatrixXcd   eigenvectors;
    std::vector<std::pair<double, int>> order;   // (n_eff^2, column index)
};
 
// ---------------------------------------------------------------------------
// Structure parameters (C++17 inline constants: one definition, safe in headers)
// ---------------------------------------------------------------------------
//~ inline constexpr double core_index = 3.5;
//~ inline constexpr double clad_index = 3.1693;   // InP cladding
//~ inline constexpr double core_size  = 0.4e-6;   // full core width (x and y)
//~ inline constexpr double block_size = 3.4e-6;   // full computational window
 
//~ inline constexpr double n_SiO2 = 1.444;
//~ inline constexpr double n_air  = 1.0;
 
// ---------------------------------------------------------------------------
// Index helpers: interior nodes only (p = 1..M_x-2, q = 1..M_y-2).
// The outer ring is the Dirichlet boundary (E = 0), not part of the system.
// ---------------------------------------------------------------------------
//~ inline bool is_unknown(int p, int q, int M_x, int M_y)
//~ {
    //~ return p >= 1 && p <= M_x - 2 && q >= 1 && q <= M_y - 2;
//~ }
 
//~ inline int index_of(int p, int q, int M_y)
//~ {
    //~ return (p - 1) * (M_y - 2) + (q - 1);
//~ }
 
// ---------------------------------------------------------------------------
// Mesh construction
// ---------------------------------------------------------------------------
// Square quantum-well core in InP cladding (cell-centred mesh).
void initialise_mesh_core_WQ(Grid2D& spatial_pos, Grid2D& epsilon_r,
                             int M_x, int M_y);
 
// Rib waveguide; reads refractive_index_profile.dat (y [m], n).
void initialise_mesh_rib_WG(
    std::vector<std::vector<double>>& spatial_pos,
    std::vector<std::vector<double>>& epsilon_r,
    double dx,
    double dy,
    int& M_x,
    int& M_y, 
    double etch_depth, 
    double wg_width, 
    double window_width, 
    double dielectric_thickness);
 
// ---------------------------------------------------------------------------
// Matrix assembly: A E = n_eff^2 E in k0-normalised coordinates, Eqs. (4.99)-(4.105)
// ---------------------------------------------------------------------------
Eigen::SparseMatrix<double> assemble_matrix(const Grid2D& spatial_pos,
                                            const Grid2D& epsilon_r,
                                            int M_x, int M_y, double k_0);
 
// ---------------------------------------------------------------------------
// Eigen solve (shift-invert Arnoldi about sigma)
// ---------------------------------------------------------------------------
ModeResult solve_modes(const Eigen::SparseMatrix<double>& A,
                       int num_modes, int ncv, double sigma);
 
// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
void print_modes(const ModeResult& result);
 
// Writes mode_<m>.dat: x[um] y[um] Re(E) Im(E) |E|^2, peak normalised to +1.
void export_modes(const ModeResult& result, const Grid2D& spatial_pos,
                  int M_x, int M_y);
 
#endif // WG_SOLVER_H
