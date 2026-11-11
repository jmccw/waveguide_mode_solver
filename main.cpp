// Jordan Walsh, Integrated Photonics
// jordan.walsh@tyndall.ie
//
// Semivectorial finite-difference mode solver
// Ref: Kawano & Kitoh, "Introduction to Optical Waveguide Analysis", Ch. 4,
//      Eqs. (4.99)-(4.105).

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

//#include "wg_solver.h"

#include <Eigen/Sparse>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>

// ---------------------------------------------------------------------------
// Structure parameters (square core waveguide - unrelated to rib WG)
// ---------------------------------------------------------------------------
static const double core_index = 3.5;
static const double clad_index = 3.1693;   // InP cladding
static const double core_size  = 0.4e-6;   // full core width (x and y)
static const double block_size = 3.4e-6;   // full computational window

// misc. constants
const double n_SiO2 = 1.444;
const double n_air = 1.0;

class wg_simulation {
	
	private:
	
		std::vector<double> profile_y;
		std::vector<double> profile_n;
		
		std::vector<std::vector<double>> spatial_pos;
		std::vector<std::vector<double>> epsilon_r;
		
		double dx = 5e-9;
		double dy = 50e-9;
		double lambda = 1550e-9;
		double k_0    = 2.0 * M_PI / lambda;

		// Misc. solver parameters (ignore)
		const int num_modes = 10;	// number of modes to solve for.
		const int ncv       = 30;   // > num_modes (+1), <= matrix dimension
		int M_x = 0; // initialised to zero
		int M_y = 0; 
		
		bool write_files_out = true;
		bool write_to_console = true;
		
		bool mesh_initilised = false; // false by default.
			// user should define a mesh type.
					
	public:

		double etch_depth; 
		double wg_width; 
		double window_width; 
		double dielectric_thickness;
		
		void set_dx(double dx_in) { this->mesh_initilised = false; this->dx = dx_in; }
		void set_dy(double dy_in) { this->mesh_initilised = false; this->dy = dy_in; }
		void set_propagation_wavelength(double lambda_in) {
			this->lambda = lambda_in;
			this->k_0    = 2.0 * M_PI / lambda_in;
			this->mesh_initilised = false;
		}
		
		int get_M_x() const {
			//~ if (M_x == 0) throw std::logic_error("M_x not initialised");
			return M_x;
		}
		
		int get_M_y() const {
			//~ if (M_y == 0) throw std::logic_error("M_y not initialised");
			return M_y;
		}
		
		// Cell-centred mesh: node p sits at x = -L/2 + (p + 0.5)*dx, dx = L/M.
		// Cell edges are at -L/2 + k*dx, so the core edge (+/- core_size/2) must be
		void initialise_rib_WG(
			double etch_depth_in, 
			double wg_width_in, 
			double window_width_in, 
			double dielectric_thickness_in)
		{
			this->etch_depth = etch_depth_in;
			this->wg_width = wg_width_in;
			this->window_width = window_width_in;
			this->dielectric_thickness = dielectric_thickness_in;

			// ---------------------------------------------------------
			// Read refractive-index profile: z [m], n
			// Here the profile coordinate is interpreted as vertical y.
			// ---------------------------------------------------------
			std::vector<double> profile_y;
			std::vector<double> profile_n;

			std::ifstream profile_file("refractive_index_profile.dat");
			if (!profile_file) {
				throw std::runtime_error(
					"Cannot open refractive_index_profile.dat");
			}

			std::string line;
			int i = 0;
			int idx_air_end = 0;
			while (std::getline(profile_file, line)) {
				if (line.empty() || line[0] == '#')
					continue;

				std::istringstream iss(line);
				double y, n;

				if (iss >> y >> n) {
					profile_y.push_back(y);
					profile_n.push_back(n);
					
					if(n >= 1.01 && idx_air_end == 0) idx_air_end = i;

					//~ std::cout << "y=" << y << " n=" << n << std::endl; 
				}
				i++;
			}
			
			// Read refractive index profile, determine distance from window edge 
			// to actual beggining of epi structure (this is so that when an air 
			// block is included in the input refractive index profile file the 
			// correct etch depth is used).
			
			double air_thickness = std::abs(profile_y[0] - profile_y[idx_air_end]); // 1.5e-6 in example file.
			if(write_to_console == true) std:: cout << "air_thickness=" << air_thickness*1e6 << "um"<< std::endl;

			if (profile_y.size() < 2) {
				throw std::runtime_error(
					"Refractive-index profile must contain at least two points.");
			}

			if (!std::is_sorted(profile_y.begin(), profile_y.end())) {
				throw std::runtime_error(
					"Profile coordinates must be sorted in ascending order.");
			}
			
			const double block_size_x = window_width_in;
			const double block_size_y = std::abs(profile_y[0]-profile_y[profile_y.size()-1]); // Set to your actual vertical domain.
			
			this->M_x = static_cast<int>(std::round(block_size_x / dx));
			this->M_y = static_cast<int>(std::round(block_size_y / dy));
			
			std::vector<std::vector<double>> spatial_pos(2);
			std::vector<std::vector<double>> epsilon_r(this->M_x, std::vector<double>(this->M_y, 0.0));
			
			//~ const double dx = block_size_x / M_x;
			//~ const double dy = block_size_y / M_y;

			// Centre the profile vertically so its midpoint is y = 0.
			const double profile_mid =
				0.5 * (profile_y.front() + profile_y.back());

			for (double& y : profile_y)
				y -= profile_mid;

			// Linear interpolation of n(y).
			auto interpolate_n = [&](double y) -> double {
				if (y <= profile_y.front())
					return profile_n.front();

				if (y >= profile_y.back())
					return profile_n.back();

				auto it = std::upper_bound(
					profile_y.begin(), profile_y.end(), y);

				const std::size_t i =
					static_cast<std::size_t>(it - profile_y.begin() - 1);

				const double t =
					(y - profile_y[i]) /
					(profile_y[i + 1] - profile_y[i]);

				return profile_n[i] +
					t * (profile_n[i + 1] - profile_n[i]);
			};

			// ---------------------------------------------------------
			// Mesh coordinates: cell-centred
			// ---------------------------------------------------------
			std::vector<double> x(M_x), y(M_y);

			for (int p = 0; p < M_x; ++p)
				x[p] = -block_size_x / 2.0 + (p + 0.5) * dx;

			for (int q = 0; q < M_y; ++q)
				y[q] = -block_size_y / 2.0 + (q + 0.5) * dy;

			// Ensure the output arrays have the required dimensions.
			epsilon_r.assign(M_x, std::vector<double>(M_y));

			// ---------------------------------------------------------
			// Build epsilon_r(x,y)
			// ---------------------------------------------------------
			for (int p = 0; p < M_x; ++p) {
				for (int q = 0; q < M_y; ++q) {
					const double n = interpolate_n(y[q]);
					// Rib defintion
					if(std::abs(x[p]) <= wg_width/2.0) epsilon_r[p][q] = n * n;
					else epsilon_r[p][q] = n_air * n_air;
					if(std::abs(x[p]) < wg_width/2.0 + dielectric_thickness && std::abs(x[p]) > wg_width/2.0) epsilon_r[p][q] = n_SiO2 * n_SiO2;
					if(std::abs(x[p]) > wg_width/2.0 && std::abs(y[q]-y[0]) > air_thickness + etch_depth - dielectric_thickness) epsilon_r[p][q] = n_SiO2 * n_SiO2;
					if(std::abs(y[q]-y[0]) > etch_depth+air_thickness) epsilon_r[p][q] = n * n;
					if(std::abs(x[p]) < wg_width/2.0 + dielectric_thickness && std::abs(y[q]-y[0]) > air_thickness-dielectric_thickness && std::abs(y[q]-y[0]) < air_thickness) epsilon_r[p][q] = n_SiO2 * n_SiO2;
					if(std::abs(y[q]-y[0]) < air_thickness-dielectric_thickness) epsilon_r[p][q] = n_air * n_air;
				}
			}

			spatial_pos.resize(2);
			spatial_pos[0] = x;
			spatial_pos[1] = y;
			
			this->epsilon_r = epsilon_r;
			this->spatial_pos = spatial_pos;
			this->profile_n = profile_n;
			this->profile_y = profile_y;

			// ---------------------------------------------------------
			// Write mesh: x [um], y [um], relative permittivity
			// ---------------------------------------------------------
			if(this->write_files_out == true) {
				std::ofstream file("permittivity_mesh.dat");
				if (!file)
					throw std::runtime_error("Cannot write permittivity_mesh.dat");

				file << std::setprecision(10);

				for (int p = 0; p < M_x; ++p) {
					for (int q = 0; q < M_y; ++q) {
						file << x[p] * 1e6 << " "
							 << y[q] * 1e6 << " "
							 << epsilon_r[p][q] << "\n";
					}
					file << "\n";
				}
			}
			
			// allow matrices to be defined..
			this->mesh_initilised = true;
		}


		void E_TE() {
			if(this->mesh_initilised != true){
				throw std::runtime_error("Simulation mesh not initialised.\n");
				return;
			}
			
			const int N_p = M_x - 2;
			const int N_q = M_y - 2;
			const int M   = N_p * N_q;

			auto index_of = [&](int p, int q) { return (p - 1) * N_q + (q - 1); };
			auto is_unknown = [&](int p, int q) {
				return p >= 1 && p <= M_x - 2 && q >= 1 && q <= M_y - 2;
			};

			// ---- assemble A  (A E = (beta/k0)^2 E = n_eff^2 E) --------------------
			// Eq. (4.99) in normalised coordinates x_bar = k0*x, so k0^2 -> 1 and
			// the eigenvalue is n_eff^2.
			std::vector<Eigen::Triplet<double>> triplets;
			triplets.reserve(5 * static_cast<size_t>(M));

			for (int p = 1; p <= M_x - 2; ++p) {
				for (int q = 1; q <= M_y - 2; ++q) {

					// Spacings, Eqs. (4.56)-(4.59): w = x_p - x_{p-1}, e = x_{p+1} - x_p,
					// n = y_q - y_{q-1}, s = y_{q+1} - y_q  (normalised by 1/k0).
					const double w = k_0 * std::abs(spatial_pos[0][p]     - spatial_pos[0][p - 1]);
					const double e = k_0 * std::abs(spatial_pos[0][p + 1] - spatial_pos[0][p]);
					const double n = k_0 * std::abs(spatial_pos[1][q]     - spatial_pos[1][q - 1]);
					const double s = k_0 * std::abs(spatial_pos[1][q + 1] - spatial_pos[1][q]);

					const double eps   = epsilon_r[p][q];
					const double eps_w = epsilon_r[p - 1][q];
					const double eps_e = epsilon_r[p + 1][q];

					// Eqs. (4.100)-(4.105)
					const double alpha_w = 2.0 / (w * (e + w)) * (2.0 * eps_w) / (eps + eps_w);
					const double alpha_e = 2.0 / (e * (e + w)) * (2.0 * eps_e) / (eps + eps_e);
					const double alpha_n = 2.0 / (n * (n + s));
					const double alpha_s = 2.0 / (s * (n + s));

					const double alpha_x = -4.0 / (e * w) + alpha_e + alpha_w;   // (4.104)
					const double alpha_y = -alpha_n - alpha_s;                   // (4.105)

					const int r = index_of(p, q);

					triplets.emplace_back(r, r, alpha_x + alpha_y + eps);   // k0^2*eps -> eps

					// Couplings to neighbours; neighbours on the Dirichlet boundary
					// are zero and are simply dropped.
					if (is_unknown(p - 1, q)) triplets.emplace_back(r, index_of(p - 1, q), alpha_w);
					if (is_unknown(p + 1, q)) triplets.emplace_back(r, index_of(p + 1, q), alpha_e);
					if (is_unknown(p, q - 1)) triplets.emplace_back(r, index_of(p, q - 1), alpha_n);
					if (is_unknown(p, q + 1)) triplets.emplace_back(r, index_of(p, q + 1), alpha_s);
				}
			}

			Eigen::SparseMatrix<double> A(M, M);
			A.setFromTriplets(triplets.begin(), triplets.end());
			A.makeCompressed();

			if (write_to_console == true)
				std::cout << "Matrix size: " << M << " x " << M << "\n(dx=" << dx*1e6 
					<< "um dy=" << dy*1e6 << "um)" << std::endl;

			// ---- shift-invert eigen solve -----------------------------------------
			// Eigenvalues closest to sigma are returned. sigma just above n_core^2
			// targets the highest-index (fundamental) modes.
			const double sigma = core_index * core_index /*+ 0.05*/;

			Spectra::SparseGenRealShiftSolve<double> op(A);
			Spectra::GenEigsRealShiftSolver<Spectra::SparseGenRealShiftSolve<double>>
				eigs(op, num_modes, ncv, sigma);

			eigs.init();
			eigs.compute(Spectra::SortRule::LargestMagn);   // largest 1/(lambda - sigma)
			
			if(write_to_console == true)
				if (eigs.info() != Spectra::CompInfo::Successful) {
					std::cerr << "Eigen solve did not converge ("
							   << " of " << num_modes
							  << " converged). Try increasing ncv.\n";
					return;
				}

			Eigen::VectorXcd eigenvalues = eigs.eigenvalues();
			Eigen::MatrixXcd V           = eigs.eigenvectors();

			// Sort by n_eff^2 descending
			std::vector<std::pair<double, int>> order;
			for (int i = 0; i < eigenvalues.size(); ++i)
				order.emplace_back(eigenvalues[i].real(), i);
			std::sort(order.begin(), order.end(),
					  [](const auto& a, const auto& b) { return a.first > b.first; });

			std::cout << std::fixed << std::setprecision(6);
			for (size_t m = 0; m < order.size(); ++m) {
				const double neff2 = order[m].first;
				const double neff  = (neff2 > 0.0) ? std::sqrt(neff2) : 0.0;
				const bool guided  = (neff > clad_index && neff < core_index);
				if(write_to_console == true) 
					std::cout << "Mode " << m + 1 << ": n_eff^2 = " << neff2
						  << "  n_eff = " << neff
						  << (guided ? "  (guided)" : "  (not guided / box mode)") << '\n';
			}

			// ---- export -----------------------------------------------------------
			// Files: x[um] y[um] Re(E) Im(E) |E|^2, normalised to max |E| = 1.
			// Eigenvectors are real up to a global phase; rotate so the peak is real.
			
			if(write_files_out == true) {
				for (size_t m = 0; m < order.size(); ++m) {
					const int col = order[m].second;

					int peak = 0;
					V.col(col).cwiseAbs().maxCoeff(&peak);
					const std::complex<double> phase = V(peak, col);
					const double max_E = std::abs(phase);

					std::ofstream file("E_TE_mode_" + std::to_string(m + 1) + ".dat");
					file << std::setprecision(10);

					for (int p = 0; p < M_x; ++p) {
						for (int q = 0; q < M_y; ++q) {
							std::complex<double> E(0.0, 0.0);   // zero on Dirichlet boundary
							if (is_unknown(p, q) && max_E > 0.0)
								E = V(index_of(p, q), col) / phase;   // peak -> +1

							file << spatial_pos[0][p] * 1e6 << " "
								 << spatial_pos[1][q] * 1e6 << " "
								 << E.real() << " " << E.imag() << " "
								 << std::norm(E) << "\n";
						}
						file << "\n";
					}
				}
			}

		}
		
		void H_TE() {
			if(this->mesh_initilised != true){
				throw std::runtime_error("Simulation mesh not initialised.\n");
				return;
			}
			
			const int N_p = M_x - 2;
			const int N_q = M_y - 2;
			const int M   = N_p * N_q;

			auto index_of = [&](int p, int q) { return (p - 1) * N_q + (q - 1); };
			auto is_unknown = [&](int p, int q) {
				return p >= 1 && p <= M_x - 2 && q >= 1 && q <= M_y - 2;
			};

			// ---- assemble A  (A E = (beta/k0)^2 E = n_eff^2 E) --------------------
			// Eq. (4.123) in normalised coordinates x_bar = k0*x, so k0^2 -> 1 and
			// the eigenvalue is n_eff^2.
			std::vector<Eigen::Triplet<double>> triplets;
			triplets.reserve(5 * static_cast<size_t>(M));

			for (int p = 1; p <= M_x - 2; ++p) {
				for (int q = 1; q <= M_y - 2; ++q) {

					// Spacings, Eqs. (4.56)-(4.59): w = x_p - x_{p-1}, e = x_{p+1} - x_p,
					// n = y_q - y_{q-1}, s = y_{q+1} - y_q  (normalised by 1/k0).
					const double w = k_0 * std::abs(spatial_pos[0][p]     - spatial_pos[0][p - 1]);
					const double e = k_0 * std::abs(spatial_pos[0][p + 1] - spatial_pos[0][p]);
					const double n = k_0 * std::abs(spatial_pos[1][q]     - spatial_pos[1][q - 1]);
					const double s = k_0 * std::abs(spatial_pos[1][q + 1] - spatial_pos[1][q]);

					const double eps   = epsilon_r[p][q];
					const double eps_w = epsilon_r[p - 1][q];
					const double eps_e = epsilon_r[p + 1][q];

					// Eqs. (4.124)-(4.129): weighting on the x-neighbours (w, e),
					// plain Laplacian in y.
					const double alpha_w = 2.0 / (w * (e + w)) * (2.0 * eps) / (eps + eps_w);
					const double alpha_e = 2.0 / (e * (e + w)) * (2.0 * eps) / (eps + eps_e);
					const double alpha_n = 2.0 / (n * (n + s));
					const double alpha_s = 2.0 / (s * (n + s));

					const double alpha_x = -alpha_e - alpha_w;   // (4.128)
					const double alpha_y = -alpha_n - alpha_s;   // (4.129)

					const int r = index_of(p, q);

					triplets.emplace_back(r, r, alpha_x + alpha_y + eps);   // k0^2*eps -> eps

					// Couplings to neighbours; neighbours on the Dirichlet boundary
					// are zero and are simply dropped.
					if (is_unknown(p - 1, q)) triplets.emplace_back(r, index_of(p - 1, q), alpha_w);
					if (is_unknown(p + 1, q)) triplets.emplace_back(r, index_of(p + 1, q), alpha_e);
					if (is_unknown(p, q - 1)) triplets.emplace_back(r, index_of(p, q - 1), alpha_n);
					if (is_unknown(p, q + 1)) triplets.emplace_back(r, index_of(p, q + 1), alpha_s);
				}
			}

			Eigen::SparseMatrix<double> A(M, M);
			A.setFromTriplets(triplets.begin(), triplets.end());
			A.makeCompressed();

			if (write_to_console == true)
				std::cout << "Matrix size: " << M << " x " << M << "\n(dx=" << dx*1e6 
					<< "um dy=" << dy*1e6 << "um)" << std::endl;

			// ---- shift-invert eigen solve -----------------------------------------
			// Eigenvalues closest to sigma are returned. sigma just above n_core^2
			// targets the highest-index (fundamental) modes.
			const double sigma = core_index * core_index /*+ 0.05*/;

			Spectra::SparseGenRealShiftSolve<double> op(A);
			Spectra::GenEigsRealShiftSolver<Spectra::SparseGenRealShiftSolve<double>>
				eigs(op, num_modes, ncv, sigma);

			eigs.init();
			eigs.compute(Spectra::SortRule::LargestMagn);   // largest 1/(lambda - sigma)
			
			if(write_to_console == true)
				if (eigs.info() != Spectra::CompInfo::Successful) {
					std::cerr << "Eigen solve did not converge ("
							   << " of " << num_modes
							  << " converged). Try increasing ncv.\n";
					return;
				}

			Eigen::VectorXcd eigenvalues = eigs.eigenvalues();
			Eigen::MatrixXcd V           = eigs.eigenvectors();

			// Sort by n_eff^2 descending
			std::vector<std::pair<double, int>> order;
			for (int i = 0; i < eigenvalues.size(); ++i)
				order.emplace_back(eigenvalues[i].real(), i);
			std::sort(order.begin(), order.end(),
					  [](const auto& a, const auto& b) { return a.first > b.first; });

			std::cout << std::fixed << std::setprecision(6);
			for (size_t m = 0; m < order.size(); ++m) {
				const double neff2 = order[m].first;
				const double neff  = (neff2 > 0.0) ? std::sqrt(neff2) : 0.0;
				const bool guided  = (neff > clad_index && neff < core_index);
				if(write_to_console == true) 
					std::cout << "Mode " << m + 1 << ": n_eff^2 = " << neff2
						  << "  n_eff = " << neff
						  << (guided ? "  (guided)" : "  (not guided / box mode)") << '\n';
			}

			// ---- export -----------------------------------------------------------
			// Files: x[um] y[um] Re(E) Im(E) |E|^2, normalised to max |E| = 1.
			// Eigenvectors are real up to a global phase; rotate so the peak is real.
			
			if(write_files_out == true) {
				for (size_t m = 0; m < order.size(); ++m) {
					const int col = order[m].second;

					int peak = 0;
					V.col(col).cwiseAbs().maxCoeff(&peak);
					const std::complex<double> phase = V(peak, col);
					const double max_E = std::abs(phase);

					std::ofstream file("H_TE_mode_" + std::to_string(m + 1) + ".dat");
					file << std::setprecision(10);

					for (int p = 0; p < M_x; ++p) {
						for (int q = 0; q < M_y; ++q) {
							std::complex<double> E(0.0, 0.0);   // zero on Dirichlet boundary
							if (is_unknown(p, q) && max_E > 0.0)
								E = V(index_of(p, q), col) / phase;   // peak -> +1

							file << spatial_pos[0][p] * 1e6 << " "
								 << spatial_pos[1][q] * 1e6 << " "
								 << E.real() << " " << E.imag() << " "
								 << std::norm(E) << "\n";
						}
						file << "\n";
					}
				}
			}
		}
		
		void E_TM() {
			if(this->mesh_initilised != true){
				throw std::runtime_error("Simulation mesh not initialised.\n");
				return;
			}

			const int N_p = M_x - 2;
			const int N_q = M_y - 2;
			const int M   = N_p * N_q;

			auto index_of = [&](int p, int q) { return (p - 1) * N_q + (q - 1); };
			auto is_unknown = [&](int p, int q) {
				return p >= 1 && p <= M_x - 2 && q >= 1 && q <= M_y - 2;
			};

			// ---- assemble A  (A E = (beta/k0)^2 E = n_eff^2 E) --------------------
			// Eq. (4.133) in normalised coordinates x_bar = k0*x, so k0^2 -> 1 and
			// the eigenvalue is n_eff^2.
			std::vector<Eigen::Triplet<double>> triplets;
			triplets.reserve(5 * static_cast<size_t>(M));

			for (int p = 1; p <= M_x - 2; ++p) {
				for (int q = 1; q <= M_y - 2; ++q) {

					// Spacings, Eqs. (4.56)-(4.59): w = x_p - x_{p-1}, e = x_{p+1} - x_p,
					// n = y_q - y_{q-1}, s = y_{q+1} - y_q  (normalised by 1/k0).
					const double w = k_0 * std::abs(spatial_pos[0][p]     - spatial_pos[0][p - 1]);
					const double e = k_0 * std::abs(spatial_pos[0][p + 1] - spatial_pos[0][p]);
					const double n = k_0 * std::abs(spatial_pos[1][q]     - spatial_pos[1][q - 1]);
					const double s = k_0 * std::abs(spatial_pos[1][q + 1] - spatial_pos[1][q]);

					const double eps   = epsilon_r[p][q];
					const double eps_n = epsilon_r[p][q - 1];
					const double eps_s = epsilon_r[p][q + 1];

					// Eqs. (4.134)-(4.139). Note the permittivity weighting sits on the
					// y-neighbours (n, s), the opposite of the quasi-TE case.
					const double alpha_w = 2.0 / (w * (e + w));
					const double alpha_e = 2.0 / (e * (e + w));
					const double alpha_n = 2.0 / (n * (n + s)) * (2.0 * eps_n) / (eps + eps_n);
					const double alpha_s = 2.0 / (s * (n + s)) * (2.0 * eps_s) / (eps + eps_s);

					const double alpha_x = -alpha_e - alpha_w;                   // (4.138)
					const double alpha_y = -4.0 / (n * s) + alpha_n + alpha_s;   // (4.139)

					const int r = index_of(p, q);

					triplets.emplace_back(r, r, alpha_x + alpha_y + eps);   // k0^2*eps -> eps

					// Couplings to neighbours; neighbours on the Dirichlet boundary
					// are zero and are simply dropped.
					if (is_unknown(p - 1, q)) triplets.emplace_back(r, index_of(p - 1, q), alpha_w);
					if (is_unknown(p + 1, q)) triplets.emplace_back(r, index_of(p + 1, q), alpha_e);
					if (is_unknown(p, q - 1)) triplets.emplace_back(r, index_of(p, q - 1), alpha_n);
					if (is_unknown(p, q + 1)) triplets.emplace_back(r, index_of(p, q + 1), alpha_s);
				}
			}


			Eigen::SparseMatrix<double> A(M, M);
			A.setFromTriplets(triplets.begin(), triplets.end());
			A.makeCompressed();

			if (write_to_console == true)
				std::cout << "Matrix size: " << M << " x " << M << "\n(dx=" << dx*1e6 
					<< "um dy=" << dy*1e6 << "um)" << std::endl;

			// ---- shift-invert eigen solve -----------------------------------------
			// Eigenvalues closest to sigma are returned. sigma just above n_core^2
			// targets the highest-index (fundamental) modes.
			const double sigma = core_index * core_index /*+ 0.05*/;

			Spectra::SparseGenRealShiftSolve<double> op(A);
			Spectra::GenEigsRealShiftSolver<Spectra::SparseGenRealShiftSolve<double>>
				eigs(op, num_modes, ncv, sigma);

			eigs.init();
			eigs.compute(Spectra::SortRule::LargestMagn);   // largest 1/(lambda - sigma)
			
			if(write_to_console == true)
				if (eigs.info() != Spectra::CompInfo::Successful) {
					std::cerr << "Eigen solve did not converge ("
							   << " of " << num_modes
							  << " converged). Try increasing ncv.\n";
					return;
				}

			Eigen::VectorXcd eigenvalues = eigs.eigenvalues();
			Eigen::MatrixXcd V           = eigs.eigenvectors();

			// Sort by n_eff^2 descending
			std::vector<std::pair<double, int>> order;
			for (int i = 0; i < eigenvalues.size(); ++i)
				order.emplace_back(eigenvalues[i].real(), i);
			std::sort(order.begin(), order.end(),
					  [](const auto& a, const auto& b) { return a.first > b.first; });

			std::cout << std::fixed << std::setprecision(6);
			for (size_t m = 0; m < order.size(); ++m) {
				const double neff2 = order[m].first;
				const double neff  = (neff2 > 0.0) ? std::sqrt(neff2) : 0.0;
				const bool guided  = (neff > clad_index && neff < core_index);
				if(write_to_console == true) 
					std::cout << "Mode " << m + 1 << ": n_eff^2 = " << neff2
						  << "  n_eff = " << neff
						  << (guided ? "  (guided)" : "  (not guided / box mode)") << '\n';
			}

			// ---- export -----------------------------------------------------------
			// Files: x[um] y[um] Re(E) Im(E) |E|^2, normalised to max |E| = 1.
			// Eigenvectors are real up to a global phase; rotate so the peak is real.
			
			if(write_files_out == true) {
				for (size_t m = 0; m < order.size(); ++m) {
					const int col = order[m].second;

					int peak = 0;
					V.col(col).cwiseAbs().maxCoeff(&peak);
					const std::complex<double> phase = V(peak, col);
					const double max_E = std::abs(phase);

					std::ofstream file("E_TM_mode_" + std::to_string(m + 1) + ".dat");
					file << std::setprecision(10);

					for (int p = 0; p < M_x; ++p) {
						for (int q = 0; q < M_y; ++q) {
							std::complex<double> E(0.0, 0.0);   // zero on Dirichlet boundary
							if (is_unknown(p, q) && max_E > 0.0)
								E = V(index_of(p, q), col) / phase;   // peak -> +1

							file << spatial_pos[0][p] * 1e6 << " "
								 << spatial_pos[1][q] * 1e6 << " "
								 << E.real() << " " << E.imag() << " "
								 << std::norm(E) << "\n";
						}
						file << "\n";
					}
				}
			}

		}
		
		void H_TM() {
			if(this->mesh_initilised != true){
				throw std::runtime_error("Simulation mesh not initialised.\n");
				return;
			}

			const int N_p = M_x - 2;
			const int N_q = M_y - 2;
			const int M   = N_p * N_q;

			auto index_of = [&](int p, int q) { return (p - 1) * N_q + (q - 1); };
			auto is_unknown = [&](int p, int q) {
				return p >= 1 && p <= M_x - 2 && q >= 1 && q <= M_y - 2;
			};

			// ---- assemble A  (A E = (beta/k0)^2 E = n_eff^2 E) --------------------
			// Eq. (4.123) with x <-> y, in normalised coordinates x_bar = k0*x, so k0^2 -> 1 and
			// the eigenvalue is n_eff^2.
			std::vector<Eigen::Triplet<double>> triplets;
			triplets.reserve(5 * static_cast<size_t>(M));

			for (int p = 1; p <= M_x - 2; ++p) {
				for (int q = 1; q <= M_y - 2; ++q) {

					// Spacings, Eqs. (4.56)-(4.59): w = x_p - x_{p-1}, e = x_{p+1} - x_p,
					// n = y_q - y_{q-1}, s = y_{q+1} - y_q  (normalised by 1/k0).
					const double w = k_0 * std::abs(spatial_pos[0][p]     - spatial_pos[0][p - 1]);
					const double e = k_0 * std::abs(spatial_pos[0][p + 1] - spatial_pos[0][p]);
					const double n = k_0 * std::abs(spatial_pos[1][q]     - spatial_pos[1][q - 1]);
					const double s = k_0 * std::abs(spatial_pos[1][q + 1] - spatial_pos[1][q]);

					const double eps   = epsilon_r[p][q];
					const double eps_n = epsilon_r[p][q - 1];
					const double eps_s = epsilon_r[p][q + 1];

					// (4.124)-(4.129) with x <-> y, Eq. (4.141): weighting on the
					// y-neighbours (n, s), plain Laplacian in x.
					const double alpha_w = 2.0 / (w * (e + w));
					const double alpha_e = 2.0 / (e * (e + w));
					const double alpha_n = 2.0 / (n * (n + s)) * (2.0 * eps) / (eps + eps_n);
					const double alpha_s = 2.0 / (s * (n + s)) * (2.0 * eps) / (eps + eps_s);

					const double alpha_x = -alpha_e - alpha_w;
					const double alpha_y = -alpha_n - alpha_s;

					const int r = index_of(p, q);

					triplets.emplace_back(r, r, alpha_x + alpha_y + eps);   // k0^2*eps -> eps

					// Couplings to neighbours; neighbours on the Dirichlet boundary
					// are zero and are simply dropped.
					if (is_unknown(p - 1, q)) triplets.emplace_back(r, index_of(p - 1, q), alpha_w);
					if (is_unknown(p + 1, q)) triplets.emplace_back(r, index_of(p + 1, q), alpha_e);
					if (is_unknown(p, q - 1)) triplets.emplace_back(r, index_of(p, q - 1), alpha_n);
					if (is_unknown(p, q + 1)) triplets.emplace_back(r, index_of(p, q + 1), alpha_s);
				}
			}

			Eigen::SparseMatrix<double> A(M, M);
			A.setFromTriplets(triplets.begin(), triplets.end());
			A.makeCompressed();

			if (write_to_console == true)
				std::cout << "Matrix size: " << M << " x " << M << "\n(dx=" << dx*1e6 
					<< "um dy=" << dy*1e6 << "um)" << std::endl;

			// ---- shift-invert eigen solve -----------------------------------------
			// Eigenvalues closest to sigma are returned. sigma just above n_core^2
			// targets the highest-index (fundamental) modes.
			const double sigma = core_index * core_index /*+ 0.05*/;

			Spectra::SparseGenRealShiftSolve<double> op(A);
			Spectra::GenEigsRealShiftSolver<Spectra::SparseGenRealShiftSolve<double>>
				eigs(op, num_modes, ncv, sigma);

			eigs.init();
			eigs.compute(Spectra::SortRule::LargestMagn);   // largest 1/(lambda - sigma)
			
			if(write_to_console == true)
				if (eigs.info() != Spectra::CompInfo::Successful) {
					std::cerr << "Eigen solve did not converge ("
							   << " of " << num_modes
							  << " converged). Try increasing ncv.\n";
					return;
				}

			Eigen::VectorXcd eigenvalues = eigs.eigenvalues();
			Eigen::MatrixXcd V           = eigs.eigenvectors();

			// Sort by n_eff^2 descending
			std::vector<std::pair<double, int>> order;
			for (int i = 0; i < eigenvalues.size(); ++i)
				order.emplace_back(eigenvalues[i].real(), i);
			std::sort(order.begin(), order.end(),
					  [](const auto& a, const auto& b) { return a.first > b.first; });

			std::cout << std::fixed << std::setprecision(6);
			for (size_t m = 0; m < order.size(); ++m) {
				const double neff2 = order[m].first;
				const double neff  = (neff2 > 0.0) ? std::sqrt(neff2) : 0.0;
				const bool guided  = (neff > clad_index && neff < core_index);
				if(write_to_console == true) 
					std::cout << "Mode " << m + 1 << ": n_eff^2 = " << neff2
						  << "  n_eff = " << neff
						  << (guided ? "  (guided)" : "  (not guided / box mode)") << '\n';
			}

			// ---- export -----------------------------------------------------------
			// Files: x[um] y[um] Re(E) Im(E) |E|^2, normalised to max |E| = 1.
			// Eigenvectors are real up to a global phase; rotate so the peak is real.
			
			if(write_files_out == true) {
				for (size_t m = 0; m < order.size(); ++m) {
					const int col = order[m].second;

					int peak = 0;
					V.col(col).cwiseAbs().maxCoeff(&peak);
					const std::complex<double> phase = V(peak, col);
					const double max_E = std::abs(phase);

					std::ofstream file("H_TM_mode_" + std::to_string(m + 1) + ".dat");
					file << std::setprecision(10);

					for (int p = 0; p < M_x; ++p) {
						for (int q = 0; q < M_y; ++q) {
							std::complex<double> E(0.0, 0.0);   // zero on Dirichlet boundary
							if (is_unknown(p, q) && max_E > 0.0)
								E = V(index_of(p, q), col) / phase;   // peak -> +1

							file << spatial_pos[0][p] * 1e6 << " "
								 << spatial_pos[1][q] * 1e6 << " "
								 << E.real() << " " << E.imag() << " "
								 << std::norm(E) << "\n";
						}
						file << "\n";
					}
				}
			}

		}
		
		void solve_modes(){
			#pragma omp parallel num_threads(4)
			{
				#pragma omp single
				{
					#pragma omp task
					{ try { this->E_TE(); } catch (const std::exception& e) { std::cerr << "E_TE: " << e.what() << "\n"; } }

					#pragma omp task
					{ try { this->H_TE(); } catch (const std::exception& e) { std::cerr << "H_TE: " << e.what() << "\n"; } }

					#pragma omp task
					{ try { this->E_TM(); } catch (const std::exception& e) { std::cerr << "E_TM: " << e.what() << "\n"; } }

					#pragma omp task
					{ try { this->H_TM(); } catch (const std::exception& e) { std::cerr << "H_TM: " << e.what() << "\n"; } }
				}   // the implicit barrier at the end of the parallel region waits for all four tasks
			}
		}
};
	
	int main(int argc, char* argv[])
	{
		setvbuf(stdout, NULL, _IONBF, 0);
		
		// Create simulation object.
		wg_simulation sim1;
		
		// Arbitrary mesh initialiser (user can create their own)
		sim1.initialise_rib_WG(1.85e-6, 2e-6, 10e-6, 300e-9);
		
		// Threaded function sets up matrices in parallel and solves
		sim1.solve_modes();
		
		// ..or to compute one type individually
		//~ sim1.E_TE();

		// This version supports constant uniform grid spacing, dx and 
		//   dy can be set independantly using:
		//~ sim1.set_dx();
		//~ sim1.set_dy();

		return 0;
	}
