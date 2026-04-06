#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <chrono>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

// Stable matrix exponential implementation (Eigen scaling-squaring + Padé)
Eigen::MatrixXd matrix_exp_stable(const Eigen::MatrixXd& A) {
    return A.exp();
}

// Global constants
const double ARNOLDI_TOL = 1e-8;
const int MAX_RESTARTS = 10;

// CSRHost class for host-side matrix handling
struct CSRHost {
    int rows;
    int cols;
    long long nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;

    CSRHost() : rows(0), cols(0), nnz(0) {}

    static uint32_t load_le_u32(const unsigned char* p) {
        return static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) |
               (static_cast<uint32_t>(p[3]) << 24);
    }

    static uint64_t load_le_u64(const unsigned char* p) {
        return static_cast<uint64_t>(p[0]) |
               (static_cast<uint64_t>(p[1]) << 8) |
               (static_cast<uint64_t>(p[2]) << 16) |
               (static_cast<uint64_t>(p[3]) << 24) |
               (static_cast<uint64_t>(p[4]) << 32) |
               (static_cast<uint64_t>(p[5]) << 40) |
               (static_cast<uint64_t>(p[6]) << 48) |
               (static_cast<uint64_t>(p[7]) << 56);
    }

    void from_matrix_market(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        bool header_parsed = false;
        std::string field, symmetry, format, object;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            ss >> object;
            std::transform(object.begin(), object.end(), object.begin(), ::tolower);
            if (object != "%%matrixmarket") continue;
            ss >> object >> format >> field >> symmetry;
            std::transform(object.begin(), object.end(), object.begin(), ::tolower);
            std::transform(format.begin(), format.end(), format.begin(), ::tolower);
            std::transform(field.begin(), field.end(), field.begin(), ::tolower);
            std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::tolower);
            if (object != "matrix") {
                std::cerr << "Error: MatrixMarket object must be 'matrix'" << std::endl;
                exit(EXIT_FAILURE);
            }
            if (format != "coordinate") {
                std::cerr << "Error: Only 'coordinate' format is supported" << std::endl;
                exit(EXIT_FAILURE);
            }
            if (field != "real" && field != "pattern") {
                std::cerr << "Error: Only 'real' or 'pattern' fields are supported" << std::endl;
                exit(EXIT_FAILURE);
            }
            if (symmetry != "general" && symmetry != "symmetric") {
                std::cerr << "Error: Only 'general' or 'symmetric' symmetry types are supported" << std::endl;
                exit(EXIT_FAILURE);
            }
            header_parsed = true;
            break;
        }

        if (!header_parsed) {
            std::cerr << "Error: Failed to parse MatrixMarket header" << std::endl;
            exit(EXIT_FAILURE);
        }

        const bool is_pattern = (field == "pattern");
        const bool is_symmetric = (symmetry == "symmetric");

        bool dims_parsed = false;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '%') continue;
            std::stringstream ss(line);
            if (!(ss >> rows >> cols >> nnz)) {
                std::cerr << "Error: Failed to read matrix dimensions" << std::endl;
                exit(EXIT_FAILURE);
            }
            dims_parsed = true;
            break;
        }
        if (!dims_parsed) {
            std::cerr << "Error: Matrix dimensions line not found in file " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        row_ptr.assign(rows + 1, 0);
        std::vector<std::tuple<int, int, double>> entries;
        entries.reserve(static_cast<size_t>(nnz) * (is_symmetric ? 2 : 1));

        long long read_entries = 0;
        while (read_entries < nnz && std::getline(file, line)) {
            if (line.empty() || line[0] == '%') continue;
            std::stringstream ss(line);
            int r, c;
            double val = 1.0;
            if (!(ss >> r >> c)) {
                std::cerr << "Error: Failed to read row/col for entry " << read_entries << std::endl;
                exit(EXIT_FAILURE);
            }
            if (!is_pattern) {
                if (!(ss >> val)) {
                    std::cerr << "Error: Failed to read value for entry " << read_entries << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            r -= 1;
            c -= 1;
            if (r < 0 || r >= rows || c < 0 || c >= cols) {
                std::cerr << "Error: MatrixMarket index out of bounds r=" << (r + 1)
                          << " c=" << (c + 1) << std::endl;
                exit(EXIT_FAILURE);
            }
            entries.emplace_back(r, c, val);
            if (is_symmetric && r != c) {
                entries.emplace_back(c, r, val);
            }
            ++read_entries;
        }
        file.close();

        if (read_entries != nnz) {
            std::cerr << "Error: Expected " << nnz << " entries but read " << read_entries << std::endl;
            exit(EXIT_FAILURE);
        }

        std::sort(entries.begin(), entries.end());

        std::vector<std::tuple<int, int, double>> merged;
        merged.reserve(entries.size());
        for (const auto& e : entries) {
            if (merged.empty() || std::get<0>(merged.back()) != std::get<0>(e) || std::get<1>(merged.back()) != std::get<1>(e)) {
                merged.push_back(e);
            } else {
                std::get<2>(merged.back()) += std::get<2>(e);
            }
        }

        nnz = static_cast<long long>(merged.size());
        col_idx.clear();
        values.clear();
        col_idx.reserve(nnz);
        values.reserve(nnz);

        int current_row = 0;
        for (const auto& entry : merged) {
            int r, c;
            double val;
            std::tie(r, c, val) = entry;

            while (current_row < r) {
                row_ptr[current_row + 1] = static_cast<int>(col_idx.size());
                current_row++;
            }
            col_idx.push_back(c);
            values.push_back(val);
        }
        while (current_row < rows) {
            row_ptr[current_row + 1] = static_cast<int>(col_idx.size());
            current_row++;
        }
    }

    void from_binary_csr(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open binary CSR file " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        file.seekg(0, std::ios::end);
        const std::streamoff file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (file_size < 64) {
            std::cerr << "Error: Binary CSR file too small: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        unsigned char header[64];
        file.read(reinterpret_cast<char*>(header), 64);
        if (!file) {
            std::cerr << "Error: Failed to read 64-byte header from " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        const unsigned char expected_magic[8] = {'C', 'S', 'R', 0, 0, 0, 0, 1};
        if (std::memcmp(header + 0, expected_magic, 8) != 0) {
            std::cerr << "Error: Invalid Binary CSR magic in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        const uint32_t version = load_le_u32(header + 8);
        const uint32_t flags = load_le_u32(header + 12);
        const uint64_t nrows_u64 = load_le_u64(header + 16);
        const uint64_t ncols_u64 = load_le_u64(header + 24);
        const uint64_t nnz_u64 = load_le_u64(header + 32);
        const uint32_t index_dtype = load_le_u32(header + 40);
        const uint32_t value_dtype = load_le_u32(header + 44);

        if (version != 1) {
            std::cerr << "Error: Unsupported Binary CSR version " << version << " in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!(index_dtype == 1 || index_dtype == 2)) {
            std::cerr << "Error: Unsupported index_dtype=" << index_dtype << " in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        if (!(value_dtype == 1 || value_dtype == 2)) {
            std::cerr << "Error: Unsupported value_dtype=" << value_dtype << " in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        const size_t index_bytes = (index_dtype == 1) ? sizeof(uint32_t) : sizeof(uint64_t);
        const size_t value_bytes = (value_dtype == 1) ? sizeof(float) : sizeof(double);
        const uint64_t indptr_len = nrows_u64 + 1;

        uint64_t expected_payload = 0;
        expected_payload += indptr_len * static_cast<uint64_t>(index_bytes);
        expected_payload += nnz_u64 * static_cast<uint64_t>(index_bytes);
        expected_payload += nnz_u64 * static_cast<uint64_t>(value_bytes);
        const uint64_t expected_total = 64 + expected_payload;
        if (expected_total != static_cast<uint64_t>(file_size)) {
            std::cerr << "Error: Binary CSR size mismatch in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<uint64_t> indptr64(indptr_len, 0);
        std::vector<uint64_t> indices64(nnz_u64, 0);
        std::vector<double> data64(nnz_u64, 0.0);

        if (index_dtype == 1) {
            std::vector<uint32_t> tmp(indptr_len);
            file.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(uint32_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indptr from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < tmp.size(); ++i) indptr64[i] = static_cast<uint64_t>(tmp[i]);

            std::vector<uint32_t> tmp_idx(nnz_u64);
            file.read(reinterpret_cast<char*>(tmp_idx.data()), static_cast<std::streamsize>(tmp_idx.size() * sizeof(uint32_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indices from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < tmp_idx.size(); ++i) indices64[i] = static_cast<uint64_t>(tmp_idx[i]);
        } else {
            file.read(reinterpret_cast<char*>(indptr64.data()), static_cast<std::streamsize>(indptr64.size() * sizeof(uint64_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indptr from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            file.read(reinterpret_cast<char*>(indices64.data()), static_cast<std::streamsize>(indices64.size() * sizeof(uint64_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indices from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        if (value_dtype == 1) {
            std::vector<float> tmp_val(nnz_u64);
            file.read(reinterpret_cast<char*>(tmp_val.data()), static_cast<std::streamsize>(tmp_val.size() * sizeof(float)));
            if (!file) {
                std::cerr << "Error: Failed to read data from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < tmp_val.size(); ++i) data64[i] = static_cast<double>(tmp_val[i]);
        } else {
            file.read(reinterpret_cast<char*>(data64.data()), static_cast<std::streamsize>(data64.size() * sizeof(double)));
            if (!file) {
                std::cerr << "Error: Failed to read data from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        const bool flag_symmetric_upper = (flags & (1u << 2)) != 0;

        rows = static_cast<int>(nrows_u64);
        cols = static_cast<int>(ncols_u64);
        nnz = static_cast<long long>(nnz_u64);

        if (flag_symmetric_upper) {
            if (rows != cols) {
                std::cerr << "Error: symmetric_upper requires square matrix in " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            std::vector<std::tuple<int, int, double>> entries;
            entries.reserve(static_cast<size_t>(nnz_u64) * 2);
            for (int r = 0; r < rows; ++r) {
                uint64_t start = indptr64[r];
                uint64_t end = indptr64[r + 1];
                for (uint64_t p = start; p < end; ++p) {
                    int c = static_cast<int>(indices64[p]);
                    double v = data64[p];
                    entries.emplace_back(r, c, v);
                    if (c != r) {
                        entries.emplace_back(c, r, v);
                    }
                }
            }

            std::sort(entries.begin(), entries.end());

            std::vector<std::tuple<int, int, double>> merged;
            merged.reserve(entries.size());
            for (const auto& e : entries) {
                if (merged.empty() || std::get<0>(merged.back()) != std::get<0>(e) || std::get<1>(merged.back()) != std::get<1>(e)) {
                    merged.push_back(e);
                } else {
                    std::get<2>(merged.back()) += std::get<2>(e);
                }
            }

            row_ptr.assign(rows + 1, 0);
            col_idx.clear();
            values.clear();
            col_idx.reserve(merged.size());
            values.reserve(merged.size());

            int current_row = 0;
            for (const auto& entry : merged) {
                int r, c;
                double v;
                std::tie(r, c, v) = entry;
                while (current_row < r) {
                    row_ptr[current_row + 1] = static_cast<int>(col_idx.size());
                    current_row++;
                }
                col_idx.push_back(c);
                values.push_back(v);
            }
            while (current_row < rows) {
                row_ptr[current_row + 1] = static_cast<int>(col_idx.size());
                current_row++;
            }
            nnz = static_cast<long long>(values.size());
        } else {
            row_ptr.resize(rows + 1);
            col_idx.resize(nnz_u64);
            values.resize(nnz_u64);
            for (int r = 0; r <= rows; ++r) {
                row_ptr[r] = static_cast<int>(indptr64[r]);
            }
            for (uint64_t k = 0; k < nnz_u64; ++k) {
                col_idx[k] = static_cast<int>(indices64[k]);
                values[k] = data64[k];
            }
        }
    }
};

// CPU CSR SpMV
static void spmv_csr(const int* row_ptr, const int* col_idx, const double* vals,
                     const double* x, double* y, int n) {
    for (int r = 0; r < n; ++r) {
        double acc = 0.0;
        for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k) {
            acc += vals[k] * x[col_idx[k]];
        }
        y[r] = acc;
    }
}

// ArnoldiParams
struct ArnoldiParams {
    int m;
    double t;
    double tol;
    int max_restarts;
    bool reorthogonalize;

    ArnoldiParams(int m_val = 100, double t_val = 1.0, double tol_val = ARNOLDI_TOL,
                  int max_restarts_val = MAX_RESTARTS, bool reortho = true)
        : m(m_val), t(t_val), tol(tol_val), max_restarts(max_restarts_val), reorthogonalize(reortho) {}
};

enum class ExpmvMode {
    AdaptiveTimeStepping,
    RestartedKrylov
};

struct TimeSteppingParams {
    double dt_init{0.0};
    double dt_min{0.0};
    double dt_max{0.0};
    double tol_step{0.0};
    int max_steps{0};
    int max_rejects{20};
    double safety{0.8};
    double grow{1.5};
    double shrink{0.5};
    bool adapt_m{false};
    int m_max{0};
    int m_step{0};
};

// ArnoldiRunnerCPU: CPU-only Arnoldi Krylov for exp(A)*v
struct ArnoldiRunnerCPU {
    const CSRHost& A;
    ArnoldiParams params;
    TimeSteppingParams ts_params;
    bool converged = false;
    int restarts_done = 0;
    double last_residual = 0.0;

    Eigen::MatrixXd H_m;
    Eigen::VectorXd e1;
    Eigen::VectorXd wH;
    double v_norm;

    std::vector<double> V_m;  // column-major: V_m[col * n + row], size n * (m+1)
    std::vector<double> w;
    std::vector<double> y;
    int n;
    int allocated_m_max;

    ArnoldiRunnerCPU(const CSRHost& matrix, const ArnoldiParams& arnoldi_params)
        : A(matrix), params(arnoldi_params), n(matrix.rows), allocated_m_max(arnoldi_params.m) {
        ts_params.dt_max = params.t;
        ts_params.dt_min = std::max(1e-12, params.t * 1e-6);
        double denom = (params.max_restarts > 0) ? params.max_restarts : 1;
        ts_params.dt_init = params.t / static_cast<double>(denom);
        ts_params.tol_step = params.tol;
        ts_params.max_steps = (params.max_restarts > 0) ? params.max_restarts : 1000;
        ts_params.max_rejects = 20;
        ts_params.safety = 0.8;
        ts_params.grow = 1.5;
        ts_params.shrink = 0.5;
        ts_params.adapt_m = false;
        ts_params.m_max = params.m;
        ts_params.m_step = 0;

        V_m.resize(static_cast<size_t>(n) * (allocated_m_max + 1), 0.0);
        w.resize(n, 0.0);
        y.resize(n, 0.0);
    }

    void init_q1(const std::vector<double>& v_host) {
        double norm_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_sq += v_host[i] * v_host[i];
        }
        v_norm = std::sqrt(norm_sq);

        double scale = (v_norm > 1e-14) ? (1.0 / v_norm) : 1.0;
        for (int i = 0; i < n; ++i) {
            V_m[i] = v_host[i] * scale;  // q1 = V_m[:,0]
        }
    }

    void spmv_qj(int j) {
        double* qj = V_m.data() + static_cast<size_t>(j) * n;
        std::fill(w.begin(), w.end(), 0.0);
        spmv_csr(A.row_ptr.data(), A.col_idx.data(), A.values.data(),
                 qj, w.data(), n);
    }

    void orthogonalize_mgs(int j) {
        int m_lim = static_cast<int>(H_m.cols());
        for (int i = 0; i <= j && i < m_lim; ++i) {
            double* qi = V_m.data() + static_cast<size_t>(i) * n;
            double h_ij = 0.0;
            for (int k = 0; k < n; ++k) {
                h_ij += w[k] * qi[k];
            }
            H_m(i, j) = h_ij;
            for (int k = 0; k < n; ++k) {
                w[k] -= h_ij * qi[k];
            }
        }

        if (params.reorthogonalize) {
            for (int i = 0; i <= j && i < m_lim; ++i) {
                double* qi = V_m.data() + static_cast<size_t>(i) * n;
                double corr = 0.0;
                for (int k = 0; k < n; ++k) {
                    corr += w[k] * qi[k];
                }
                H_m(i, j) += corr;
                for (int k = 0; k < n; ++k) {
                    w[k] -= corr * qi[k];
                }
            }
        }
    }

    bool normalize_new_vector(int j) {
        double norm_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_sq += w[i] * w[i];
        }
        double global_norm = std::sqrt(norm_sq);

        if (global_norm < 1e-14 || std::isnan(global_norm) || std::isinf(global_norm)) {
            if (j + 1 < H_m.rows() && j < H_m.cols()) {
                H_m(j + 1, j) = 0.0;
            }
            return false;
        }

        double scale = 1.0 / global_norm;
        double* q_next = V_m.data() + static_cast<size_t>(j + 1) * n;
        for (int i = 0; i < n; ++i) {
            q_next[i] = w[i] * scale;
        }

        if (j + 1 < H_m.rows() && j < H_m.cols()) {
            H_m(j + 1, j) = global_norm;
        }
        return true;
    }

    void small_expm_and_lift(double segment_t, int k_used) {
        if (k_used <= 0) return;

        e1 = Eigen::VectorXd::Zero(k_used + 1);
        e1(0) = 1.0;

        Eigen::MatrixXd H_m_square = H_m.block(0, 0, k_used, k_used);
        Eigen::MatrixXd exp_tH_square = matrix_exp_stable(segment_t * H_m_square);

        Eigen::MatrixXd exp_tH = Eigen::MatrixXd::Zero(k_used + 1, k_used + 1);
        exp_tH.block(0, 0, k_used, k_used) = exp_tH_square;

        wH = exp_tH * e1;

        std::fill(y.begin(), y.end(), 0.0);
        for (int i = 0; i <= k_used; ++i) {
            if (std::abs(wH(i)) > 1e-15) {
                const double alpha = v_norm * wH(i);
                double* V_i = V_m.data() + static_cast<size_t>(i) * n;
                for (int k = 0; k < n; ++k) {
                    y[k] += alpha * V_i[k];
                }
            }
        }
    }

    double residual_estimate(double segment_t, int k_used) {
        if (k_used <= 0) return 0.0;
        Eigen::MatrixXd H_m_square = H_m.block(0, 0, k_used, k_used);
        Eigen::MatrixXd exp_tH_square = matrix_exp_stable(segment_t * H_m_square);

        Eigen::VectorXd e1_small = Eigen::VectorXd::Zero(k_used);
        e1_small(0) = 1.0;

        Eigen::VectorXd exp_tH_e1 = exp_tH_square * e1_small;
        double e_k_exp_tH_e1 = exp_tH_e1(k_used - 1);

        double h_k_plus_1_k = H_m(k_used, k_used - 1);
        return std::abs(h_k_plus_1_k * e_k_exp_tH_e1) * v_norm;
    }

    void restart_from_y() {
        double norm_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_sq += y[i] * y[i];
        }
        v_norm = std::sqrt(norm_sq);

        double scale = (v_norm > 1e-14) ? (1.0 / v_norm) : 1.0;
        for (int i = 0; i < n; ++i) {
            V_m[i] = y[i] * scale;
        }
    }

    void compute_expmv(const std::vector<double>& v_host, std::vector<double>& y_host) {
        auto total_compute_start = std::chrono::steady_clock::now();

        init_q1(v_host);

        double total_t = params.t;
        double dt = std::min(std::max(ts_params.dt_init, ts_params.dt_min), ts_params.dt_max);
        double t_done = 0.0;
        int accepted_steps = 0;
        converged = false;

        while (t_done < total_t && accepted_steps < ts_params.max_steps) {
            dt = std::min(dt, total_t - t_done);
            int rejects = 0;
            int m_current = params.m;

            while (true) {
                H_m.setZero(m_current + 1, m_current);
                int k_used = 0;
                bool breakdown = false;

                for (int j = 0; j < m_current; ++j) {
                    spmv_qj(j);
                    orthogonalize_mgs(j);
                    bool ok = normalize_new_vector(j);
                    k_used = j + 1;
                    if (!ok) {
                        breakdown = true;
                        break;
                    }
                }

                small_expm_and_lift(dt, k_used);
                double err = residual_estimate(dt, k_used);
                if (breakdown) {
                    err = 0.0;
                }

                std::cout << "[STEP] t=" << t_done << " dt=" << dt
                          << " err=" << err << " m_used=" << k_used
                          << " accepts=" << accepted_steps
                          << " rejects=" << rejects << std::endl;

                if (err <= ts_params.tol_step || breakdown) {
                    t_done += dt;
                    accepted_steps++;
                    last_residual = err;

                    if (t_done >= total_t) {
                        converged = true;
                        break;
                    }

                    restart_from_y();

                    if (err < 0.1 * ts_params.tol_step) {
                        double dt_new = dt * ts_params.grow * ts_params.safety;
                        dt = std::min(dt_new, ts_params.dt_max);
                    }
                    break;
                } else {
                    rejects++;
                    if (rejects >= ts_params.max_rejects) {
                        std::cerr << "[STEP] rejected too many times, aborting" << std::endl;
                        converged = false;
                        last_residual = err;
                        t_done = total_t;
                        break;
                    }
                    if (ts_params.adapt_m && (m_current + ts_params.m_step) <= ts_params.m_max && ts_params.m_step > 0) {
                        m_current += ts_params.m_step;
                    } else {
                        double dt_new = std::max(dt * ts_params.shrink * ts_params.safety, ts_params.dt_min);
                        dt = dt_new;
                        m_current = params.m;
                    }
                    continue;
                }
            }
        }

        restarts_done = accepted_steps;

        for (int i = 0; i < n; ++i) {
            y_host[i] = y[i];
        }

        const double N = static_cast<double>(n);
        const double nnz = static_cast<double>(A.nnz);
        const double m_val = static_cast<double>(params.m);
        const double S = static_cast<double>(accepted_steps);

        std::cout << "[FLOPs] N=" << N << " nnz=" << nnz << " m=" << m_val << " steps=" << S << std::endl;

        auto total_compute_end = std::chrono::steady_clock::now();
        double total_compute_ms =
            std::chrono::duration<double, std::milli>(total_compute_end - total_compute_start).count();
        std::cout << "METRIC total_compute_expmv_ms=" << total_compute_ms << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "Starting matrix_exp_cpu (CPU reference)." << std::endl;

    bool use_file = false;
    std::string matrix_file;
    int test_size = 1000;
    int m_param = -1;
    int max_restarts_cli = MAX_RESTARTS;
    double t_cli = 1.0;

    std::vector<std::string> args(argv + 1, argv + argc);
    size_t pos_idx = 0;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--m" && i + 1 < args.size()) {
            m_param = std::atoi(args[i + 1].c_str());
            ++i;
            continue;
        }
        if (a == "--max-restarts" && i + 1 < args.size()) {
            max_restarts_cli = std::atoi(args[i + 1].c_str());
            ++i;
            continue;
        }
        if (a == "--t" && i + 1 < args.size()) {
            t_cli = std::atof(args[i + 1].c_str());
            ++i;
            continue;
        }

        if (pos_idx == 0) {
            if (a.find(".") != std::string::npos || a.find("/") != std::string::npos || a.rfind(".mtx") != std::string::npos) {
                use_file = true;
                matrix_file = a;
            } else {
                test_size = std::atoi(a.c_str());
                if (test_size <= 0) {
                    std::cerr << "Error: Matrix size must be a positive integer. Using default 1000." << std::endl;
                    test_size = 1000;
                }
            }
            ++pos_idx;
            continue;
        }
        if (pos_idx == 1) {
            m_param = std::atoi(a.c_str());
            ++pos_idx;
            continue;
        }
        if (pos_idx == 2) {
            max_restarts_cli = std::atoi(a.c_str());
            ++pos_idx;
            continue;
        }
    }

    try {
        CSRHost test_matrix;

        auto ends_with = [](const std::string& s, const std::string& suffix) -> bool {
            return s.size() >= suffix.size() &&
                   s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        auto has_bincsr_magic = [](const std::string& path) -> bool {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) return false;
            unsigned char hdr[8] = {0};
            f.read(reinterpret_cast<char*>(hdr), 8);
            if (!f || f.gcount() != 8) return false;
            const unsigned char magic[8] = {'C', 'S', 'R', 0, 0, 0, 0, 1};
            return std::memcmp(hdr, magic, 8) == 0;
        };

        if (use_file) {
            std::cout << "Loading matrix from file: " << matrix_file << std::endl;
            const bool is_bincsr_ext = ends_with(matrix_file, ".bincsr");
            const bool is_mtx_ext = ends_with(matrix_file, ".mtx") || ends_with(matrix_file, ".mtx.gz");
            bool loaded = false;

            if (is_bincsr_ext || has_bincsr_magic(matrix_file)) {
                test_matrix.from_binary_csr(matrix_file);
                loaded = true;
                std::cout << "Detected Binary CSR v1 input" << std::endl;
            } else if (is_mtx_ext) {
                test_matrix.from_matrix_market(matrix_file);
                loaded = true;
                std::cout << "Detected MatrixMarket input" << std::endl;
            }

            if (!loaded) {
                std::cerr << "Error: Could not detect matrix format for file " << matrix_file << std::endl;
                return EXIT_FAILURE;
            }
            std::cout << "Loaded matrix: " << test_matrix.rows << "x" << test_matrix.cols
                      << " with " << test_matrix.nnz << " non-zeros" << std::endl;
        } else {
            std::cout << "Matrix size (generated): " << test_size << std::endl;
            test_matrix.rows = test_size;
            test_matrix.cols = test_size;
            test_matrix.nnz = test_size;

            test_matrix.row_ptr.resize(test_matrix.rows + 1);
            test_matrix.col_idx.resize(test_matrix.nnz);
            test_matrix.values.resize(test_matrix.nnz);

            for (int i = 0; i < test_matrix.rows; ++i) {
                test_matrix.row_ptr[i] = i;
                test_matrix.col_idx[i] = i;
                test_matrix.values[i] = 1.0 + (double)i / test_matrix.rows;
            }
            test_matrix.row_ptr[test_matrix.rows] = test_matrix.nnz;

            std::cout << "Created test matrix: " << test_matrix.rows << "x" << test_matrix.cols
                      << " with " << test_matrix.nnz << " non-zeros" << std::endl;
        }

        double max_abs_A = 0.0;
        for (double v : test_matrix.values) {
            max_abs_A = std::max(max_abs_A, std::abs(v));
        }
        if (max_abs_A <= 0.0) {
            max_abs_A = 1.0;
        }
        double t_safe = t_cli;
        double t_limit = 10.0 / max_abs_A;
        if (t_safe > t_limit) {
            std::cout << "[WARN] Requested t=" << t_safe << " is large for max|A|=" << max_abs_A
                      << ". Clamping t to " << t_limit << std::endl;
            t_safe = t_limit;
        }

        int m_val = m_param;
        if (m_val <= 0) {
            m_val = std::min(30, std::max(5, test_matrix.rows - 1));
        }
        m_val = std::max(1, std::min(m_val, test_matrix.rows - 1));

        ArnoldiParams params(m_val, t_safe, 1e-6, max_restarts_cli, true);

        std::cout << "Arnoldi params: m=" << params.m
                  << " t=" << params.t
                  << " tol=" << params.tol
                  << " max_restarts=" << params.max_restarts
                  << std::endl;

        ArnoldiRunnerCPU runner(test_matrix, params);

        std::vector<double> v_host(test_matrix.rows, 0.1);
        std::vector<double> y_host(test_matrix.rows, 0.0);

        runner.compute_expmv(v_host, y_host);

        if (const char* out_path = std::getenv("OUTPUT_Y_FILE"); out_path && out_path[0] != '\0') {
            std::ofstream ofs(out_path);
            if (!ofs) {
                std::cerr << "Error: could not open OUTPUT_Y_FILE=" << out_path << " for writing" << std::endl;
                return EXIT_FAILURE;
            }
            ofs.setf(std::ios::scientific);
            ofs.precision(16);
            for (double v : y_host) {
                ofs << v << "\n";
            }
            ofs.close();
            std::cout << "Saved y to " << out_path << std::endl;
        }

        std::cout << "Computation completed. Sample results:" << std::endl;
        for (int i = 0; i < std::min(10, (int)y_host.size()); ++i) {
            std::cout << "y[" << i << "] = " << y_host[i] << std::endl;
        }
        std::cout << "Arnoldi summary: "
                  << "m=" << m_val
                  << ", max_restarts=" << max_restarts_cli
                  << ", restarts_done=" << runner.restarts_done
                  << ", converged=" << (runner.converged ? "yes" : "no")
                  << ", residual=" << runner.last_residual
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
