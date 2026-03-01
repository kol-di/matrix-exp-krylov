#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>
#include <iterator>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>

// CUDA includes
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <nvToolsExt.h>

// Eigen for small matrix exponentiation (using only stable modules)
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

// Stable matrix exponential implementation using only core Eigen modules
Eigen::MatrixXd matrix_exp_stable(const Eigen::MatrixXd& A) {
    // Use Eigen's matrix exponential (scaling-squaring + Pade), more stable for non-normal matrices
    return A.exp();
}

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

// Error checking macro
#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(func) \
{ \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSPARSE(func) \
{ \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_NCCL(func) \
{ \
    ncclResult_t status = (func); \
    if (status != ncclSuccess) { \
        fprintf(stderr, "NCCL Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Global constants
const double ARNOLDI_TOL = 1e-8; // Tolerance for Arnoldi residual
const int MAX_RESTARTS = 10;     // Default maximum number of restarts
const double C_EXP_FLOP_FACTOR = 1.0; // Heuristic factor for CPU expm cost (scaling-squaring/Pade), per spec

// CSRHost class for host-side matrix handling
struct CSRHost {
    int rows;
    int cols;
    long long nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;

    CSRHost() : rows(0), cols(0), nnz(0) {}

    // Method to read matrix from Matrix Market format
    void from_matrix_market(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        // === Parse header ===
        bool header_parsed = false;
        std::string field, symmetry, format, object;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            ss >> object;
            // Normalize tokens to lowercase for robustness
            std::transform(object.begin(), object.end(), object.begin(), ::tolower);
            if (object != "%%matrixmarket") {
                continue; // keep searching until header is found
            }
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

        // === Read size line (skip comments/empty) ===
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
            // Convert to 0-based indices
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

        // Sort entries by row then column
        std::sort(entries.begin(), entries.end());

        // Deduplicate by summing values of identical (row,col)
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
            std::cerr << "Error: Binary CSR file too small (expected at least 64 bytes): " << filename << std::endl;
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
        const uint32_t index_dtype = load_le_u32(header + 40); // 1=u32, 2=u64
        const uint32_t value_dtype = load_le_u32(header + 44); // 1=f32, 2=f64
        const uint64_t reserved0 = load_le_u64(header + 48);
        const uint64_t reserved1 = load_le_u64(header + 56);

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
        if (reserved0 != 0 || reserved1 != 0) {
            std::cerr << "Error: Reserved header fields must be zero in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        if (nrows_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            ncols_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            nnz_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            std::cerr << "Error: Matrix dimensions/nnz exceed int32 limits used by solver in " << filename << std::endl;
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
            std::cerr << "Error: Binary CSR size mismatch in " << filename
                      << " (expected " << expected_total << " bytes, got " << file_size << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::vector<uint64_t> indptr64(indptr_len, 0);
        std::vector<uint64_t> indices64(nnz_u64, 0);
        std::vector<double> data64(nnz_u64, 0.0);

        if (index_dtype == 1) {
            std::vector<uint32_t> tmp(indptr_len);
            file.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(uint32_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indptr(u32) from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < tmp.size(); ++i) indptr64[i] = static_cast<uint64_t>(tmp[i]);

            std::vector<uint32_t> tmp_idx(nnz_u64);
            file.read(reinterpret_cast<char*>(tmp_idx.data()), static_cast<std::streamsize>(tmp_idx.size() * sizeof(uint32_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indices(u32) from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < tmp_idx.size(); ++i) indices64[i] = static_cast<uint64_t>(tmp_idx[i]);
        } else {
            file.read(reinterpret_cast<char*>(indptr64.data()), static_cast<std::streamsize>(indptr64.size() * sizeof(uint64_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indptr(u64) from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            file.read(reinterpret_cast<char*>(indices64.data()), static_cast<std::streamsize>(indices64.size() * sizeof(uint64_t)));
            if (!file) {
                std::cerr << "Error: Failed to read indices(u64) from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        if (value_dtype == 1) {
            std::vector<float> tmp_val(nnz_u64);
            file.read(reinterpret_cast<char*>(tmp_val.data()), static_cast<std::streamsize>(tmp_val.size() * sizeof(float)));
            if (!file) {
                std::cerr << "Error: Failed to read data(f32) from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
            for (size_t i = 0; i < tmp_val.size(); ++i) data64[i] = static_cast<double>(tmp_val[i]);
        } else {
            file.read(reinterpret_cast<char*>(data64.data()), static_cast<std::streamsize>(data64.size() * sizeof(double)));
            if (!file) {
                std::cerr << "Error: Failed to read data(f64) from " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        rows = static_cast<int>(nrows_u64);
        cols = static_cast<int>(ncols_u64);
        nnz = static_cast<long long>(nnz_u64);

        if (indptr64.size() != static_cast<size_t>(rows + 1)) {
            std::cerr << "Error: indptr length mismatch in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        if (indptr64[0] != 0) {
            std::cerr << "Error: indptr[0] must be 0 in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        for (int r = 0; r < rows; ++r) {
            if (indptr64[r] > indptr64[r + 1]) {
                std::cerr << "Error: indptr is not non-decreasing at row " << r << " in " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (indptr64[rows] != nnz_u64) {
            std::cerr << "Error: indptr[nrows] != nnz in " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        for (uint64_t k = 0; k < nnz_u64; ++k) {
            if (indices64[k] >= ncols_u64) {
                std::cerr << "Error: Column index out of range at k=" << k << " in " << filename << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        const bool flag_sorted = (flags & (1u << 0)) != 0;
        const bool flag_no_dup = (flags & (1u << 1)) != 0;
        const bool flag_symmetric_upper = (flags & (1u << 2)) != 0;

        std::cout << "Binary CSR header: "
                  << "version=" << version
                  << " flags=0x" << std::hex << flags << std::dec
                  << " index_dtype=" << (index_dtype == 1 ? "u32" : "u64")
                  << " value_dtype=" << (value_dtype == 1 ? "f32" : "f64")
                  << " symmetric_upper=" << (flag_symmetric_upper ? "yes" : "no")
                  << std::endl;

        if (flag_sorted || flag_no_dup) {
            for (int r = 0; r < rows; ++r) {
                uint64_t start = indptr64[r];
                uint64_t end = indptr64[r + 1];
                for (uint64_t p = start + 1; p < end; ++p) {
                    if (flag_sorted && indices64[p - 1] > indices64[p]) {
                        std::cerr << "Error: Row " << r << " is not sorted but sorted flag is set in " << filename << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    if (flag_no_dup && indices64[p - 1] == indices64[p]) {
                        std::cerr << "Error: Row " << r << " has duplicates but no_duplicates flag is set in " << filename << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }

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
                if (indptr64[r] > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                    std::cerr << "Error: indptr value exceeds int32 at row " << r << " in " << filename << std::endl;
                    exit(EXIT_FAILURE);
                }
                row_ptr[r] = static_cast<int>(indptr64[r]);
            }
            for (uint64_t k = 0; k < nnz_u64; ++k) {
                if (indices64[k] > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                    std::cerr << "Error: index value exceeds int32 at k=" << k << " in " << filename << std::endl;
                    exit(EXIT_FAILURE);
                }
                col_idx[k] = static_cast<int>(indices64[k]);
                values[k] = data64[k];
            }
        }
    }

    // Partition A by rows for num_gpus
    std::vector<CSRHost> partition_rows(int num_gpus) const {
        std::vector<CSRHost> partitions(num_gpus);
        int rows_per_gpu = rows / num_gpus;
        int remainder_rows = rows % num_gpus;

        int current_row_start = 0;
        for (int i = 0; i < num_gpus; ++i) {
            int current_rows = rows_per_gpu + (i < remainder_rows ? 1 : 0);
            
            partitions[i].rows = current_rows;
            partitions[i].cols = cols;
            
            if (current_rows == 0) {
                partitions[i].nnz = 0;
                partitions[i].row_ptr.assign(1, 0);
                continue;
            }

            int current_row_end = current_row_start + current_rows;
            long long current_nnz = row_ptr[current_row_end] - row_ptr[current_row_start];
            
            partitions[i].nnz = current_nnz;
            partitions[i].row_ptr.assign(current_rows + 1, 0);

            // Copy row_ptr, col_idx, values
            for (int k = 0; k < current_rows; ++k) {
                partitions[i].row_ptr[k] = row_ptr[current_row_start + k] - row_ptr[current_row_start];
            }
            partitions[i].row_ptr[current_rows] = current_nnz; // Last element

            partitions[i].col_idx.assign(col_idx.begin() + row_ptr[current_row_start], col_idx.begin() + row_ptr[current_row_end]);
            partitions[i].values.assign(values.begin() + row_ptr[current_row_start], values.begin() + row_ptr[current_row_end]);

            current_row_start = current_row_end;
        }
        return partitions;
    }

    // Build owner/ghost maps for communication
    // This function will need to determine for each column index in off-diag, which GPU owns it.
    // And for each GPU, which other GPUs need its rows (ghosts).
    // The structure needs to support efficient p2p copies.
    struct GhostMap {
        std::vector<int> ghost_indices; // Indices of elements needed from other GPUs
        std::vector<int> ghost_owners; // Which GPU owns each ghost element
        std::map<int, std::vector<int>> owner_to_consumer_map; // map from owner GPU to list of ghost indices that this GPU needs (RECEIVE map)
        std::map<int, std::vector<int>> consumer_to_owner_map; // map from consumer GPU to list of ghost indices that this GPU owns (SEND map)
        std::vector<int> send_offsets; // Offsets for sending data in d_ghost_send_buffer (for each peer)
        std::vector<int> recv_offsets; // Offsets for receiving data in d_ghost_recv_buffer (for each peer)
        std::vector<int> send_counts; // Counts for sending data (for each peer)
        std::vector<int> recv_counts; // Counts for receiving data (for each peer)

        // New members for more explicit send/receive logic
        std::vector<int> outgoing_global_col_indices; // Flattened list of global column indices this GPU needs to send
        std::vector<int> incoming_global_col_indices; // Flattened list of global column indices this GPU needs to receive
        std::map<int, std::pair<size_t, size_t>> outgoing_peer_info; // Map: consumer_gpu_id -> {offset in outgoing_global_col_indices, count}
        std::map<int, std::pair<size_t, size_t>> incoming_peer_info; // Map: owner_gpu_id -> {offset in incoming_global_col_indices, count}
        size_t total_recv_size; // Total size of d_ghost_recv_buffer needed
        size_t total_send_size; // Total size of d_ghost_send_buffer needed
    };

    GhostMap build_owner_ghost_maps(const std::vector<CSRHost>& partitions, int my_gpu_id, int num_gpus) const {
        GhostMap ghost_map;

        // Pre-size offset/count vectors to avoid out-of-bounds writes
        ghost_map.send_offsets.assign(num_gpus, 0);
        ghost_map.send_counts.assign(num_gpus, 0);
        ghost_map.recv_offsets.assign(num_gpus, 0);
        ghost_map.recv_counts.assign(num_gpus, 0);

        // Determine global row ranges for each GPU
        std::vector<std::pair<int, int>> gpu_row_ranges(num_gpus);
        int current_global_row = 0;
        for (int i = 0; i < num_gpus; ++i) {
            gpu_row_ranges[i] = {current_global_row, current_global_row + partitions[i].rows - 1};
            current_global_row += partitions[i].rows;
        }

        // ============================ Building RECEIVE maps for my_gpu_id ===============================
        // Identify ghost columns for current GPU (elements it needs to receive)
        const CSRHost& my_partition = partitions[my_gpu_id];
        
        for (int k = 0; k < my_partition.nnz; ++k) {
            int col = my_partition.col_idx[k];
            // Check if the column owner is not this GPU
            bool is_local_col = (col >= gpu_row_ranges[my_gpu_id].first && col <= gpu_row_ranges[my_gpu_id].second);
            if (!is_local_col) {
                // Determine owner of this ghost column
                for (int i = 0; i < num_gpus; ++i) {
                    if (col >= gpu_row_ranges[i].first && col <= gpu_row_ranges[i].second) {
                        ghost_map.owner_to_consumer_map[i].push_back(col); // This GPU (my_gpu_id) needs 'col' from GPU 'i'
                        break;
                    }
                }
            }
        }

        // Sort/unique per-owner lists and flatten in owner order to make offsets consistent
        size_t current_recv_offset = 0;
        for (int p_id = 0; p_id < num_gpus; ++p_id) {
            if (p_id == my_gpu_id) continue;
            auto& lst = ghost_map.owner_to_consumer_map[p_id];
            std::sort(lst.begin(), lst.end());
            lst.erase(std::unique(lst.begin(), lst.end()), lst.end());

            size_t count = lst.size();
            if (count > 0) {
                ghost_map.incoming_peer_info[p_id] = {current_recv_offset, count};
                ghost_map.recv_offsets[p_id] = current_recv_offset;
                ghost_map.recv_counts[p_id] = count;

                // Append to the flattened incoming list in the same order
                ghost_map.incoming_global_col_indices.insert(
                    ghost_map.incoming_global_col_indices.end(),
                    lst.begin(), lst.end());

                current_recv_offset += count;
            }
        }
        ghost_map.total_recv_size = current_recv_offset;

        // ============================ Building SEND maps for my_gpu_id ==================================
        // Iterate through all OTHER partitions and see what they need from MY_GPU_ID
        for (int p_id = 0; p_id < num_gpus; ++p_id) {
            if (p_id == my_gpu_id) continue;
            
            const CSRHost& other_partition = partitions[p_id];
            for (int k = 0; k < other_partition.nnz; ++k) {
                int col = other_partition.col_idx[k];
                // If 'col' is owned by 'my_gpu_id' and needed by 'p_id'
                if (col >= gpu_row_ranges[my_gpu_id].first && col <= gpu_row_ranges[my_gpu_id].second) {
                    bool is_local_to_other = (col >= gpu_row_ranges[p_id].first && col <= gpu_row_ranges[p_id].second);
                    if (!is_local_to_other) {
                        ghost_map.consumer_to_owner_map[p_id].push_back(col); // This GPU (my_gpu_id) needs to send 'col' to GPU 'p_id'
                    }
                }
            }
            // Sort and unique the consumer_to_owner_map for this peer
            std::sort(ghost_map.consumer_to_owner_map[p_id].begin(), ghost_map.consumer_to_owner_map[p_id].end());
            ghost_map.consumer_to_owner_map[p_id].erase(
                std::unique(ghost_map.consumer_to_owner_map[p_id].begin(), ghost_map.consumer_to_owner_map[p_id].end()),
                ghost_map.consumer_to_owner_map[p_id].end()
            );
        }
        // Flatten outgoing lists in consumer order to match offsets
        size_t current_send_offset = 0;
        for (int p_id = 0; p_id < num_gpus; ++p_id) {
            if (p_id == my_gpu_id) continue;
            auto& lst = ghost_map.consumer_to_owner_map[p_id];
            size_t count = lst.size();
            if (count > 0) {
                ghost_map.outgoing_peer_info[p_id] = {current_send_offset, count};
                ghost_map.send_offsets[p_id] = current_send_offset;
                ghost_map.send_counts[p_id] = count;

                ghost_map.outgoing_global_col_indices.insert(
                    ghost_map.outgoing_global_col_indices.end(),
                    lst.begin(), lst.end());

                current_send_offset += count;
            }
        }
        ghost_map.total_send_size = current_send_offset;
        
        return ghost_map;
    }
};

// Validate that SEND and RECEIVE ghost plans are consistent between all GPU pairs
void validate_ghost_maps(const std::vector<CSRHost::GhostMap>& ghost_maps, int num_gpus) {
    auto normalize = [](std::vector<int> v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    };
    auto sample_vec = [](const std::vector<int>& v) {
        std::ostringstream oss;
        size_t limit = 10;
        for (size_t i = 0; i < v.size() && i < limit; ++i) {
            if (i > 0) oss << ",";
            oss << v[i];
        }
        if (v.size() > limit) oss << " ...";
        return oss.str();
    };

    for (int owner = 0; owner < num_gpus; ++owner) {
        for (int consumer = 0; consumer < num_gpus; ++consumer) {
            if (owner == consumer) continue;

            std::vector<int> send, recv;
            auto send_it = ghost_maps[owner].consumer_to_owner_map.find(consumer);
            if (send_it != ghost_maps[owner].consumer_to_owner_map.end()) {
                send = normalize(send_it->second);
            }
            auto recv_it = ghost_maps[consumer].owner_to_consumer_map.find(owner);
            if (recv_it != ghost_maps[consumer].owner_to_consumer_map.end()) {
                recv = normalize(recv_it->second);
            }

            if (send != recv) {
                std::vector<int> diff_send, diff_recv;
                std::set_difference(send.begin(), send.end(), recv.begin(), recv.end(), std::back_inserter(diff_send));
                std::set_difference(recv.begin(), recv.end(), send.begin(), send.end(), std::back_inserter(diff_recv));

                std::cerr << "[ERROR] Ghost map mismatch between owner " << owner
                          << " and consumer " << consumer << std::endl;
                std::cerr << "  send size: " << send.size() << " recv size: " << recv.size() << std::endl;
                if (!diff_send.empty()) {
                    std::cerr << "  Present only in send (first entries): " << sample_vec(diff_send) << std::endl;
                }
                if (!diff_recv.empty()) {
                    std::cerr << "  Present only in recv (first entries): " << sample_vec(diff_recv) << std::endl;
                }
                exit(EXIT_FAILURE);
            }
        }
    }
}

// DeviceContext class for GPU resource management
struct DeviceContext {
    int device_id;
    cudaStream_t stream_compute, stream_comm, stream_reduce;
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;

    // Device pointers for CSR matrix
    int* d_row_ptr;
    int* d_col_idx;
    double* d_values;

    // cuSPARSE matrix and vector descriptors
    cusparseSpMatDescr_t matA_descr; // legacy (full) descriptor, kept for cleanup safety
    cusparseSpMatDescr_t matA_on_descr;
    cusparseSpMatDescr_t matA_off_descr;
    cusparseDnVecDescr_t vec_v_descr, vec_w_descr, vec_y_descr;
    cusparseDnVecDescr_t vec_q_local_descr;
    cusparseDnVecDescr_t vec_q_ghost_descr[2];

    // Device pointers for Arnoldi vectors and work vectors
    // V_m will store the Arnoldi basis vectors as columns
    double* d_V_m; // size: local_rows * m
    double* d_q;   // current Arnoldi vector q_j (local part)
    double* d_w;   // work vector A*q_j
    double* d_y;   // final result vector

    // Ghost buffers for communication (double buffering)
    double* d_ghost_recv_buffer[2]; // Two buffers for double buffering
    double* d_ghost_send_buffer[2]; // Two buffers for double buffering
    cudaEvent_t ghost_recv_ready[2];   // events to signal recv completion per buffer
    cudaEvent_t send_ready[2];         // events to signal send buffer readiness per buffer
    cudaEvent_t send_done[2];          // events to signal all outgoing copies reading send buffer are enqueued/completed

    int local_rows; // Number of rows owned by this GPU
    int local_nnz;  // Number of non-zero elements in local CSR partition

    // Device-side scalars for AllReduce and temporary computations
    double* d_scalar_host_ptr; // Pinned host scratch for small D2H copies
    double* d_scalar_device;   // Device buffer used in NCCL AllReduce ops
    double* d_scalar_device_aux; // Additional device scalar for transforms
    
    // cuSPARSE work buffer for SpMV operations
    void* d_spmv_buffer;
    size_t spmv_buffer_size;

    // Host-pinned buffer for collecting h_ij (GPU0 only used; allocate 2*(m+1))
    double* h_pinned_h;
    size_t h_pinned_capacity;
    size_t h_pinned_stride;   // stride between h_first and h_corr (m_arnoldi+1)

    // On/off-diagonal CSR storage
    int* d_row_ptr_on;
    int* d_col_idx_on;
    double* d_values_on;
    int* d_row_ptr_off;
    int* d_col_idx_off;
    double* d_values_off;

    // Persisted device copies of ghost index lists
    int* d_outgoing_global_col_indices;
    int* d_incoming_global_col_indices;

    DeviceContext(int dev_id) : 
        device_id(dev_id), 
        stream_compute(nullptr), stream_comm(nullptr), stream_reduce(nullptr),
        cublas_handle(nullptr), cusparse_handle(nullptr),
        d_row_ptr(nullptr), d_col_idx(nullptr), d_values(nullptr),
        matA_descr(nullptr), matA_on_descr(nullptr), matA_off_descr(nullptr),
        vec_v_descr(nullptr), vec_w_descr(nullptr), vec_y_descr(nullptr),
        vec_q_local_descr(nullptr),
        d_V_m(nullptr), d_q(nullptr), d_w(nullptr), d_y(nullptr),
        d_scalar_host_ptr(nullptr), d_scalar_device(nullptr), d_scalar_device_aux(nullptr),
        d_spmv_buffer(nullptr), spmv_buffer_size(0),
        h_pinned_h(nullptr), h_pinned_capacity(0), h_pinned_stride(0),
        d_row_ptr_on(nullptr), d_col_idx_on(nullptr), d_values_on(nullptr),
        d_row_ptr_off(nullptr), d_col_idx_off(nullptr), d_values_off(nullptr),
        d_outgoing_global_col_indices(nullptr), d_incoming_global_col_indices(nullptr)
    {
        d_ghost_recv_buffer[0] = nullptr;
        d_ghost_recv_buffer[1] = nullptr;
        d_ghost_send_buffer[0] = nullptr;
        d_ghost_send_buffer[1] = nullptr;
        ghost_recv_ready[0] = nullptr;
        ghost_recv_ready[1] = nullptr;
        send_ready[0] = nullptr;
        send_ready[1] = nullptr;
        send_done[0] = nullptr;
        send_done[1] = nullptr;
        vec_q_ghost_descr[0] = nullptr;
        vec_q_ghost_descr[1] = nullptr;
    }

    // Initialize cuBLAS/cuSPARSE handles
    void init_handles() {
        CHECK_CUDA(cudaSetDevice(device_id));
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
        CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
    }

    // Create CUDA streams
    void create_streams() {
        CHECK_CUDA(cudaSetDevice(device_id));
        CHECK_CUDA(cudaStreamCreate(&stream_compute));
        CHECK_CUDA(cudaStreamCreate(&stream_comm));
        CHECK_CUDA(cudaStreamCreate(&stream_reduce));
        
        // Set streams for cuBLAS and cuSPARSE handles
        CHECK_CUBLAS(cublasSetStream(cublas_handle, stream_compute));
        CHECK_CUSPARSE(cusparseSetStream(cusparse_handle, stream_compute));

        // Create events for ghost readiness (no timing to reduce overhead)
        CHECK_CUDA(cudaEventCreateWithFlags(&ghost_recv_ready[0], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&ghost_recv_ready[1], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&send_ready[0], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&send_ready[1], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&send_done[0], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&send_done[1], cudaEventDisableTiming));
        // Initial record so first wait is safe/no-op
        CHECK_CUDA(cudaEventRecord(send_done[0], stream_comm));
        CHECK_CUDA(cudaEventRecord(send_done[1], stream_comm));
    }

    // Allocate device memory and copy CSR partition from host
    int allocated_m_max; // tracks allocation bound for V_m/pinned buffers

    void alloc_from(const CSRHost& part, int global_cols, int m_alloc_max) {
        CHECK_CUDA(cudaSetDevice(device_id));

        local_rows = part.rows;
        local_nnz = part.nnz;
        
        // Debug: print allocation info
        std::cout << "GPU " << device_id << " allocating: local_rows=" << local_rows 
                  << ", local_nnz=" << local_nnz << std::endl;

        // Allocate and copy CSR matrix data
        if (local_nnz > 0) {
            CHECK_CUDA(cudaMalloc(&d_row_ptr, sizeof(int) * (part.rows + 1)));
            CHECK_CUDA(cudaMalloc(&d_col_idx, sizeof(int) * part.nnz));
            CHECK_CUDA(cudaMalloc(&d_values, sizeof(double) * part.nnz));

            CHECK_CUDA(cudaMemcpyAsync(d_row_ptr, part.row_ptr.data(), sizeof(int) * (part.rows + 1), cudaMemcpyHostToDevice, stream_compute));
            CHECK_CUDA(cudaMemcpyAsync(d_col_idx, part.col_idx.data(), sizeof(int) * part.nnz, cudaMemcpyHostToDevice, stream_compute));
            CHECK_CUDA(cudaMemcpyAsync(d_values, part.values.data(), sizeof(double) * part.nnz, cudaMemcpyHostToDevice, stream_compute));
        }

        // Allocate memory for work vectors (q, w, y) and basis V_m
        CHECK_CUDA(cudaMalloc(&d_q, sizeof(double) * local_rows));
        CHECK_CUDA(cudaMalloc(&d_w, sizeof(double) * local_rows));
        CHECK_CUDA(cudaMalloc(&d_y, sizeof(double) * local_rows));
        // V_m sized for maximum planned Arnoldi dimension (m_alloc_max + 1 vectors)
        allocated_m_max = std::max(1, m_alloc_max);
        CHECK_CUDA(cudaMalloc(&d_V_m, sizeof(double) * local_rows * (allocated_m_max + 1)));

        // Allocate pinned host memory for scalar reduction results
        CHECK_CUDA(cudaMallocHost(&d_scalar_host_ptr, sizeof(double)));
        // Allocate device memory for cuBLAS scalar results
        CHECK_CUDA(cudaMalloc(&d_scalar_device, sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_scalar_device_aux, sizeof(double)));
    }

    // Create cuSPARSE descriptors (on/off split) and allocate ghost buffers
    void create_descriptors(const CSRHost& part, int global_cols, const CSRHost::GhostMap& ghost_map, int global_row_offset, int /*m_arnoldi*/) {
        CHECK_CUDA(cudaSetDevice(device_id));
        (void)global_cols;
        // Allocate pinned buffer for h_ij:
        // layout: [ h_first (m_alloc_max+1) | h_corr (m_alloc_max+1) ]
        // reuse m_alloc_max as the stride basis
        int stride_basis = std::max(1, allocated_m_max);
        h_pinned_stride = static_cast<size_t>(stride_basis + 1);
        size_t needed = 2 * h_pinned_stride;
        if (h_pinned_capacity < needed) {
            if (h_pinned_h) {
                CHECK_CUDA(cudaFreeHost(h_pinned_h));
            }
            CHECK_CUDA(cudaMallocHost(&h_pinned_h, sizeof(double) * needed));
            h_pinned_capacity = needed;
        }

        // Build map: global column -> ghost position
        std::unordered_map<int, int> global_to_ghost;
        for (size_t idx = 0; idx < ghost_map.incoming_global_col_indices.size(); ++idx) {
            global_to_ghost[ghost_map.incoming_global_col_indices[idx]] = static_cast<int>(idx);
        }

        // Split CSR into on- and off-diagonal parts with remapped columns
        std::vector<int> row_ptr_on(part.rows + 1, 0), row_ptr_off(part.rows + 1, 0);
        std::vector<int> col_idx_on;
        std::vector<int> col_idx_off;
        std::vector<double> values_on;
        std::vector<double> values_off;
        col_idx_on.reserve(part.nnz);
        col_idx_off.reserve(part.nnz);
        values_on.reserve(part.nnz);
        values_off.reserve(part.nnz);

        int local_col_start = global_row_offset;
        int local_col_end = global_row_offset + part.rows;

        for (int r = 0; r < part.rows; ++r) {
            int row_begin = part.row_ptr[r];
            int row_end = part.row_ptr[r + 1];
            for (int idx = row_begin; idx < row_end; ++idx) {
                int col = part.col_idx[idx];
                double val = part.values[idx];
                if (col >= local_col_start && col < local_col_end) {
                    row_ptr_on[r + 1]++;
                    col_idx_on.push_back(col - local_col_start); // local column index
                    values_on.push_back(val);
                } else {
                    auto it = global_to_ghost.find(col);
                    if (it == global_to_ghost.end()) {
                        std::cerr << "ERROR: Missing ghost mapping for global column " << col
                                  << " on GPU " << device_id << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    row_ptr_off[r + 1]++;
                    col_idx_off.push_back(it->second); // remap to ghost position
                    values_off.push_back(val);
                }
            }
        }

        for (int r = 0; r < part.rows; ++r) {
            row_ptr_on[r + 1] += row_ptr_on[r];
            row_ptr_off[r + 1] += row_ptr_off[r];
        }

        int nnz_on = static_cast<int>(col_idx_on.size());
        int nnz_off = static_cast<int>(col_idx_off.size());

        // Allocate on-diag CSR
        CHECK_CUDA(cudaMalloc(&d_row_ptr_on, sizeof(int) * (part.rows + 1)));
        CHECK_CUDA(cudaMemcpyAsync(d_row_ptr_on, row_ptr_on.data(), sizeof(int) * (part.rows + 1), cudaMemcpyHostToDevice, stream_compute));
        if (nnz_on > 0) {
            CHECK_CUDA(cudaMalloc(&d_col_idx_on, sizeof(int) * nnz_on));
            CHECK_CUDA(cudaMalloc(&d_values_on, sizeof(double) * nnz_on));
            CHECK_CUDA(cudaMemcpyAsync(d_col_idx_on, col_idx_on.data(), sizeof(int) * nnz_on, cudaMemcpyHostToDevice, stream_compute));
            CHECK_CUDA(cudaMemcpyAsync(d_values_on, values_on.data(), sizeof(double) * nnz_on, cudaMemcpyHostToDevice, stream_compute));
            CHECK_CUSPARSE(cusparseCreateCsr(&matA_on_descr, part.rows, part.rows, nnz_on,
                                             d_row_ptr_on, d_col_idx_on, d_values_on,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        } else {
            matA_on_descr = nullptr;
        }

        // Allocate off-diag CSR
        CHECK_CUDA(cudaMalloc(&d_row_ptr_off, sizeof(int) * (part.rows + 1)));
        CHECK_CUDA(cudaMemcpyAsync(d_row_ptr_off, row_ptr_off.data(), sizeof(int) * (part.rows + 1), cudaMemcpyHostToDevice, stream_compute));
        if (nnz_off > 0) {
            CHECK_CUDA(cudaMalloc(&d_col_idx_off, sizeof(int) * nnz_off));
            CHECK_CUDA(cudaMalloc(&d_values_off, sizeof(double) * nnz_off));
            CHECK_CUDA(cudaMemcpyAsync(d_col_idx_off, col_idx_off.data(), sizeof(int) * nnz_off, cudaMemcpyHostToDevice, stream_compute));
            CHECK_CUDA(cudaMemcpyAsync(d_values_off, values_off.data(), sizeof(double) * nnz_off, cudaMemcpyHostToDevice, stream_compute));
            CHECK_CUSPARSE(cusparseCreateCsr(&matA_off_descr, part.rows, static_cast<int>(ghost_map.total_recv_size), nnz_off,
                                             d_row_ptr_off, d_col_idx_off, d_values_off,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        } else {
            matA_off_descr = nullptr;
        }

        // Ghost buffers allocated once
        if (ghost_map.total_recv_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_ghost_recv_buffer[0], sizeof(double) * ghost_map.total_recv_size));
            CHECK_CUDA(cudaMalloc(&d_ghost_recv_buffer[1], sizeof(double) * ghost_map.total_recv_size));
        }

        if (ghost_map.total_send_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_ghost_send_buffer[0], sizeof(double) * ghost_map.total_send_size));
            CHECK_CUDA(cudaMalloc(&d_ghost_send_buffer[1], sizeof(double) * ghost_map.total_send_size));
        }

        // Dense vector descriptors for BLAS/SpMV operations
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_w_descr, local_rows, d_w, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y_descr, local_rows, d_y, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_q_local_descr, local_rows, d_q, CUDA_R_64F));
        if (ghost_map.total_recv_size > 0) {
            CHECK_CUSPARSE(cusparseCreateDnVec(&vec_q_ghost_descr[0], ghost_map.total_recv_size, d_ghost_recv_buffer[0], CUDA_R_64F));
            CHECK_CUSPARSE(cusparseCreateDnVec(&vec_q_ghost_descr[1], ghost_map.total_recv_size, d_ghost_recv_buffer[1], CUDA_R_64F));
        } else {
            vec_q_ghost_descr[0] = nullptr;
            vec_q_ghost_descr[1] = nullptr;
        }

        // Query and allocate buffer size for SpMV operations (use max of on/off)
        spmv_buffer_size = 0;
        const double alpha = 1.0, beta = 0.0;
        size_t buf_on = 0, buf_off = 0;
        if (matA_on_descr) {
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha, matA_on_descr, vec_q_local_descr, &beta, vec_w_descr,
                                                   CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_on));
            spmv_buffer_size = std::max(spmv_buffer_size, buf_on);
        }
        if (matA_off_descr && vec_q_ghost_descr[0]) {
            CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha, matA_off_descr, vec_q_ghost_descr[0], &beta, vec_w_descr,
                                                   CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_off));
            spmv_buffer_size = std::max(spmv_buffer_size, buf_off);
        }
        if (spmv_buffer_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_spmv_buffer, spmv_buffer_size));
            std::cout << "  GPU " << device_id << " SpMV buffer size: " << spmv_buffer_size << " bytes" << std::endl;
        } else {
            d_spmv_buffer = nullptr;
        }

        // Persisted ghost index lists on device
        if (!ghost_map.outgoing_global_col_indices.empty()) {
            CHECK_CUDA(cudaMalloc(&d_outgoing_global_col_indices, sizeof(int) * ghost_map.outgoing_global_col_indices.size()));
            CHECK_CUDA(cudaMemcpyAsync(d_outgoing_global_col_indices, ghost_map.outgoing_global_col_indices.data(),
                                       sizeof(int) * ghost_map.outgoing_global_col_indices.size(), cudaMemcpyHostToDevice, stream_compute));
        }
        if (!ghost_map.incoming_global_col_indices.empty()) {
            CHECK_CUDA(cudaMalloc(&d_incoming_global_col_indices, sizeof(int) * ghost_map.incoming_global_col_indices.size()));
            CHECK_CUDA(cudaMemcpyAsync(d_incoming_global_col_indices, ghost_map.incoming_global_col_indices.data(),
                                       sizeof(int) * ghost_map.incoming_global_col_indices.size(), cudaMemcpyHostToDevice, stream_compute));
        }
    }

    // Destroy handles, descriptors, and free device memory
    void destroy() {
        CHECK_CUDA(cudaSetDevice(device_id));

        if (cublas_handle) CHECK_CUBLAS(cublasDestroy(cublas_handle));
        if (cusparse_handle) CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));

        if (matA_descr) CHECK_CUSPARSE(cusparseDestroySpMat(matA_descr));
        if (matA_on_descr) CHECK_CUSPARSE(cusparseDestroySpMat(matA_on_descr));
        if (matA_off_descr) CHECK_CUSPARSE(cusparseDestroySpMat(matA_off_descr));
        if (vec_v_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_v_descr));
        if (vec_w_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_w_descr));
        if (vec_y_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y_descr));
        if (vec_q_local_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_q_local_descr));
        if (vec_q_ghost_descr[0]) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_q_ghost_descr[0]));
        if (vec_q_ghost_descr[1]) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_q_ghost_descr[1]));

        if (d_row_ptr) CHECK_CUDA(cudaFree(d_row_ptr));
        if (d_col_idx) CHECK_CUDA(cudaFree(d_col_idx));
        if (d_values) CHECK_CUDA(cudaFree(d_values));

        if (d_V_m) CHECK_CUDA(cudaFree(d_V_m));
        if (d_q) CHECK_CUDA(cudaFree(d_q));
        if (d_w) CHECK_CUDA(cudaFree(d_w));
        if (d_y) CHECK_CUDA(cudaFree(d_y));

        if (d_row_ptr_on) CHECK_CUDA(cudaFree(d_row_ptr_on));
        if (d_col_idx_on) CHECK_CUDA(cudaFree(d_col_idx_on));
        if (d_values_on) CHECK_CUDA(cudaFree(d_values_on));
        if (d_row_ptr_off) CHECK_CUDA(cudaFree(d_row_ptr_off));
        if (d_col_idx_off) CHECK_CUDA(cudaFree(d_col_idx_off));
        if (d_values_off) CHECK_CUDA(cudaFree(d_values_off));

        if (d_ghost_recv_buffer[0]) CHECK_CUDA(cudaFree(d_ghost_recv_buffer[0]));
        if (d_ghost_recv_buffer[1]) CHECK_CUDA(cudaFree(d_ghost_recv_buffer[1]));
        if (d_ghost_send_buffer[0]) CHECK_CUDA(cudaFree(d_ghost_send_buffer[0]));
        if (d_ghost_send_buffer[1]) CHECK_CUDA(cudaFree(d_ghost_send_buffer[1]));
        if (d_outgoing_global_col_indices) CHECK_CUDA(cudaFree(d_outgoing_global_col_indices));
        if (d_incoming_global_col_indices) CHECK_CUDA(cudaFree(d_incoming_global_col_indices));

        if (d_scalar_host_ptr) CHECK_CUDA(cudaFreeHost(d_scalar_host_ptr));
        if (d_scalar_device) CHECK_CUDA(cudaFree(d_scalar_device));
        if (d_scalar_device_aux) CHECK_CUDA(cudaFree(d_scalar_device_aux));
        if (d_spmv_buffer) CHECK_CUDA(cudaFree(d_spmv_buffer));

        if (stream_compute) CHECK_CUDA(cudaStreamDestroy(stream_compute));
        if (stream_comm) CHECK_CUDA(cudaStreamDestroy(stream_comm));
        if (stream_reduce) CHECK_CUDA(cudaStreamDestroy(stream_reduce));
        if (ghost_recv_ready[0]) CHECK_CUDA(cudaEventDestroy(ghost_recv_ready[0]));
        if (ghost_recv_ready[1]) CHECK_CUDA(cudaEventDestroy(ghost_recv_ready[1]));
        if (send_ready[0]) CHECK_CUDA(cudaEventDestroy(send_ready[0]));
        if (send_ready[1]) CHECK_CUDA(cudaEventDestroy(send_ready[1]));
        if (send_done[0]) CHECK_CUDA(cudaEventDestroy(send_done[0]));
        if (send_done[1]) CHECK_CUDA(cudaEventDestroy(send_done[1]));

        if (h_pinned_h) CHECK_CUDA(cudaFreeHost(h_pinned_h));
    }
};

// CUDA kernel to gather elements from d_q into a send buffer
__global__ void gather_q_elements_for_send_kernel(const double* d_q_src, const int* d_global_indices_to_send, 
                                                 double* d_send_buffer_dest, int num_elements, 
                                                 int global_row_offset_of_owner) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int global_col_idx = d_global_indices_to_send[idx];
        int local_row_idx = global_col_idx - global_row_offset_of_owner;
        d_send_buffer_dest[idx] = d_q_src[local_row_idx];
    }
}

__global__ void negate_scalar_kernel(const double* src, double* dst) {
    if (threadIdx.x == 0) {
        dst[0] = -src[0];
    }
}

__global__ void square_scalar_kernel(const double* src, double* dst) {
    if (threadIdx.x == 0) {
        dst[0] = src[0] * src[0];
    }
}

__global__ void sqrt_scalar_kernel(double* val) {
    if (threadIdx.x == 0) {
        val[0] = sqrt(val[0]);
    }
}

__global__ void reciprocal_scalar_kernel(double* val) {
    if (threadIdx.x == 0) {
        val[0] = 1.0 / val[0];
    }
}

// Simple CSR SpMV kernels (row-parallel)
__global__ void spmv_csr_on_kernel(const int* row_ptr, const int* col_idx, const double* vals,
                                   const double* x, double* y, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        double acc = 0.0;
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int k = start; k < end; ++k) {
            acc += vals[k] * x[col_idx[k]];
        }
        y[r] = acc;
    }
}

__global__ void spmv_csr_off_kernel(const int* row_ptr, const int* col_idx, const double* vals,
                                    const double* x_ghost, double* y, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        double acc = 0.0;
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int k = start; k < end; ++k) {
            acc += vals[k] * x_ghost[col_idx[k]];
        }
        y[r] += acc; // accumulate into y
    }
}

// NcclContext class for NCCL communication
struct NcclContext {
    std::vector<ncclComm_t> comms;
    int num_gpus;

    NcclContext(int n_gpus) : num_gpus(n_gpus) {}

    void init_all(const std::vector<int>& device_ids) {
        comms.resize(num_gpus);
        CHECK_NCCL(ncclCommInitAll(comms.data(), num_gpus, device_ids.data()));
    }

    void allreduce_sum(double* d_send_recv_buffer, size_t count, ncclComm_t comm, cudaStream_t stream) {
        CHECK_NCCL(ncclAllReduce(d_send_recv_buffer, d_send_recv_buffer, count, ncclDouble, ncclSum, comm, stream));
    }

    void allreduce_sumsq(double* d_send_recv_buffer, size_t count, ncclComm_t comm, cudaStream_t stream) {
        // Assuming d_send_recv_buffer already contains local squares. Same as sum.
        CHECK_NCCL(ncclAllReduce(d_send_recv_buffer, d_send_recv_buffer, count, ncclDouble, ncclSum, comm, stream));
    }

    void destroy() {
        for (int i = 0; i < num_gpus; ++i) {
            if (comms[i]) CHECK_NCCL(ncclCommDestroy(comms[i]));
        }
    }
};

// ArnoldiParams struct
struct ArnoldiParams {
    int m;              // Dimension of the Krylov subspace
    double t;           // Time parameter for exp(tA)v
    double tol;         // Tolerance for residual norm
    int max_restarts;   // Maximum number of Arnoldi restarts
    bool reorthogonalize; // Flag for reorthogonalization

    ArnoldiParams(int m_val = 100, double t_val = 1.0, double tol_val = ARNOLDI_TOL, int max_restarts_val = MAX_RESTARTS, bool reortho = true)
        : m(m_val), t(t_val), tol(tol_val), max_restarts(max_restarts_val), reorthogonalize(reortho) {}
};

enum class ExpmvMode {
    AdaptiveTimeStepping,
    RestartedKrylov // future
};

struct TimeSteppingParams {
    double dt_init{0.0};     // initial dt
    double dt_min{0.0};      // minimal dt
    double dt_max{0.0};      // maximal dt
    double tol_step{0.0};    // local error tolerance
    int max_steps{0};        // safety cap on accepted steps
    int max_rejects{20};     // max rejects per step
    double safety{0.8};      // safety factor for dt update
    double grow{1.5};        // dt growth factor
    double shrink{0.5};      // dt shrink factor
    bool adapt_m{false};     // allow increasing m
    int m_max{0};            // upper bound for m when adapting
    int m_step{0};           // increment for m when adapting
};

// ArnoldiRunner class: High-level orchestrator
struct ArnoldiRunner {
    int num_gpus;
    std::vector<DeviceContext> device_contexts;
    NcclContext nccl_context;
    ArnoldiParams params;
    ExpmvMode mode{ExpmvMode::AdaptiveTimeStepping};
    TimeSteppingParams ts_params;
    bool converged = false;
    int restarts_done = 0;
    double last_residual = 0.0;
    double current_segment_t = 0.0; // effective t used in current Arnoldi pass
    std::vector<CSRHost::GhostMap> ghost_maps;

    // Host-side data for small H_m matrix and wH vector
    Eigen::MatrixXd H_m; // Hessenberg matrix
    Eigen::VectorXd e1;  // First canonical basis vector
    Eigen::VectorXd wH;  // exp(t*H_m)*e1
    double v_norm;       // Norm of initial vector v (before normalization)

    // Global matrix dimensions
    int global_rows;
    int global_cols;
    long long global_nnz;

    // Helper to get global row offset for a given GPU
    int get_global_row_offset(int gpu_id) const {
        int offset = 0;
        for (int i = 0; i < gpu_id; ++i) {
            offset += device_contexts[i].local_rows;
        }
        return offset;
    }

    ArnoldiRunner(int n_gpus, const ArnoldiParams& arnoldi_params, int total_rows, int total_cols, long long total_nnz)
        : num_gpus(n_gpus), nccl_context(n_gpus), params(arnoldi_params), global_rows(total_rows), global_cols(total_cols), global_nnz(total_nnz) {
        device_contexts.reserve(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            device_contexts.emplace_back(i);
        }
        // Default time-stepping params derived from Arnoldi params
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
    }

    void init_all_devices(const std::vector<CSRHost>& partitions, const std::vector<int>& device_ids, const std::vector<CSRHost::GhostMap>& ghost_maps_in) {
        // Initialize NCCL communicator
        nccl_context.init_all(device_ids);
        this->ghost_maps = ghost_maps_in; // Store ghost maps

        if (static_cast<int>(device_ids.size()) != num_gpus) {
            std::cerr << "ERROR: device_ids size (" << device_ids.size() << ") must match num_gpus (" << num_gpus << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < num_gpus; ++i) {
            device_contexts[i].device_id = device_ids[i];
        }

        // Safety guard against invalid Arnoldi dimension (avoids buffer overruns in V_m)
        if (params.m >= global_rows) {
            std::cerr << "ERROR: Requested Arnoldi m=" << params.m 
                      << " must be less than global_rows=" << global_rows << std::endl;
            exit(EXIT_FAILURE);
        }
        // Invariant: matrix must be square, and row/col ownership matches row partition
        if (global_rows != global_cols) {
            std::cerr << "ERROR: Matrix must be square. rows=" << global_rows << " cols=" << global_cols << std::endl;
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id)); // Set device before initializing context
            device_contexts[i].init_handles();
            device_contexts[i].create_streams();
            int alloc_m = std::max(params.m, ts_params.m_max > 0 ? ts_params.m_max : params.m);
            device_contexts[i].alloc_from(partitions[i], global_cols, alloc_m);
            device_contexts[i].create_descriptors(partitions[i], global_cols, ghost_maps[i], get_global_row_offset(i), alloc_m); // Pass alloc_m
            // Enable peer access between all pairs of GPUs
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
                    
                    // Check if peer access is supported
                    int can_access;
                    cudaError_t status = cudaDeviceCanAccessPeer(&can_access, device_contexts[i].device_id, device_contexts[j].device_id);
                    if (status == cudaSuccess && can_access) {
                        // Try to enable peer access, ignore if already enabled
                        cudaError_t peer_status = cudaDeviceEnablePeerAccess(device_contexts[j].device_id, 0);
                        if (peer_status != cudaSuccess && peer_status != cudaErrorPeerAccessAlreadyEnabled) {
                            std::cerr << "Warning: Could not enable peer access between GPU " 
                                      << device_contexts[i].device_id << " and GPU " << device_contexts[j].device_id 
                                      << ": " << cudaGetErrorString(peer_status) << std::endl;
                        } else if (peer_status == cudaErrorPeerAccessAlreadyEnabled) {
                            // Clear sticky error flag
                            cudaGetLastError();
                        }
                    } else {
                        std::cerr << "Warning: Peer access not supported between GPU " 
                                  << device_contexts[i].device_id << " and GPU " << device_contexts[j].device_id << std::endl;
                    }
                }
            }
        }
    }

    void compute_expmv(const std::vector<double>& v_host, std::vector<double>& y_host) {
        NvtxRange total_range("total_compute_expmv");
        // Full procedure: normalize q1 -> Arnoldi restarts -> small exponentiation -> lift -> restart/finish
        // Details will be filled in subsequent steps.
        // 1. Initialize q1 (norm of v and scale v)
        init_q1(v_host);

        if (mode != ExpmvMode::AdaptiveTimeStepping) {
            std::cerr << "Expmv mode not implemented" << std::endl;
            return;
        }

        // === Adaptive time-stepping parameters ===
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
                NvtxRange r_step("arnoldi_step_attempt");
                // Resize H for current m
                H_m.setZero(m_current + 1, m_current);
                for (int i = 0; i < num_gpus; ++i) {
                    if (m_current > device_contexts[i].allocated_m_max) {
                        std::cerr << "ERROR: m_current exceeds allocated_m_max on GPU "
                                  << i << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
                int k_used = 0;
                bool breakdown = false;

                // Build Arnoldi basis of length m_current
                for (int j = 0; j < m_current; ++j) {
                    // A) Prepare q_j (ghost exchange and on-diag SpMV)
                    spmv_on_off(j);
                    // B) Orthogonalization (Modified Gram-Schmidt)
                    orthogonalize_mgs(j);
                    // C) Normalize new vector and add to basis
                    bool ok = normalize_new_vector(j);
                    k_used = j + 1;
                    // D) Store H_m column
                    store_H_column(j);
                    if (!ok) { breakdown = true; break; }
                }

                // E) Compute small exponentiation on CPU
                small_expm_and_lift(dt, k_used);
                double err = residual_estimate(dt, k_used);
                if (breakdown) {
                    err = 0.0; // treat breakdown as convergence within subspace
                }

                std::cout << "[STEP] t=" << t_done << " dt=" << dt
                          << " err=" << err << " m_used=" << k_used
                          << " accepts=" << accepted_steps
                          << " rejects=" << rejects << std::endl;

                if (err <= ts_params.tol_step || breakdown) {
                    // ACCEPT
                    t_done += dt;
                    accepted_steps++;
                    last_residual = err;

                    // If finished total time, keep d_y as final result and exit outer loop
                    if (t_done >= total_t) {
                        converged = true;
                        break;
                    }

                    // Prepare next step start vector
                    restart_from_y();

                    // Adapt dt up
                    if (err < 0.1 * ts_params.tol_step) {
                        double dt_new = dt * ts_params.grow * ts_params.safety;
                        dt = std::min(dt_new, ts_params.dt_max);
                    }
                    break; // proceed to next accepted step
                } else {
                    // REJECT
                    rejects++;
                    if (rejects >= ts_params.max_rejects) {
                        std::cerr << "[STEP] rejected too many times, aborting" << std::endl;
                        converged = false;
                        last_residual = err;
                        t_done = total_t; // force exit outer
                        break;
                    }
                    if (ts_params.adapt_m && (m_current + ts_params.m_step) <= ts_params.m_max && ts_params.m_step > 0) {
                        m_current += ts_params.m_step;
                    } else {
                        double dt_new = std::max(dt * ts_params.shrink * ts_params.safety, ts_params.dt_min);
                        dt = dt_new;
                        m_current = params.m;
                    }
                    // retry same time with new dt/m
                    continue;
                }
            }
        }

        restarts_done = accepted_steps; // in this mode: number of accepted time steps

        // Copy final y (in d_y) to host
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[i].stream_compute)); // Ensure all compute is done
            int global_offset = get_global_row_offset(i);
            CHECK_CUDA(cudaMemcpyAsync(y_host.data() + global_offset,
                                       device_contexts[i].d_y, 
                                       sizeof(double) * device_contexts[i].local_rows, 
                                       cudaMemcpyDeviceToHost, 
                                       device_contexts[i].stream_compute));
        }
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[i].stream_compute));
        }

        // FLOPs (approx, assumes fixed m per step, k_used=m)
        const double N = static_cast<double>(global_rows);
        const double nnz = static_cast<double>(global_nnz);
        const double m_val = static_cast<double>(params.m);
        const double S = static_cast<double>(accepted_steps);

        const double flops_gpu_segment =
              2.0 * nnz * m_val                 // SpMV
            + 4.0 * N * m_val * (m_val + 1.0)   // MGS + reorthogonalization
            + 5.0 * N * m_val                   // normalize + lift
            + 2.0 * N;                          // service vector ops

        const double flops_gpu_total =
              3.0 * N * S                        // init_q1 + restart_from_y
            + S * flops_gpu_segment;

        const double flops_cpu_segment =
              2.0 * C_EXP_FLOP_FACTOR * m_val * m_val * m_val   // exp(t*H) ~ O(m^3)
            + 2.0 * m_val * m_val;                              // small gemv + misc

        const double flops_cpu_total = S * flops_cpu_segment;

        std::cout << "[FLOPs] N=" << N << " nnz=" << nnz << " m=" << m_val << " steps=" << S << std::endl;
        std::cout << "[FLOPs] GPU segment: " << flops_gpu_segment
                  << " | GPU total: " << flops_gpu_total << std::endl;
        std::cout << "[FLOPs] CPU segment: " << flops_cpu_segment
                  << " | CPU total: " << flops_cpu_total
                  << " (C_exp=" << C_EXP_FLOP_FACTOR << ")" << std::endl;
    }

    // Internal methods (details to be implemented)
    void init_q1(const std::vector<double>& v_host) {
        NvtxRange r_init("init_q1");
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            int global_offset = get_global_row_offset(i);

            // Copy local part of v_host to d_q on device
            CHECK_CUDA(cudaMemcpyAsync(dc.d_q, 
                                       v_host.data() + global_offset,
                                       sizeof(double) * dc.local_rows,
                                       cudaMemcpyHostToDevice,
                                       dc.stream_compute));
            CHECK_CUBLAS(cublasDnrm2(dc.cublas_handle, dc.local_rows, dc.d_q, 1, dc.d_scalar_device));
            square_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device, dc.d_scalar_device_aux);
        }

        // AllReduce sum of local_v_norms_squared to get global_norm_sq
        CHECK_NCCL(ncclGroupStart());
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            nccl_context.allreduce_sum(dc.d_scalar_device_aux, 1, nccl_context.comms[i], dc.stream_compute);
        }
        CHECK_NCCL(ncclGroupEnd());
        
        // Convert to global norm on each GPU
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            sqrt_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device_aux);
        }

        double global_norm = 0.0;
        CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
        CHECK_CUDA(cudaMemcpyAsync(&global_norm, device_contexts[0].d_scalar_device_aux, sizeof(double),
                                   cudaMemcpyDeviceToHost, device_contexts[0].stream_compute));
        CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_compute));
        
        // Store the norm of the original vector v (before normalization)
        v_norm = global_norm;

        // Scale d_q on each device
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            reciprocal_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device_aux);
            CHECK_CUBLAS(cublasDscal(dc.cublas_handle, dc.local_rows, dc.d_scalar_device_aux, dc.d_q, 1));
            
            // Copy q1 to V_m[0] (first column of Arnoldi basis)
            CHECK_CUDA(cudaMemcpyAsync(dc.d_V_m, dc.d_q, sizeof(double) * dc.local_rows, 
                                       cudaMemcpyDeviceToDevice, dc.stream_compute));
        }
        
        // Synchronize to ensure q1 is copied to V_m[0]
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[i].stream_compute));
        }
    }

    void ghost_exchange_qj(int j) {
        int buf_idx = j % 2;

        // (A) Owners: wait for previous use of this send buffer, then gather q_j
        for (int owner = 0; owner < num_gpus; ++owner) {
            CHECK_CUDA(cudaSetDevice(device_contexts[owner].device_id));
            DeviceContext& dc_owner = device_contexts[owner];
            CSRHost::GhostMap& gm_owner = ghost_maps[owner];

            if (gm_owner.outgoing_global_col_indices.empty()) {
                continue;
            }

            // Ensure prior consumers finished reading this buffer before overwrite
            CHECK_CUDA(cudaStreamWaitEvent(dc_owner.stream_comm, dc_owner.send_done[buf_idx], 0));

            double* d_qj_local = dc_owner.d_V_m + j * dc_owner.local_rows;
            int n_send = static_cast<int>(gm_owner.outgoing_global_col_indices.size());
            int blocks = (n_send + 255) / 256;
            gather_q_elements_for_send_kernel<<<blocks, 256, 0, dc_owner.stream_comm>>>(
                d_qj_local,
                dc_owner.d_outgoing_global_col_indices,
                dc_owner.d_ghost_send_buffer[buf_idx],
                n_send,
                get_global_row_offset(owner));
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(dc_owner.send_ready[buf_idx], dc_owner.stream_comm));
        }

        // (B) Owners: enqueue outgoing copies on owner stream, then mark send_done
        for (int owner = 0; owner < num_gpus; ++owner) {
            CHECK_CUDA(cudaSetDevice(device_contexts[owner].device_id));
            DeviceContext& dc_owner = device_contexts[owner];
            CSRHost::GhostMap& gm_owner = ghost_maps[owner];

            if (!gm_owner.outgoing_peer_info.empty()) {
                CHECK_CUDA(cudaStreamWaitEvent(dc_owner.stream_comm, dc_owner.send_ready[buf_idx], 0));

                for (auto const& [consumer, out_info] : gm_owner.outgoing_peer_info) {
                    size_t send_offset = out_info.first;
                    size_t count = out_info.second;
                    if (count == 0) continue;

                    CSRHost::GhostMap& gm_consumer = ghost_maps[consumer];
                    auto in_it = gm_consumer.incoming_peer_info.find(owner);
                    if (in_it == gm_consumer.incoming_peer_info.end()) {
                        fprintf(stderr, "Ghost plan mismatch: owner %d missing incoming_peer_info on consumer %d\n", owner, consumer);
                        exit(EXIT_FAILURE);
                    }
                    size_t recv_offset = in_it->second.first;
                    size_t recv_count  = in_it->second.second;
                    if (recv_count != count) {
                        fprintf(stderr, "Ghost plan mismatch: owner %d -> consumer %d count send=%zu recv=%zu\n",
                                owner, consumer, count, recv_count);
                        exit(EXIT_FAILURE);
                    }

                    DeviceContext& dc_consumer = device_contexts[consumer];

                    CHECK_CUDA(cudaMemcpyPeerAsync(
                        dc_consumer.d_ghost_recv_buffer[buf_idx] + recv_offset, dc_consumer.device_id,
                        dc_owner.d_ghost_send_buffer[buf_idx] + send_offset,  dc_owner.device_id,
                        sizeof(double) * count,
                        dc_owner.stream_comm));
                }
            }
            // Signal that this send buffer is free to reuse after all enqueued copies
            CHECK_CUDA(cudaEventRecord(dc_owner.send_done[buf_idx], dc_owner.stream_comm));
        }

        // (C) Consumers: wait for all relevant owners' send_done, then mark recv_ready
        for (int consumer = 0; consumer < num_gpus; ++consumer) {
            CHECK_CUDA(cudaSetDevice(device_contexts[consumer].device_id));
            DeviceContext& dc_consumer = device_contexts[consumer];
            CSRHost::GhostMap& gm_consumer = ghost_maps[consumer];

            for (auto const& [owner, in_info] : gm_consumer.incoming_peer_info) {
                size_t count = in_info.second;
                if (count == 0) continue;
                DeviceContext& dc_owner = device_contexts[owner];
                CHECK_CUDA(cudaStreamWaitEvent(dc_consumer.stream_comm, dc_owner.send_done[buf_idx], 0));
            }

            CHECK_CUDA(cudaEventRecord(dc_consumer.ghost_recv_ready[buf_idx], dc_consumer.stream_comm));
        }
    }

    void spmv_on_off(int j) {
        // Phase 0: start ghost exchange for q_j (double-buffered)
        ghost_exchange_qj(j);

        // Phase 1: Launch on-diag SpMV on all GPUs in parallel
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];

            double* d_qj_local = dc.d_V_m + j * dc.local_rows;

            // Zero out d_w before accumulation
            CHECK_CUDA(cudaMemsetAsync(dc.d_w, 0, sizeof(double) * dc.local_rows, dc.stream_compute));

            // Custom CSR on-diag SpMV
            int threads = 256;
            int blocks = (dc.local_rows + threads - 1) / threads;
            spmv_csr_on_kernel<<<blocks, threads, 0, dc.stream_compute>>>(
                dc.d_row_ptr_on, dc.d_col_idx_on, dc.d_values_on,
                d_qj_local, dc.d_w, dc.local_rows);
            CHECK_CUDA(cudaGetLastError());
        }

        // Phase 2: Off-diag SpMV using ghost buffer, accumulate into w
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            CSRHost::GhostMap& gm = ghost_maps[i];
            if (!dc.d_row_ptr_off || gm.total_recv_size == 0) continue;

            int current_recv_buffer_idx = j % 2;
            CHECK_CUDA(cudaStreamWaitEvent(dc.stream_compute, dc.ghost_recv_ready[current_recv_buffer_idx], 0));

            // Custom CSR off-diag SpMV accumulate into w
            int threads = 256;
            int blocks = (dc.local_rows + threads - 1) / threads;
            spmv_csr_off_kernel<<<blocks, threads, 0, dc.stream_compute>>>(
                dc.d_row_ptr_off, dc.d_col_idx_off, dc.d_values_off,
                dc.d_ghost_recv_buffer[current_recv_buffer_idx],
                dc.d_w, dc.local_rows);
            CHECK_CUDA(cudaGetLastError());
        }
    }

    void orthogonalize_mgs(int j) {
        NvtxRange r_ortho("orthogonalize_mgs");

        // Modified Gram-Schmidt orthogonalization: w = w - sum_i (h_{i,j} * q_i)
        int m_lim = static_cast<int>(H_m.cols());
        // Use pinned host buffer on GPU0: first half = h_first, second half = h_corr (stride = h_pinned_stride)
        double* h_first = device_contexts[0].h_pinned_h;
        double* h_corr  = device_contexts[0].h_pinned_h ? device_contexts[0].h_pinned_h + device_contexts[0].h_pinned_stride : nullptr;
        for (int i = 0; i <= j; ++i) {
            if (h_first) { h_first[i] = 0.0; }
            if (h_corr)  { h_corr[i]  = 0.0; }
        }
        for (int i = 0; i <= j; ++i) {
            for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                DeviceContext& dc = device_contexts[gpu_id];
                
                if (i > j) {
                    std::cerr << "ERROR: MGS index out of order: i=" << i << " > j=" << j << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (i >= m_lim) {
                    std::cerr << "ERROR: Accessing V_m[" << i << "] but m_lim=" << m_lim << std::endl;
                    exit(EXIT_FAILURE);
                }
                if (i >= dc.allocated_m_max + 1) {
                    std::cerr << "ERROR: V_m index exceeds allocation: i=" << i << " allocated_m_max=" << dc.allocated_m_max << std::endl;
                    exit(EXIT_FAILURE);
                }

                double* d_q_i = dc.d_V_m + i * dc.local_rows;
                CHECK_CUBLAS(cublasDdot(dc.cublas_handle, dc.local_rows, dc.d_w, 1, d_q_i, 1, dc.d_scalar_device));
            }

            // AllReduce to get global dot product h_{i,j}
            CHECK_NCCL(ncclGroupStart());
            for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                DeviceContext& dc = device_contexts[gpu_id];
                nccl_context.allreduce_sum(dc.d_scalar_device, 1, nccl_context.comms[gpu_id], dc.stream_compute);
            }
            CHECK_NCCL(ncclGroupEnd());

            // Prepare alpha = -h_{i,j} on device and update w
            for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                DeviceContext& dc = device_contexts[gpu_id];
                double* d_q_i = dc.d_V_m + i * dc.local_rows;
                negate_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device, dc.d_scalar_device_aux);
                CHECK_CUBLAS(cublasDaxpy(dc.cublas_handle, dc.local_rows, dc.d_scalar_device_aux, d_q_i, 1, dc.d_w, 1));
            }
            // Async copy h_{i,j} from first pass to pinned host buffer
            if (h_first) {
                CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
                CHECK_CUDA(cudaMemcpyAsync(&h_first[i], device_contexts[0].d_scalar_device, sizeof(double),
                                           cudaMemcpyDeviceToHost, device_contexts[0].stream_compute));
            }
        }

        // Optional reorthogonalization if enabled
        if (params.reorthogonalize) {
            // Check orthogonality loss and perform second pass if needed
            // For simplicity, we'll always do a second pass if reorthogonalization is enabled
            for (int i = 0; i <= j; ++i) {
                for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                    DeviceContext& dc = device_contexts[gpu_id];
                    
                    double* d_q_i = dc.d_V_m + i * dc.local_rows;
                    CHECK_CUBLAS(cublasDdot(dc.cublas_handle, dc.local_rows, dc.d_w, 1, d_q_i, 1, dc.d_scalar_device));
                }

                // AllReduce for reorthogonalization correction
                CHECK_NCCL(ncclGroupStart());
                for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                    DeviceContext& dc = device_contexts[gpu_id];
                    nccl_context.allreduce_sum(dc.d_scalar_device, 1, nccl_context.comms[gpu_id], dc.stream_compute);
                }
                CHECK_NCCL(ncclGroupEnd());
                
                for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                    DeviceContext& dc = device_contexts[gpu_id];
                    
                    double* d_q_i = dc.d_V_m + i * dc.local_rows;
                    negate_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device, dc.d_scalar_device_aux);
                    CHECK_CUBLAS(cublasDaxpy(dc.cublas_handle, dc.local_rows, dc.d_scalar_device_aux, d_q_i, 1, dc.d_w, 1));
                }

                // Async copy correction to pinned buffer
                if (h_corr) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
                    CHECK_CUDA(cudaMemcpyAsync(&h_corr[i], device_contexts[0].d_scalar_device, sizeof(double),
                                               cudaMemcpyDeviceToHost, device_contexts[0].stream_compute));
                }
            }
        }

        // Synchronize once and update H_m from buffered h_col
        CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
        CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_compute));
        for (int i = 0; i <= j && i < m_lim; ++i) {
            double first = (h_first) ? h_first[i] : 0.0;
            double corr  = (h_corr)  ? h_corr[i]  : 0.0;
            H_m(i, j) = first + corr;
        }
    }

    bool normalize_new_vector(int j) {
        // Phase 1: Compute local norms (device scalars)
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            CHECK_CUBLAS(cublasDnrm2(dc.cublas_handle, dc.local_rows, dc.d_w, 1, dc.d_scalar_device));
            square_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device, dc.d_scalar_device_aux);
        }

        // Phase 2: AllReduce to get global norm squared
        CHECK_NCCL(ncclGroupStart());
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            nccl_context.allreduce_sumsq(dc.d_scalar_device_aux, 1, nccl_context.comms[gpu_id], dc.stream_compute);
        }
        CHECK_NCCL(ncclGroupEnd());
        
        // Convert to global norm on each GPU
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            sqrt_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device_aux);
        }

        // Fetch global norm on host (GPU 0) for breakdown check / H_m storage
        double global_norm = 0.0;
        CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
        CHECK_CUDA(cudaMemcpyAsync(&global_norm, device_contexts[0].d_scalar_device_aux, sizeof(double),
                                   cudaMemcpyDeviceToHost, device_contexts[0].stream_compute));
        CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_compute));

        // Check for zero or invalid norm (Arnoldi breakdown)
        if (global_norm < 1e-14 || std::isnan(global_norm) || std::isinf(global_norm)) {
        if (j + 1 < H_m.rows() && j < H_m.cols()) {
            H_m(j + 1, j) = 0.0;
        }
            return false;
        }

        // Phase 3: Normalize w and store as q_{j+1} (next Arnoldi vector)
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            reciprocal_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device_aux);
            CHECK_CUBLAS(cublasDscal(dc.cublas_handle, dc.local_rows, dc.d_scalar_device_aux, dc.d_w, 1));

            double* d_q_next = dc.d_V_m + (j + 1) * dc.local_rows;
            CHECK_CUDA(cudaMemcpyAsync(d_q_next, dc.d_w, sizeof(double) * dc.local_rows, 
                                       cudaMemcpyDeviceToDevice, dc.stream_compute));
            CHECK_CUDA(cudaMemcpyAsync(dc.d_q, dc.d_w, sizeof(double) * dc.local_rows, 
                                       cudaMemcpyDeviceToDevice, dc.stream_compute));
        }
        
        // Synchronize all GPUs to ensure V_m[j+1] is ready for next iteration
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[gpu_id].stream_compute));
        }

        if (j + 1 < H_m.rows() && j < H_m.cols()) {
            H_m(j + 1, j) = global_norm;
        }
        return true;
    }

    void store_H_column(int j) {
        
        // The H_m matrix is already being populated during orthogonalize_mgs and normalize_new_vector
        // This method is mainly for logging/debugging purposes
        // In a more sophisticated implementation, you might want to copy H_m to device memory
        // or perform additional validation here
        
        std::cout << "    H_m column " << j << " stored. Current H_m size: " << H_m.rows() << "x" << H_m.cols() << std::endl;
    }

    void small_expm_and_lift(double segment_t, int k_used) {
        NvtxRange r_small("small_expm_and_lift");

        // Guard: if no steps were performed (should not happen), skip
        if (k_used <= 0) return;

        // Phase 1: Compute small matrix exponential on CPU using Eigen
        // Create e1 vector (first canonical basis vector)
        e1 = Eigen::VectorXd::Zero(k_used + 1);
        e1(0) = 1.0;

        // Extract the k_used x k_used upper block of H_m
        Eigen::MatrixXd H_m_square = H_m.block(0, 0, k_used, k_used);
        
        // Compute exp(t * H_m_square) using stable matrix exponential
        Eigen::MatrixXd exp_tH_square = matrix_exp_stable(segment_t * H_m_square);
        
        // Extend to (m+1) x (m+1) for consistency
        // exp(t*H_m) is approximated by a block structure
        Eigen::MatrixXd exp_tH = Eigen::MatrixXd::Zero(k_used + 1, k_used + 1);
        exp_tH.block(0, 0, k_used, k_used) = exp_tH_square;
        
        // Compute wH = exp(t * H_m) * e1
        // Since e1 has e1(0)=1 and rest=0, we only need the first column of exp_tH
        wH = exp_tH * e1;

        // Phase 2: Lift result back to original space on GPU
        // y_local = ||v|| * V_m * wH (where ||v|| was stored in init_q1)
        
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            
            // Zero out d_y
            CHECK_CUDA(cudaMemsetAsync(dc.d_y, 0, sizeof(double) * dc.local_rows, dc.stream_compute));
            
            // Compute y_local = V_m * wH using AXPY operations
            // y = sum_i (wH[i] * V_m[:, i])
            for (int i = 0; i <= k_used; ++i) {
                if (std::abs(wH(i)) > 1e-15) { // Skip near-zero coefficients
                    const double alpha = v_norm * wH(i);
                    double* d_V_i = dc.d_V_m + i * dc.local_rows;
                    CHECK_CUDA(cudaMemcpyAsync(dc.d_scalar_device_aux, &alpha, sizeof(double),
                                               cudaMemcpyHostToDevice, dc.stream_compute));
                    CHECK_CUBLAS(cublasDaxpy(dc.cublas_handle, dc.local_rows, dc.d_scalar_device_aux, d_V_i, 1, dc.d_y, 1));
                }
            }
        }
    }

    double residual_estimate(double segment_t, int k_used) {
        if (k_used <= 0) return 0.0;
        // Residual-based convergence check: ||r|| ≈ |h_{k,k-1}| * ||e_k^T * exp(t*H_k) * e1||
        Eigen::MatrixXd H_m_square = H_m.block(0, 0, k_used, k_used);
        Eigen::MatrixXd exp_tH_square = matrix_exp_stable(segment_t * H_m_square);

        Eigen::VectorXd e1_small = Eigen::VectorXd::Zero(k_used);
        e1_small(0) = 1.0;

        Eigen::VectorXd exp_tH_e1 = exp_tH_square * e1_small;
        double e_k_exp_tH_e1 = exp_tH_e1(k_used - 1);

        double h_k_plus_1_k = H_m(k_used, k_used - 1);
        double residual_norm = std::abs(h_k_plus_1_k * e_k_exp_tH_e1) * v_norm;

        return residual_norm;
    }

    // Use the current y as the next starting vector (restart) and rebuild q1/V_m[0]
    void restart_from_y() {
        // 1) Local norms of y on each GPU (device scalars)
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            CHECK_CUBLAS(cublasDnrm2(dc.cublas_handle, dc.local_rows, dc.d_y, 1, dc.d_scalar_device));
            square_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device, dc.d_scalar_device_aux);
        }

        // 2) AllReduce to obtain global norm of y
        CHECK_NCCL(ncclGroupStart());
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            nccl_context.allreduce_sumsq(dc.d_scalar_device_aux, 1, nccl_context.comms[gpu_id], dc.stream_compute);
        }
        CHECK_NCCL(ncclGroupEnd());

        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            sqrt_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device_aux);
        }

        double global_norm = 0.0;
        CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
        CHECK_CUDA(cudaMemcpyAsync(&global_norm, device_contexts[0].d_scalar_device_aux, sizeof(double),
                                   cudaMemcpyDeviceToHost, device_contexts[0].stream_compute));
        CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_compute));

        v_norm = global_norm;

        // 3) Normalize y to obtain new q1 and store in V_m[:,0]
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];

            CHECK_CUDA(cudaMemcpyAsync(dc.d_q, dc.d_y, sizeof(double) * dc.local_rows,
                                       cudaMemcpyDeviceToDevice, dc.stream_compute));
            reciprocal_scalar_kernel<<<1, 1, 0, dc.stream_compute>>>(dc.d_scalar_device_aux);
            CHECK_CUBLAS(cublasDscal(dc.cublas_handle, dc.local_rows, dc.d_scalar_device_aux, dc.d_q, 1));
            CHECK_CUDA(cudaMemcpyAsync(dc.d_V_m, dc.d_q, sizeof(double) * dc.local_rows,
                                       cudaMemcpyDeviceToDevice, dc.stream_compute));
        }

        // 4) Ensure q1 is ready on all GPUs
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[gpu_id].stream_compute));
        }
    }

    void destroy() {
        for (int i = 0; i < num_gpus; ++i) {
            device_contexts[i].destroy();
        }
        nccl_context.destroy();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "Starting matrix_exp application." << std::endl;

    // Parse command-line arguments:
    // Positional style: ./matrix_exp [matrix_or_size] [m] [max_restarts]
    // Named style:      --m N, --max-restarts N
    bool use_file = false;
    std::string matrix_file;
    int test_size = 1000; // Default size for generated matrix
    int m_param = -1;     // If -1, will be derived from matrix size
    int max_restarts_cli = MAX_RESTARTS; // If <=0, run until convergence
    double t_cli = 1.0;   // Time parameter (default 1.0)

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

        // Positional handling
        if (pos_idx == 0) {
            if (a.find(".") != std::string::npos || a.find("/") != std::string::npos || a.rfind(".mtx") != std::string::npos) {
                use_file = true;
                matrix_file = a;
            } else {
                test_size = std::atoi(a.c_str());
                if (test_size <= 0) {
                    std::cerr << "Error: Matrix size must be a positive integer. Using default size 1000." << std::endl;
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
        // Check available GPUs
        int num_gpus;
        CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
        std::cout << "Found " << num_gpus << " GPU(s)" << std::endl;
        
        if (num_gpus < 1) {
            std::cerr << "No GPUs found!" << std::endl;
            return EXIT_FAILURE;
        }

        // Use available GPUs (limit to 4 for this example)
        int num_gpus_to_use = std::min(num_gpus, 4);

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

        // Prepare matrix
        CSRHost test_matrix;
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
                std::cerr << "Error: Could not detect matrix format for file " << matrix_file
                          << ". Supported: .bincsr, .mtx, .mtx.gz" << std::endl;
                return EXIT_FAILURE;
            }
            std::cout << "Loaded matrix: " << test_matrix.rows << "x" << test_matrix.cols
                      << " with " << test_matrix.nnz << " non-zeros" << std::endl;
        } else {
            std::cout << "Matrix size (generated): " << test_size << std::endl;
            // Create a simple test matrix (diagonal matrix for testing)
            test_matrix.rows = test_size;
            test_matrix.cols = test_size;
            test_matrix.nnz = test_size;
            
            test_matrix.row_ptr.resize(test_matrix.rows + 1);
            test_matrix.col_idx.resize(test_matrix.nnz);
            test_matrix.values.resize(test_matrix.nnz);
            
            for (int i = 0; i < test_matrix.rows; ++i) {
                test_matrix.row_ptr[i] = i;
                test_matrix.col_idx[i] = i;
                // Use different diagonal values to avoid degenerate case
                test_matrix.values[i] = 1.0 + (double)i / test_matrix.rows;
            }
            test_matrix.row_ptr[test_matrix.rows] = test_matrix.nnz;

            std::cout << "Created test matrix: " << test_matrix.rows << "x" << test_matrix.cols 
                      << " with " << test_matrix.nnz << " non-zeros" << std::endl;
        }

        // Prevent zero-row partitions (must precede device_ids allocation)
        num_gpus_to_use = std::min(num_gpus_to_use, test_matrix.rows);

        std::vector<int> device_ids(num_gpus_to_use);
        for (int i = 0; i < num_gpus_to_use; ++i) {
            device_ids[i] = i;
        }

        // Quick sanity: clamp time step if matrix has huge entries to avoid exp overflow
        double max_abs_A = 0.0;
        for (double v : test_matrix.values) {
            max_abs_A = std::max(max_abs_A, std::abs(v));
        }
        if (max_abs_A <= 0.0) {
            max_abs_A = 1.0; // avoid division by zero if matrix is all zeros
        }
        double t_safe = t_cli;
        // Keep t*||A||_max below ~10 to avoid exp overflow on very large entries
        double t_limit = 10.0 / max_abs_A;
        if (t_safe > t_limit) {
            std::cout << "[WARN] Requested t=" << t_safe << " is large for max|A|=" << max_abs_A
                      << ". Clamping t to " << t_limit << " to avoid overflow." << std::endl;
            t_safe = t_limit;
        }

        // Partition matrix across GPUs
        auto partitions = test_matrix.partition_rows(num_gpus_to_use);
        std::cout << "Partitioned matrix across " << num_gpus_to_use << " GPUs" << std::endl;

        // Build ghost maps
        std::vector<CSRHost::GhostMap> ghost_maps(num_gpus_to_use);
        for (int i = 0; i < num_gpus_to_use; ++i) {
            ghost_maps[i] = test_matrix.build_owner_ghost_maps(partitions, i, num_gpus_to_use);
        }
        validate_ghost_maps(ghost_maps, num_gpus_to_use);

        // Set up Arnoldi parameters
        int m_val = m_param;
        if (m_val <= 0) {
            m_val = std::min(30, std::max(5, test_matrix.rows - 1));
        }
        // Clamp to matrix size
        m_val = std::max(1, std::min(m_val, test_matrix.rows - 1));

        int max_restarts_val = max_restarts_cli;
        // max_restarts_val <= 0 means run until convergence
        ArnoldiParams params(m_val, t_safe, 1e-6, max_restarts_val, true);

        std::cout << "Arnoldi params: m=" << params.m
                  << " t=" << params.t
                  << " tol=" << params.tol
                  << " max_restarts=" << params.max_restarts
                  << std::endl;

        // Create ArnoldiRunner
        ArnoldiRunner runner(num_gpus_to_use, params, test_matrix.rows, test_matrix.cols, test_matrix.nnz);
        
        // Initialize all devices
        runner.init_all_devices(partitions, device_ids, ghost_maps);

        // Create test vector v
        std::vector<double> v_host(test_matrix.rows, 0.1); // Initial vector with all elements = 0.1
        std::vector<double> y_host(test_matrix.rows, 0.0); // Result vector

        // Compute exp(tA)v
        runner.compute_expmv(v_host, y_host);

        // Optional: dump full y_host to a file if OUTPUT_Y_FILE is set and non-empty
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

        // Print some results
        std::cout << "Computation completed. Sample results:" << std::endl;
        for (int i = 0; i < std::min(10, (int)y_host.size()); ++i) {
            std::cout << "y[" << i << "] = " << y_host[i] << std::endl;
        }
        std::cout << "Arnoldi summary: "
                  << "m=" << m_val
                  << ", max_restarts=" << max_restarts_val
                  << ", restarts_done=" << runner.restarts_done
                  << ", converged=" << (runner.converged ? "yes" : "no")
                  << ", residual=" << runner.last_residual
                  << std::endl;

        // Clean up
        runner.destroy();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}

