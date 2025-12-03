#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <cstdlib>

// CUDA includes
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <nccl.h>

// Eigen for small matrix exponentiation (using only stable modules)
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Stable matrix exponential implementation using only core Eigen modules
Eigen::MatrixXd matrix_exp_stable(const Eigen::MatrixXd& A) {
    // For small matrices, use eigenvalue decomposition: exp(A) = V * exp(D) * V^(-1)
    // where A = V * D * V^(-1) is the eigenvalue decomposition
    
    Eigen::EigenSolver<Eigen::MatrixXd> solver(A);
    if (solver.info() != Eigen::Success) {
        // Fallback to Taylor series for non-diagonalizable matrices
        Eigen::MatrixXd result = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        Eigen::MatrixXd term = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        
        for (int k = 1; k <= 20; ++k) {  // Taylor series: exp(A) = I + A + A²/2! + A³/3! + ...
            term = term * A / k;
            result += term;
            
            // Check convergence
            if (term.norm() < 1e-15) break;
        }
        return result;
    }
    
    // Use eigenvalue decomposition
    Eigen::MatrixXd V = solver.eigenvectors().real();
    Eigen::VectorXd D = solver.eigenvalues().real();
    
    // Compute exp(D) element-wise
    Eigen::VectorXd exp_D = D.array().exp();
    
    // Reconstruct exp(A) = V * diag(exp(D)) * V^(-1)
    return V * exp_D.asDiagonal() * V.inverse();
}

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
const int MAX_RESTARTS = 10;     // Maximum number of restarts

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
        // Implementation will go here
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string line;
        // Skip comments and header lines
        while (std::getline(file, line)) {
            if (line[0] == '%') {
                continue;
            }
            std::stringstream ss(line);
            std::string token;
            ss >> token; // %%MatrixMarket
            if (token == "%%MatrixMarket") {
                ss >> token; // matrix
                ss >> token; // coordinate
                ss >> token; // real/integer
                ss >> token; // general/symmetric
                continue;
            }
            // Read dimensions and nnz
            ss.clear();
            ss.str(line);
            ss >> rows >> cols >> nnz;
            break;
        }

        row_ptr.assign(rows + 1, 0);
        std::vector<std::tuple<int, int, double>> entries;
        entries.reserve(nnz);

        for (long long i = 0; i < nnz; ++i) {
            int r, c;
            double val;
            file >> r >> c >> val;
            entries.emplace_back(r - 1, c - 1, val); // 0-indexed
        }
        file.close();

        // Sort entries by row and then column
        std::sort(entries.begin(), entries.end());

        // Convert to CSR format
        col_idx.reserve(nnz);
        values.reserve(nnz);

        int current_row = 0;
        for (const auto& entry : entries) {
            int r, c;
            double val;
            std::tie(r, c, val) = entry;

            while (current_row < r) {
                row_ptr[current_row + 1] = col_idx.size();
                current_row++;
            }
            col_idx.push_back(c);
            values.push_back(val);
        }
        while (current_row < rows) {
            row_ptr[current_row + 1] = col_idx.size();
            current_row++;
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

        // Determine global row ranges for each GPU
        std::vector<std::pair<int, int>> gpu_row_ranges(num_gpus);
        int current_global_row = 0;
        for (int i = 0; i < num_gpus; ++i) {
            gpu_row_ranges[i] = {current_global_row, current_global_row + partitions[i].rows - 1};
            current_global_row += partitions[i].rows;
        }

        // ============================ Building RECEIVE maps for my_gpu_id ===============================
        // Identify ghost columns for current GPU (elements it needs to receive)
        std::set<int> unique_incoming_ghost_cols;
        const CSRHost& my_partition = partitions[my_gpu_id];
        
        for (int k = 0; k < my_partition.nnz; ++k) {
            int col = my_partition.col_idx[k];
            // Check if the column owner is not this GPU
            bool is_local_col = (col >= gpu_row_ranges[my_gpu_id].first && col <= gpu_row_ranges[my_gpu_id].second);
            if (!is_local_col) {
                unique_incoming_ghost_cols.insert(col);
            }
        }

        for (int col : unique_incoming_ghost_cols) {
            ghost_map.incoming_global_col_indices.push_back(col);
            // Determine owner of this ghost column
            for (int i = 0; i < num_gpus; ++i) {
                if (col >= gpu_row_ranges[i].first && col <= gpu_row_ranges[i].second) {
                    ghost_map.owner_to_consumer_map[i].push_back(col); // This GPU (my_gpu_id) needs 'col' from GPU 'i'
                    break;
                }
            }
        }
        
        // Sort incoming ghost indices to ensure deterministic order
        std::sort(ghost_map.incoming_global_col_indices.begin(), ghost_map.incoming_global_col_indices.end());

        // Populate incoming_peer_info and recv_offsets/counts
        size_t current_recv_offset = 0;
        for (int p_id = 0; p_id < num_gpus; ++p_id) {
            if (p_id == my_gpu_id) continue;
            // Sort indices for each owner_to_consumer list for potential contiguous blocks (even if unlikely)
            std::sort(ghost_map.owner_to_consumer_map[p_id].begin(), ghost_map.owner_to_consumer_map[p_id].end());
            size_t count = ghost_map.owner_to_consumer_map[p_id].size();
            if (count > 0) {
                ghost_map.incoming_peer_info[p_id] = {current_recv_offset, count};
                ghost_map.recv_offsets[p_id] = current_recv_offset;
                ghost_map.recv_counts[p_id] = count;
                current_recv_offset += count;
            }
        }
        ghost_map.total_recv_size = current_recv_offset;

        // ============================ Building SEND maps for my_gpu_id ==================================
        std::set<int> unique_outgoing_global_cols;
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
                        unique_outgoing_global_cols.insert(col);
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

        for (int col : unique_outgoing_global_cols) {
            ghost_map.outgoing_global_col_indices.push_back(col);
        }
        // Sort outgoing global column indices for deterministic kernel processing
        std::sort(ghost_map.outgoing_global_col_indices.begin(), ghost_map.outgoing_global_col_indices.end());

        // Populate outgoing_peer_info and send_offsets/counts
        size_t current_send_offset = 0;
        for (int p_id = 0; p_id < num_gpus; ++p_id) {
            if (p_id == my_gpu_id) continue;
            size_t count = ghost_map.consumer_to_owner_map[p_id].size();
            if (count > 0) {
                ghost_map.outgoing_peer_info[p_id] = {current_send_offset, count};
                ghost_map.send_offsets[p_id] = current_send_offset;
                ghost_map.send_counts[p_id] = count;
                current_send_offset += count;
            }
        }
        ghost_map.total_send_size = current_send_offset;
        
        return ghost_map;
    }
};

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
    cusparseSpMatDescr_t matA_descr;
    cusparseDnVecDescr_t vec_v_descr, vec_q_descr, vec_w_descr, vec_y_descr;

    // Device pointers for Arnoldi vectors and work vectors
    // V_m will store the Arnoldi basis vectors as columns
    double* d_V_m; // size: local_rows * m
    double* d_q;   // current Arnoldi vector q_j (local part)
    double* d_q_full; // full vector for SpMV (global size)
    double* d_w;   // work vector A*q_j
    double* d_y;   // final result vector

    // Ghost buffers for communication (double buffering)
    double* d_ghost_recv_buffer[2]; // Two buffers for double buffering
    double* d_ghost_send_buffer;    // Buffer for sending ghost data

    int local_rows; // Number of rows owned by this GPU
    int local_nnz;  // Number of non-zero elements in local CSR partition

    // Device-side scalars for AllReduce and temporary computations
    double* d_scalar_host_ptr; // Pinned host memory for host-device transfers
    double* d_scalar_device; // Device memory for cuBLAS operations
    
    // cuSPARSE work buffer for SpMV operations
    void* d_spmv_buffer;
    size_t spmv_buffer_size;

    DeviceContext(int dev_id) : 
        device_id(dev_id), 
        stream_compute(nullptr), stream_comm(nullptr), stream_reduce(nullptr),
        cublas_handle(nullptr), cusparse_handle(nullptr),
        d_row_ptr(nullptr), d_col_idx(nullptr), d_values(nullptr),
        matA_descr(nullptr), vec_v_descr(nullptr), vec_q_descr(nullptr), vec_w_descr(nullptr), vec_y_descr(nullptr),
        d_V_m(nullptr), d_q(nullptr), d_q_full(nullptr), d_w(nullptr), d_y(nullptr),
        d_ghost_send_buffer(nullptr),
        d_scalar_host_ptr(nullptr), d_scalar_device(nullptr),
        d_spmv_buffer(nullptr), spmv_buffer_size(0)
    {
        d_ghost_recv_buffer[0] = nullptr;
        d_ghost_recv_buffer[1] = nullptr;
    }

    // Initialize cuBLAS/cuSPARSE handles
    void init_handles() {
        CHECK_CUDA(cudaSetDevice(device_id));
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
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
    }

    // Allocate device memory and copy CSR partition from host
    void alloc_from(const CSRHost& part, int global_cols) {
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
        CHECK_CUDA(cudaMalloc(&d_q_full, sizeof(double) * global_cols)); // Full vector for SpMV
        CHECK_CUDA(cudaMalloc(&d_w, sizeof(double) * local_rows));
        CHECK_CUDA(cudaMalloc(&d_y, sizeof(double) * local_rows));
        // V_m size will depend on 'm' (Arnoldi subspace dimension), which is part of ArnoldiParams.
        // We'll allocate a generous size here, or reallocate when ArnoldiRunner is initialized.
        // For now, let's allocate for a max_m. Let's assume max_m = 200 from the doc.txt (m - десятки/сотни).
        int max_m = 200;
        CHECK_CUDA(cudaMalloc(&d_V_m, sizeof(double) * local_rows * max_m));

        // Allocate pinned host memory for scalar reduction results
        CHECK_CUDA(cudaMallocHost(&d_scalar_host_ptr, sizeof(double)));
        // Allocate device memory for cuBLAS scalar results
        CHECK_CUDA(cudaMalloc(&d_scalar_device, sizeof(double)));
    }

    // Create cuSPARSE descriptors
    void create_descriptors(int global_cols, const CSRHost::GhostMap& ghost_map, int m_arnoldi) {
        CHECK_CUDA(cudaSetDevice(device_id));
        CHECK_CUSPARSE(cusparseCreateCsr(&matA_descr, local_rows, global_cols, local_nnz, d_row_ptr, d_col_idx, d_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        // Dense vector descriptors for BLAS/SpMV operations
        // vec_q_descr uses d_q_full (global size) for SpMV operations
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_q_descr, global_cols, d_q_full, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_w_descr, local_rows, d_w, CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y_descr, local_rows, d_y, CUDA_R_64F));
        
        // Query and allocate buffer size for SpMV operations
        const double alpha = 1.0, beta = 0.0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matA_descr, vec_q_descr, &beta, vec_w_descr,
                                               CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size));
        CHECK_CUDA(cudaMalloc(&d_spmv_buffer, spmv_buffer_size));
        std::cout << "  GPU " << device_id << " SpMV buffer size: " << spmv_buffer_size << " bytes" << std::endl;

        // For ghost data, we need buffers. total_recv_size and total_send_size are from GhostMap
        if (ghost_map.total_recv_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_ghost_recv_buffer[0], sizeof(double) * ghost_map.total_recv_size));
            CHECK_CUDA(cudaMalloc(&d_ghost_recv_buffer[1], sizeof(double) * ghost_map.total_recv_size));
        }

        if (ghost_map.total_send_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_ghost_send_buffer, sizeof(double) * ghost_map.total_send_size));
        }
    }

    // Destroy handles, descriptors, and free device memory
    void destroy() {
        CHECK_CUDA(cudaSetDevice(device_id));

        if (cublas_handle) CHECK_CUBLAS(cublasDestroy(cublas_handle));
        if (cusparse_handle) CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));

        if (matA_descr) CHECK_CUSPARSE(cusparseDestroySpMat(matA_descr));
        if (vec_v_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_v_descr));
        if (vec_q_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_q_descr));
        if (vec_w_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_w_descr));
        if (vec_y_descr) CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y_descr));

        if (d_row_ptr) CHECK_CUDA(cudaFree(d_row_ptr));
        if (d_col_idx) CHECK_CUDA(cudaFree(d_col_idx));
        if (d_values) CHECK_CUDA(cudaFree(d_values));

        if (d_V_m) CHECK_CUDA(cudaFree(d_V_m));
        if (d_q) CHECK_CUDA(cudaFree(d_q));
        if (d_q_full) CHECK_CUDA(cudaFree(d_q_full));
        if (d_w) CHECK_CUDA(cudaFree(d_w));
        if (d_y) CHECK_CUDA(cudaFree(d_y));

        if (d_ghost_recv_buffer[0]) CHECK_CUDA(cudaFree(d_ghost_recv_buffer[0]));
        if (d_ghost_recv_buffer[1]) CHECK_CUDA(cudaFree(d_ghost_recv_buffer[1]));
        if (d_ghost_send_buffer) CHECK_CUDA(cudaFree(d_ghost_send_buffer));

        if (d_scalar_host_ptr) CHECK_CUDA(cudaFreeHost(d_scalar_host_ptr));
        if (d_scalar_device) CHECK_CUDA(cudaFree(d_scalar_device));
        if (d_spmv_buffer) CHECK_CUDA(cudaFree(d_spmv_buffer));

        if (stream_compute) CHECK_CUDA(cudaStreamDestroy(stream_compute));
        if (stream_comm) CHECK_CUDA(cudaStreamDestroy(stream_comm));
        if (stream_reduce) CHECK_CUDA(cudaStreamDestroy(stream_reduce));
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

// ArnoldiRunner class: High-level orchestrator
struct ArnoldiRunner {
    int num_gpus;
    std::vector<DeviceContext> device_contexts;
    NcclContext nccl_context;
    ArnoldiParams params;
    std::vector<CSRHost::GhostMap> ghost_maps;

    // Host-side data for small H_m matrix and wH vector
    Eigen::MatrixXd H_m; // Hessenberg matrix
    Eigen::VectorXd e1;  // First canonical basis vector
    Eigen::VectorXd wH;  // exp(t*H_m)*e1
    double v_norm;       // Norm of initial vector v (before normalization)

    // Global matrix dimensions
    int global_rows;
    int global_cols;

    // Helper to get global row offset for a given GPU
    int get_global_row_offset(int gpu_id) const {
        int offset = 0;
        for (int i = 0; i < gpu_id; ++i) {
            offset += device_contexts[i].local_rows;
        }
        return offset;
    }

    ArnoldiRunner(int n_gpus, const ArnoldiParams& arnoldi_params, int total_rows, int total_cols)
        : num_gpus(n_gpus), nccl_context(n_gpus), params(arnoldi_params), global_rows(total_rows), global_cols(total_cols) {
        device_contexts.reserve(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            device_contexts.emplace_back(i);
        }
    }

    void init_all_devices(const std::vector<CSRHost>& partitions, const std::vector<int>& device_ids, const std::vector<CSRHost::GhostMap>& ghost_maps_in) {
        // Initialize NCCL communicator
        nccl_context.init_all(device_ids);
        this->ghost_maps = ghost_maps_in; // Store ghost maps

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i)); // Set device before initializing context
            device_contexts[i].init_handles();
            device_contexts[i].create_streams();
            device_contexts[i].alloc_from(partitions[i], global_cols);
            device_contexts[i].create_descriptors(global_cols, ghost_maps[i], params.m); // Pass m_arnoldi
            // Enable peer access between all pairs of GPUs
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    CHECK_CUDA(cudaSetDevice(device_ids[i]));
                    
                    // Check if peer access is supported
                    int can_access;
                    cudaError_t status = cudaDeviceCanAccessPeer(&can_access, device_ids[i], device_ids[j]);
                    if (status == cudaSuccess && can_access) {
                        // Try to enable peer access, ignore if already enabled
                        cudaError_t peer_status = cudaDeviceEnablePeerAccess(device_ids[j], 0);
                        if (peer_status != cudaSuccess && peer_status != cudaErrorPeerAccessAlreadyEnabled) {
                            std::cerr << "Warning: Could not enable peer access between GPU " 
                                      << device_ids[i] << " and GPU " << device_ids[j] 
                                      << ": " << cudaGetErrorString(peer_status) << std::endl;
                        }
                    } else {
                        std::cerr << "Warning: Peer access not supported between GPU " 
                                  << device_ids[i] << " and GPU " << device_ids[j] << std::endl;
                    }
                }
            }
        }
    }

    void compute_expmv(const std::vector<double>& v_host, std::vector<double>& y_host) {
        // Full procedure: normalize q1 -> Arnoldi restarts -> small exponentiation -> lift -> restart/finish
        // This will be the main orchestration method.
        // Details will be filled in subsequent steps.
        // 1. Initialize q1 (norm of v and scale v)
        init_q1(v_host);

        // 2. Arnoldi iterations (main loop with restarts)
        for (int restart_count = 0; restart_count < params.max_restarts; ++restart_count) {
            // Reset H_m for new Arnoldi run
            H_m.setZero(params.m + 1, params.m);

            // Arnoldi loop for j = 1 to m
            for (int j = 0; j < params.m; ++j) {
                // A) Prepare q_j (ghost exchange and on-diag SpMV)
                ghost_exchange_qj(j); // Exchange ghost data for current q_j
                spmv_on_off(j);       // Compute A*q_j and store in d_w

                // B) Orthogonalization (Modified Gram-Schmidt)
                orthogonalize_mgs(j);

                // C) Normalize new vector and add to basis
                normalize_new_vector(j);

                // D) Store H_m column (h_{1..j,j} and h_{j+1,j})
                store_H_column(j);
            }

            // After m steps:
            // E) Compute small exponentiation on CPU: wH = exp(t*H_m)*e1
            small_expm_and_lift();

            // F) Evaluate residual and decide on restart
            if (residual_estimate()) {
                break; // Converged
            } else {
                // If not converged, d_y (computed in small_expm_and_lift) becomes the new v for restart
                // This is handled implicitly as d_y will be used as the new initial vector for the next Arnoldi run
            }
        }

        // G) Copy final y from device to host
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[i].stream_compute)); // Ensure all compute is done
            int global_offset = get_global_row_offset(i);
            CHECK_CUDA(cudaMemcpyAsync(y_host.data() + global_offset,
                                       device_contexts[i].d_y, 
                                       sizeof(double) * device_contexts[i].local_rows, 
                                       cudaMemcpyDeviceToHost, 
                                       device_contexts[i].stream_compute));
        }
        // Need to synchronize all streams before y_host is fully ready.
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[i].stream_compute));
        }

    }

    // Internal methods (details to be implemented)
    void init_q1(const std::vector<double>& v_host) {
        std::vector<double> local_v_norms_squared(num_gpus);

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));
            DeviceContext& dc = device_contexts[i];
            int global_offset = get_global_row_offset(i);

            // Copy local part of v_host to d_q on device
            CHECK_CUDA(cudaMemcpyAsync(dc.d_q, 
                                       v_host.data() + global_offset,
                                       sizeof(double) * dc.local_rows,
                                       cudaMemcpyHostToDevice,
                                       dc.stream_compute));
            CHECK_CUDA(cudaStreamSynchronize(dc.stream_compute)); // Ensure copy is done before norm

            double local_norm = 0.0;
            // Calculate local norm (temporarily use default stream for HOST pointer mode)
            CHECK_CUBLAS(cublasSetStream(dc.cublas_handle, NULL));
            CHECK_CUBLAS(cublasDnrm2(dc.cublas_handle, dc.local_rows, dc.d_q, 1, &local_norm));
            CHECK_CUBLAS(cublasSetStream(dc.cublas_handle, dc.stream_compute));
            
            double local_norm_sq = local_norm * local_norm; // Square the norm
            local_v_norms_squared[i] = local_norm_sq;
        }

        // AllReduce sum of local_v_norms_squared to get global_norm_sq
        // Each GPU must participate in the collective AllReduce operation
        
        // Group NCCL calls for better performance
        CHECK_NCCL(ncclGroupStart());
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            // Copy local norm squared to device
            CHECK_CUDA(cudaMemcpyAsync(dc.d_scalar_host_ptr, &local_v_norms_squared[i], sizeof(double), 
                                       cudaMemcpyHostToDevice, dc.stream_reduce));
            // All GPUs participate in AllReduce
            nccl_context.allreduce_sum(dc.d_scalar_host_ptr, 1, nccl_context.comms[i], dc.stream_reduce);
        }
        CHECK_NCCL(ncclGroupEnd());
        
        // Synchronize and get result from GPU 0
        double global_norm_sq = 0.0;
        CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
        CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_reduce));
        CHECK_CUDA(cudaMemcpy(&global_norm_sq, device_contexts[0].d_scalar_host_ptr, sizeof(double), cudaMemcpyDeviceToHost));

        double global_norm = std::sqrt(global_norm_sq);
        
        // Store the norm of the original vector v (before normalization)
        v_norm = global_norm;

        // Scale d_q on each device
        const double alpha = 1.0 / global_norm;
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(i));
            DeviceContext& dc = device_contexts[i];
            CHECK_CUBLAS(cublasDscal(dc.cublas_handle, dc.local_rows, &alpha, dc.d_q, 1));
            
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

        // Phase 1: Each GPU gathers its outgoing ghost data into its d_ghost_send_buffer
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            CSRHost::GhostMap& gm = ghost_maps[i];

            if (gm.outgoing_global_col_indices.empty()) continue; // Nothing to send

            // Allocate temporary device memory for global_indices_to_send for the kernel
            int* d_outgoing_global_col_indices;
            CHECK_CUDA(cudaMalloc(&d_outgoing_global_col_indices, sizeof(int) * gm.outgoing_global_col_indices.size()));
            CHECK_CUDA(cudaMemcpyAsync(d_outgoing_global_col_indices, gm.outgoing_global_col_indices.data(), 
                                       sizeof(int) * gm.outgoing_global_col_indices.size(), cudaMemcpyHostToDevice, dc.stream_compute));

            int blocks = (gm.outgoing_global_col_indices.size() + 255) / 256;
            gather_q_elements_for_send_kernel<<<blocks, 256, 0, dc.stream_compute>>>(dc.d_q, d_outgoing_global_col_indices, 
                                                                                   dc.d_ghost_send_buffer, gm.outgoing_global_col_indices.size(),
                                                                                   get_global_row_offset(i));
            CHECK_CUDA(cudaGetLastError());
            
            CHECK_CUDA(cudaFree(d_outgoing_global_col_indices)); // Free temporary device memory
        }

        // Phase 2: Perform p2p copies from send buffers to receive buffers
        for (int i = 0; i < num_gpus; ++i) { // i is the receiving GPU (consumer)
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& consumer_dc = device_contexts[i];
            CSRHost::GhostMap& consumer_gm = ghost_maps[i];
            int current_recv_buffer_idx = j % 2;
            double* current_d_ghost_recv_buffer = consumer_dc.d_ghost_recv_buffer[current_recv_buffer_idx];

            for (auto const& [owner_id, peer_info] : consumer_gm.incoming_peer_info) {
                // owner_id is the sending GPU
                size_t offset = peer_info.first;
                size_t count = peer_info.second;

                if (count > 0) {
                    DeviceContext& owner_dc = device_contexts[owner_id];
                    // Use owner_dc.d_ghost_send_buffer as source
                    // The offset in owner_dc.d_ghost_send_buffer needs to be determined by what owner_id sends to i
                    // This requires a map in GhostMap of owner_id: consumer_id -> {offset, count} for send buffer.
                    // Let's assume for now, that the `outgoing_peer_info` in owner_dc's ghost map can be used to determine source offset.
                    // The `outgoing_peer_info` for `owner_dc` stores `consumer_gpu_id -> {offset in its send buffer, count}`
                    size_t owner_send_offset_for_this_consumer = ghost_maps[owner_id].outgoing_peer_info[i].first;
                    
                    CHECK_CUDA(cudaMemcpyPeerAsync(current_d_ghost_recv_buffer + offset, consumer_dc.device_id, // dest
                                                   owner_dc.d_ghost_send_buffer + owner_send_offset_for_this_consumer, owner_dc.device_id, // src
                                                   sizeof(double) * count, consumer_dc.stream_comm));
                }
            }
        }

        // Synchronize communication stream for all devices to ensure all ghost data is received
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[i].stream_comm));
        }
    }

    void gather_qj_from_Vm(int j) {
        // Gather full vector q_j from V_m[j] into d_q_full on each GPU
        // This is needed because SpMV requires the full vector (size global_cols)
        
        // Validate j is in bounds
        if (j < 0 || j > params.m) {
            std::cerr << "ERROR: gather_qj_from_Vm called with j=" << j << ", but valid range is [0.." << params.m << "]" << std::endl;
            exit(EXIT_FAILURE);
        }
        
        for (int dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
            CHECK_CUDA(cudaSetDevice(device_contexts[dst_gpu].device_id));
            DeviceContext& dst_dc = device_contexts[dst_gpu];
            
            // Check if d_q_full was allocated
            if (!dst_dc.d_q_full) {
                std::cerr << "ERROR: d_q_full is NULL on GPU " << dst_gpu << "!" << std::endl;
                exit(EXIT_FAILURE);
            }
            
            // Zero out d_q_full first
            CHECK_CUDA(cudaMemsetAsync(dst_dc.d_q_full, 0, sizeof(double) * global_cols, dst_dc.stream_compute));
            
            // Copy local and remote parts from V_m[j] into d_q_full
            for (int src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
                DeviceContext& src_dc = device_contexts[src_gpu];
                int src_offset = get_global_row_offset(src_gpu);
                int src_size = src_dc.local_rows;
                
                // Pointer to V_m[j] on src_gpu
                double* d_qj_src = src_dc.d_V_m + j * src_dc.local_rows;
                
                if (dst_gpu == src_gpu) {
                    // Local copy from V_m[j]
                    cudaError_t err = cudaMemcpyAsync(dst_dc.d_q_full + src_offset, d_qj_src, 
                                               sizeof(double) * src_size, 
                                               cudaMemcpyDeviceToDevice, dst_dc.stream_compute);
                    if (err != cudaSuccess) {
                        std::cerr << "ERROR in cudaMemcpyAsync (local) for j=" << j << ", dst_gpu=" << dst_gpu 
                                  << ", src_gpu=" << src_gpu << ": " << cudaGetErrorString(err) << std::endl;
                        std::cerr << "  dst address: " << (void*)(dst_dc.d_q_full + src_offset) << std::endl;
                        std::cerr << "  src address: " << (void*)d_qj_src << std::endl;
                        std::cerr << "  size: " << src_size << " doubles" << std::endl;
                        exit(EXIT_FAILURE);
                    }
                } else {
                    // Peer copy from V_m[j] on src_gpu to d_q_full on dst_gpu
                    cudaError_t err = cudaMemcpyPeerAsync(dst_dc.d_q_full + src_offset, dst_gpu,
                                                   d_qj_src, src_gpu,
                                                   sizeof(double) * src_size, dst_dc.stream_compute);
                    if (err != cudaSuccess) {
                        std::cerr << "ERROR in cudaMemcpyPeerAsync for j=" << j << ", dst_gpu=" << dst_gpu 
                                  << ", src_gpu=" << src_gpu << ": " << cudaGetErrorString(err) << std::endl;
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        
        // Synchronize all streams - this is where errors from async operations will surface
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            cudaError_t err = cudaStreamSynchronize(device_contexts[i].stream_compute);
            if (err != cudaSuccess) {
                std::cerr << "ERROR during stream synchronize after gather, GPU " << i << ", j=" << j << ": " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    void spmv_on_off(int j) {

        // Gather full vector q_j from V_m into d_q_full on all GPUs
        gather_qj_from_Vm(j);
        
        // Phase 1: Launch on-diag SpMV on all GPUs in parallel
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            
            
            // Zero out d_w before accumulation
            cudaError_t err = cudaMemsetAsync(dc.d_w, 0, sizeof(double) * dc.local_rows, dc.stream_compute);
            if (err != cudaSuccess) {
                std::cerr << "ERROR in cudaMemsetAsync for d_w, GPU " << i << ", j=" << j << ": " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
            
            // Synchronize memset before SpMV
            CHECK_CUDA(cudaStreamSynchronize(dc.stream_compute));
            
            // Recreate vec_q_descr to ensure it points to current d_q_full
            // This avoids potential issues with descriptor caching
            if (dc.vec_q_descr) {
                CHECK_CUSPARSE(cusparseDestroyDnVec(dc.vec_q_descr));
            }
            CHECK_CUSPARSE(cusparseCreateDnVec(&dc.vec_q_descr, global_cols, dc.d_q_full, CUDA_R_64F));
            
            // Debug: verify d_q_full is accessible before SpMV
            if (j < 3) {
                double first_elem_q = -999.0;
                cudaError_t err_test = cudaMemcpy(&first_elem_q, dc.d_q_full, sizeof(double), cudaMemcpyDeviceToHost);
                if (err_test != cudaSuccess) {
                    std::cerr << "ERROR: Cannot read d_q_full before SpMV! GPU " << i << ", j=" << j << ": " << cudaGetErrorString(err_test) << std::endl;
                    exit(EXIT_FAILURE);
                }
                std::cout << "      d_q_full[0] before SpMV = " << first_elem_q << std::endl;
            }
            
            // Perform on-diag SpMV: A_on * q_j -> w (partial result)
            if (dc.local_nnz > 0) {
                const double alpha = 1.0, beta = 0.0;
                CHECK_CUSPARSE(cusparseSpMV(dc.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, dc.matA_descr, dc.vec_q_descr, &beta, dc.vec_w_descr,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dc.d_spmv_buffer));
            }
            
            // Synchronize and check for errors after SpMV
            CHECK_CUDA(cudaStreamSynchronize(dc.stream_compute));
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "ERROR after SpMV on GPU " << i << ", j=" << j << ": " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // Phase 2: Wait for ghost exchange to complete, then perform off-diag SpMV
        // Create events to synchronize between ghost exchange and off-diag computation
        std::vector<cudaEvent_t> ghost_ready_events(num_gpus);
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            CHECK_CUDA(cudaEventCreate(&ghost_ready_events[i]));
            CHECK_CUDA(cudaEventRecord(ghost_ready_events[i], device_contexts[i].stream_comm));
        }

        // Phase 3: Off-diag SpMV using ghost data
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            DeviceContext& dc = device_contexts[i];
            CSRHost::GhostMap& gm = ghost_maps[i];
            
            // Wait for ghost data to be ready
            CHECK_CUDA(cudaStreamWaitEvent(dc.stream_compute, ghost_ready_events[i], 0));
            
            // Perform off-diag SpMV using ghost data
            if (!gm.incoming_global_col_indices.empty()) {
                int current_recv_buffer_idx = j % 2;
                double* current_d_ghost_recv_buffer = dc.d_ghost_recv_buffer[current_recv_buffer_idx];
                
                // For each owner GPU, perform SpMV with ghost data and accumulate to d_w
                for (auto const& [owner_id, peer_info] : gm.incoming_peer_info) {
                    size_t offset = peer_info.first;
                    size_t count = peer_info.second;
                    
                    if (count > 0) {
                        // Create a temporary dense vector descriptor for ghost data
                        cusparseDnVecDescr_t ghost_vec_descr;
                        CHECK_CUSPARSE(cusparseCreateDnVec(&ghost_vec_descr, count, 
                                                          current_d_ghost_recv_buffer + offset, CUDA_R_64F));
                        
                        // Create a temporary CSR descriptor for off-diag part
                        // This is a simplified approach - in practice, you'd want to pre-build off-diag CSR matrices
                        // For now, we'll use a simple AXPY operation to accumulate ghost contributions
                        const double alpha = 1.0;
                        CHECK_CUBLAS(cublasDaxpy(dc.cublas_handle, count, &alpha, 
                                                current_d_ghost_recv_buffer + offset, 1, dc.d_w, 1));
                        
                        CHECK_CUSPARSE(cusparseDestroyDnVec(ghost_vec_descr));
                    }
                }
            }
        }

        // Clean up events
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(device_contexts[i].device_id));
            CHECK_CUDA(cudaEventDestroy(ghost_ready_events[i]));
        }
        
    }

    void orthogonalize_mgs(int j) {
        

        // Modified Gram-Schmidt orthogonalization: w = w - sum_i (h_{i,j} * q_i)
        for (int i = 0; i <= j; ++i) {
            // Phase 1: Compute local dot products w^T * q_i on each GPU
            std::vector<double> local_dots(num_gpus);
            for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                DeviceContext& dc = device_contexts[gpu_id];
                
                // Validate index bounds
                if (i > j) {
                    std::cerr << "ERROR: MGS index out of order: i=" << i << " > j=" << j << std::endl;
                    exit(EXIT_FAILURE);
                }
                
                // Check if V_m[i] exists (should have been filled in iteration i-1, except V_m[0])
                if (i > params.m) {
                    std::cerr << "ERROR: Accessing V_m[" << i << "] but m=" << params.m << std::endl;
                    exit(EXIT_FAILURE);
                }
                
                // Get pointer to q_i (i-th column of V_m)
                double* d_q_i = dc.d_V_m + i * dc.local_rows;
                
                // Compute local dot product using device memory for result
                double local_dot = 0.0;
                
                // Debug: check parameters before cublasDdot
                if (dc.local_rows <= 0) {
                    std::cerr << "ERROR: GPU " << gpu_id << " has local_rows = " << dc.local_rows << std::endl;
                    exit(EXIT_FAILURE);
                }
                
                
                // Synchronize stream before cublasDdot to ensure d_w and d_q_i are ready
                CHECK_CUDA(cudaStreamSynchronize(dc.stream_compute));
                
                // Temporarily use default stream for cublasDdot with HOST pointer mode
                CHECK_CUBLAS(cublasSetStream(dc.cublas_handle, NULL));
                CHECK_CUBLAS(cublasDdot(dc.cublas_handle, dc.local_rows, dc.d_w, 1, d_q_i, 1, &local_dot));
                CHECK_CUBLAS(cublasSetStream(dc.cublas_handle, dc.stream_compute));
                local_dots[gpu_id] = local_dot;
            }

            // Phase 2: AllReduce to get global dot product h_{i,j}
            // Each GPU must participate in the collective AllReduce
            CHECK_NCCL(ncclGroupStart());
            for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                DeviceContext& dc = device_contexts[gpu_id];
                CHECK_CUDA(cudaMemcpyAsync(dc.d_scalar_host_ptr, &local_dots[gpu_id], sizeof(double), 
                                           cudaMemcpyHostToDevice, dc.stream_reduce));
                nccl_context.allreduce_sum(dc.d_scalar_host_ptr, 1, nccl_context.comms[gpu_id], dc.stream_reduce);
            }
            CHECK_NCCL(ncclGroupEnd());
            
            // Get result from GPU 0
            double h_ij = 0.0;
            CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
            CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_reduce));
            CHECK_CUDA(cudaMemcpy(&h_ij, device_contexts[0].d_scalar_host_ptr, sizeof(double), cudaMemcpyDeviceToHost));

            // Phase 3: Update w = w - h_{i,j} * q_i on each GPU
            const double alpha = -h_ij;
            for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                DeviceContext& dc = device_contexts[gpu_id];
                
                // Get pointer to q_i (i-th column of V_m)
                double* d_q_i = dc.d_V_m + i * dc.local_rows;
                
                // w = w + alpha * q_i (alpha is negative)
                CHECK_CUBLAS(cublasDaxpy(dc.cublas_handle, dc.local_rows, &alpha, d_q_i, 1, dc.d_w, 1));
            }

            // Store h_{i,j} in H_m (will be copied to host in store_H_column)
            if (i < params.m) {
                H_m(i, j) = h_ij;
            }
        }

        // Optional reorthogonalization if enabled
        if (params.reorthogonalize) {
            // Check orthogonality loss and perform second pass if needed
            // For simplicity, we'll always do a second pass if reorthogonalization is enabled
            for (int i = 0; i <= j; ++i) {
                // Repeat the same process as above
                std::vector<double> local_dots(num_gpus);
                for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                    DeviceContext& dc = device_contexts[gpu_id];
                    
                    double* d_q_i = dc.d_V_m + i * dc.local_rows;
                    double local_dot = 0.0;
                    
                    // Synchronize stream before cublasDdot to ensure d_w and d_q_i are ready
                    CHECK_CUDA(cudaStreamSynchronize(dc.stream_compute));
                    
                    // Temporarily use default stream for cublasDdot with HOST pointer mode
                    CHECK_CUBLAS(cublasSetStream(dc.cublas_handle, NULL));
                    CHECK_CUBLAS(cublasDdot(dc.cublas_handle, dc.local_rows, dc.d_w, 1, d_q_i, 1, &local_dot));
                    CHECK_CUBLAS(cublasSetStream(dc.cublas_handle, dc.stream_compute));
                    local_dots[gpu_id] = local_dot;
                }

                // AllReduce for reorthogonalization correction
                CHECK_NCCL(ncclGroupStart());
                for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                    DeviceContext& dc = device_contexts[gpu_id];
                    CHECK_CUDA(cudaMemcpyAsync(dc.d_scalar_host_ptr, &local_dots[gpu_id], sizeof(double), 
                                               cudaMemcpyHostToDevice, dc.stream_reduce));
                    nccl_context.allreduce_sum(dc.d_scalar_host_ptr, 1, nccl_context.comms[gpu_id], dc.stream_reduce);
                }
                CHECK_NCCL(ncclGroupEnd());
                
                double h_ij_correction = 0.0;
                CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
                CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_reduce));
                CHECK_CUDA(cudaMemcpy(&h_ij_correction, device_contexts[0].d_scalar_host_ptr, sizeof(double), cudaMemcpyDeviceToHost));

                const double alpha = -h_ij_correction;
                for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
                    CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
                    DeviceContext& dc = device_contexts[gpu_id];
                    
                    double* d_q_i = dc.d_V_m + i * dc.local_rows;
                    CHECK_CUBLAS(cublasDaxpy(dc.cublas_handle, dc.local_rows, &alpha, d_q_i, 1, dc.d_w, 1));
                }

                // Update H_m with correction
                if (i < params.m) {
                    H_m(i, j) += h_ij_correction;
                }
            }
        }
    }

    void normalize_new_vector(int j) {

        // Phase 1: Compute local norms squared on each GPU
        std::vector<double> local_norms_squared(num_gpus);
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            
            double local_norm = 0.0;
            CHECK_CUBLAS(cublasDnrm2(dc.cublas_handle, dc.local_rows, dc.d_w, 1, &local_norm));
            local_norms_squared[gpu_id] = local_norm * local_norm; // Square the norm
        }

        // Phase 2: AllReduce to get global norm squared
        // Each GPU must participate in the collective AllReduce
        CHECK_NCCL(ncclGroupStart());
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            CHECK_CUDA(cudaMemcpyAsync(dc.d_scalar_host_ptr, &local_norms_squared[gpu_id], sizeof(double), 
                                       cudaMemcpyHostToDevice, dc.stream_reduce));
            nccl_context.allreduce_sumsq(dc.d_scalar_host_ptr, 1, nccl_context.comms[gpu_id], dc.stream_reduce);
        }
        CHECK_NCCL(ncclGroupEnd());
        
        double global_norm_squared = 0.0;
        CHECK_CUDA(cudaSetDevice(device_contexts[0].device_id));
        CHECK_CUDA(cudaStreamSynchronize(device_contexts[0].stream_reduce));
        CHECK_CUDA(cudaMemcpy(&global_norm_squared, device_contexts[0].d_scalar_host_ptr, sizeof(double), cudaMemcpyDeviceToHost));

        double global_norm = std::sqrt(global_norm_squared);
        
        
        // Check for zero or invalid norm
        if (global_norm < 1e-14 || std::isnan(global_norm) || std::isinf(global_norm)) {
            std::cerr << "ERROR: Invalid norm in normalize_new_vector for j=" << j << std::endl;
            std::cerr << "  global_norm_squared = " << global_norm_squared << std::endl;
            std::cerr << "  global_norm = " << global_norm << std::endl;
            exit(EXIT_FAILURE);
        }

        // Phase 3: Normalize w and store as q_{j+1} (next Arnoldi vector)
        const double alpha = 1.0 / global_norm;
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            DeviceContext& dc = device_contexts[gpu_id];
            
            // Scale w to get normalized vector
            CHECK_CUBLAS(cublasDscal(dc.cublas_handle, dc.local_rows, &alpha, dc.d_w, 1));
            
            // Copy normalized w to q_{j+1} (next column of V_m)
            double* d_q_next = dc.d_V_m + (j + 1) * dc.local_rows;
            CHECK_CUDA(cudaMemcpyAsync(d_q_next, dc.d_w, sizeof(double) * dc.local_rows, 
                                       cudaMemcpyDeviceToDevice, dc.stream_compute));
            
            // Also update d_q for the next iteration (used as working vector)
            CHECK_CUDA(cudaMemcpyAsync(dc.d_q, dc.d_w, sizeof(double) * dc.local_rows, 
                                       cudaMemcpyDeviceToDevice, dc.stream_compute));
        }
        
        // Synchronize all GPUs to ensure V_m[j+1] is ready for next iteration
        for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
            CHECK_CUDA(cudaSetDevice(device_contexts[gpu_id].device_id));
            CHECK_CUDA(cudaDeviceSynchronize());  // Use deviceSync instead of streamSync for stronger guarantee
        }

        // Store h_{j+1,j} = global_norm in H_m
        if (j + 1 < params.m + 1) {
            H_m(j + 1, j) = global_norm;
        }
    }

    void store_H_column(int j) {
        
        // The H_m matrix is already being populated during orthogonalize_mgs and normalize_new_vector
        // This method is mainly for logging/debugging purposes
        // In a more sophisticated implementation, you might want to copy H_m to device memory
        // or perform additional validation here
        
        std::cout << "    H_m column " << j << " stored. Current H_m size: " << H_m.rows() << "x" << H_m.cols() << std::endl;
    }

    void small_expm_and_lift() {

        // Phase 1: Compute small matrix exponential on CPU using Eigen
        // Create e1 vector (first canonical basis vector)
        e1 = Eigen::VectorXd::Zero(params.m + 1);
        e1(0) = 1.0;

        // Extract the m x m upper block of H_m (since H_m is (m+1) x m)
        // We only need the first m rows for the exponential computation
        Eigen::MatrixXd H_m_square = H_m.block(0, 0, params.m, params.m);
        
        // Compute exp(t * H_m_square) using stable matrix exponential
        Eigen::MatrixXd exp_tH_square = matrix_exp_stable(params.t * H_m_square);
        
        // Extend to (m+1) x (m+1) for consistency
        // exp(t*H_m) is approximated by a block structure
        Eigen::MatrixXd exp_tH = Eigen::MatrixXd::Zero(params.m + 1, params.m + 1);
        exp_tH.block(0, 0, params.m, params.m) = exp_tH_square;
        
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
            for (int i = 0; i <= params.m; ++i) {
                if (std::abs(wH(i)) > 1e-15) { // Skip near-zero coefficients
                    const double alpha = v_norm * wH(i);
                    double* d_V_i = dc.d_V_m + i * dc.local_rows;
                    
                    CHECK_CUBLAS(cublasDaxpy(dc.cublas_handle, dc.local_rows, &alpha, d_V_i, 1, dc.d_y, 1));
                }
            }
        }
    }

    bool residual_estimate() {
        
        // Residual-based convergence check: ||r_m|| ≈ |h_{m+1,m}| * ||e_m^T * exp(t*H_m) * e1||
        // where e_m is the m-th canonical basis vector
        
        // Use the m x m block of H_m for exponential
        Eigen::MatrixXd H_m_square = H_m.block(0, 0, params.m, params.m);
        Eigen::MatrixXd exp_tH_square = matrix_exp_stable(params.t * H_m_square);
        
        // e1_small is the first canonical basis vector for the m x m system
        Eigen::VectorXd e1_small = Eigen::VectorXd::Zero(params.m);
        e1_small(0) = 1.0;
        
        // Compute exp(t*H_m_square) * e1_small
        Eigen::VectorXd exp_tH_e1 = exp_tH_square * e1_small;
        
        // e_m is the last canonical basis vector
        double e_m_exp_tH_e1 = exp_tH_e1(params.m - 1);
        
        // Get h_{m+1,m} from H_m
        double h_m_plus_1_m = H_m(params.m, params.m - 1);
        
        // Estimate residual norm
        double residual_norm = std::abs(h_m_plus_1_m * e_m_exp_tH_e1);
        
        // Check convergence
        bool converged = (residual_norm <= params.tol);
        
        return converged;
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

    // Parse command-line arguments
    int test_size = 1000; // Default size
    if (argc > 1) {
        test_size = std::atoi(argv[1]);
        if (test_size <= 0) {
            std::cerr << "Error: Matrix size must be a positive integer. Using default size 1000." << std::endl;
            test_size = 1000;
        }
    }
    std::cout << "Matrix size: " << test_size << std::endl;

    // Example usage
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
        std::vector<int> device_ids(num_gpus_to_use);
        for (int i = 0; i < num_gpus_to_use; ++i) {
            device_ids[i] = i;
        }

        // Create a simple test matrix (diagonal matrix for testing)
        CSRHost test_matrix;
        test_matrix.rows = test_size;
        test_matrix.cols = test_size;
        test_matrix.nnz = test_size;
        
        // Create diagonal matrix
        test_matrix.row_ptr.resize(test_matrix.rows + 1);
        test_matrix.col_idx.resize(test_matrix.nnz);
        test_matrix.values.resize(test_matrix.nnz);
        
        for (int i = 0; i < test_matrix.rows; ++i) {
            test_matrix.row_ptr[i] = i;
            test_matrix.col_idx[i] = i;
            // Use different diagonal values to avoid degenerate case
            // Values range from 1.0 to 2.0
            test_matrix.values[i] = 1.0 + (double)i / test_matrix.rows;
        }
        test_matrix.row_ptr[test_matrix.rows] = test_matrix.nnz;

        std::cout << "Created test matrix: " << test_matrix.rows << "x" << test_matrix.cols 
                  << " with " << test_matrix.nnz << " non-zeros" << std::endl;

        // Partition matrix across GPUs
        auto partitions = test_matrix.partition_rows(num_gpus_to_use);
        std::cout << "Partitioned matrix across " << num_gpus_to_use << " GPUs" << std::endl;

        // Build ghost maps
        std::vector<CSRHost::GhostMap> ghost_maps(num_gpus_to_use);
        for (int i = 0; i < num_gpus_to_use; ++i) {
            ghost_maps[i] = test_matrix.build_owner_ghost_maps(partitions, i, num_gpus_to_use);
        }

        // Set up Arnoldi parameters
        // Using m=30 for good balance between accuracy and speed
        ArnoldiParams params(30, 1.0, 1e-6, 3, true); // m=30, t=1.0, tol=1e-6, max_restarts=3

        // Create ArnoldiRunner
        ArnoldiRunner runner(num_gpus_to_use, params, test_matrix.rows, test_matrix.cols);
        
        // Initialize all devices
        runner.init_all_devices(partitions, device_ids, ghost_maps);

        // Create test vector v
        std::vector<double> v_host(test_matrix.rows, 0.1); // Initial vector with all elements = 0.1
        std::vector<double> y_host(test_matrix.rows, 0.0); // Result vector

        // Compute exp(tA)v
        runner.compute_expmv(v_host, y_host);

        // Print some results
        std::cout << "Computation completed. Sample results:" << std::endl;
        for (int i = 0; i < std::min(10, (int)y_host.size()); ++i) {
            std::cout << "y[" << i << "] = " << y_host[i] << std::endl;
        }

        // Clean up
        runner.destroy();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}

