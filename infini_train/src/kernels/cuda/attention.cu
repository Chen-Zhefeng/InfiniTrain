#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include <cuda_runtime.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kFusedBlockSizeSmall = 128;
constexpr int kFusedBlockSizeLarge = 256;

template <typename T>
void FillTensor(const std::shared_ptr<Tensor> &tensor, T value, std::string_view context_identifier) {
    DispatchFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(), [=]<typename U>() { tensor->Fill<U>(static_cast<U>(value)); }, context_identifier);
}

std::shared_ptr<Tensor> OnesLikeShape(const std::vector<int64_t> &dims, DataType dtype, const Device &device,
                                      std::string_view context_identifier) {
    auto output = std::make_shared<Tensor>(dims, dtype, device);
    FillTensor(output, 1.0f, context_identifier);
    return output;
}

struct CausalMaskCacheKey {
    int64_t q_len;
    int64_t kv_len;
    DataType dtype;

    bool operator==(const CausalMaskCacheKey &other) const {
        return q_len == other.q_len && kv_len == other.kv_len && dtype == other.dtype;
    }
};

struct CausalMaskCacheKeyHash {
    size_t operator()(const CausalMaskCacheKey &key) const {
        const size_t h1 = std::hash<int64_t>{}(key.q_len);
        const size_t h2 = std::hash<int64_t>{}(key.kv_len);
        const size_t h3 = std::hash<int>{}(static_cast<int>(key.dtype));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::shared_ptr<Tensor> GetCachedCausalMask(int64_t q_len, int64_t kv_len, DataType dtype, const Device &device) {
    static std::mutex cache_mutex;
    static std::unordered_map<CausalMaskCacheKey, std::shared_ptr<Tensor>, CausalMaskCacheKeyHash> causal_mask_cache;

    const CausalMaskCacheKey cache_key{q_len, kv_len, dtype};
    {
        std::lock_guard<std::mutex> guard(cache_mutex);
        auto it = causal_mask_cache.find(cache_key);
        if (it != causal_mask_cache.end()) {
            return it->second;
        }
    }

    auto lower_tri = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device.type(), "TrilForward"}, OnesLikeShape({q_len, kv_len}, dtype, device, "CUDA BuildCausalMask"), 0);
    auto causal_mask = (lower_tri->View({1, 1, q_len, kv_len}) == 0.0f);

    {
        std::lock_guard<std::mutex> guard(cache_mutex);
        auto [it, inserted] = causal_mask_cache.emplace(cache_key, causal_mask);
        if (!inserted) {
            return it->second;
        }
    }
    return causal_mask;
}

std::shared_ptr<Tensor> BuildCausalMask(const std::shared_ptr<Tensor> &scores) {
    CHECK_EQ(scores->Dims().size(), 4);
    return GetCachedCausalMask(scores->Dims()[2], scores->Dims()[3], scores->Dtype(), scores->GetDevice());
}

std::shared_ptr<Tensor> ApplyMasks(const std::shared_ptr<Tensor> &scores, const std::shared_ptr<Tensor> &attn_mask,
                                   bool is_causal) {
    auto masked_scores = scores;
    if (attn_mask) {
        masked_scores = masked_scores->MaskedFill(attn_mask, std::numeric_limits<float>::lowest());
    }
    if (is_causal) {
        masked_scores = masked_scores->MaskedFill(BuildCausalMask(masked_scores), std::numeric_limits<float>::lowest());
    }
    return masked_scores;
}

std::shared_ptr<Tensor> RecomputeAttentionProbabilities(const std::shared_ptr<Tensor> &query,
                                                        const std::shared_ptr<Tensor> &key,
                                                        const std::shared_ptr<Tensor> &attn_mask, bool is_causal,
                                                        double scale) {
    auto scores = query->Matmul(key->Transpose(-2, -1)) * static_cast<float>(scale);
    if (!attn_mask && !is_causal) {
        return nn::function::Softmax(scores, -1);
    }
    scores = ApplyMasks(scores, attn_mask, is_causal);
    return nn::function::Softmax(scores, -1);
}

std::shared_ptr<Tensor> SliceHeadRange(const std::shared_ptr<Tensor> &tensor, int64_t head_start, int64_t head_end) {
    CHECK(tensor);
    CHECK_EQ(tensor->Dims().size(), 4);
    CHECK_GE(head_start, 0);
    CHECK_LE(head_end, tensor->Dims()[1]);
    CHECK_LT(head_start, head_end);
    const auto &dims = tensor->Dims();
    return tensor->Slice({0, head_start, 0, 0}, {dims[0], head_end, dims[2], dims[3]}, {1, 1, 1, 1});
}

std::shared_ptr<Tensor> SelectMaskForHeadRange(const std::shared_ptr<Tensor> &attn_mask, int64_t head_start,
                                               int64_t head_end) {
    if (!attn_mask) {
        return nullptr;
    }
    CHECK_EQ(attn_mask->Dims().size(), 4);
    const int64_t mask_heads = attn_mask->Dims()[1];
    if (mask_heads == 1) {
        return attn_mask;
    }
    return SliceHeadRange(attn_mask, head_start, head_end);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, int64_t>
MaybeExpandKeyValueForGQA(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                          const std::shared_ptr<Tensor> &value, bool enable_gqa) {
    const int64_t query_heads = query->Dims()[1];
    const int64_t key_heads = key->Dims()[1];
    const int64_t value_heads = value->Dims()[1];

    CHECK_EQ(key_heads, value_heads);

    if (!enable_gqa) {
        CHECK_EQ(query_heads, key_heads);
        return {key, value, 1};
    }

    CHECK_GE(query_heads, key_heads);
    CHECK_EQ(query_heads % key_heads, 0);
    const int64_t n_rep = query_heads / key_heads;
    if (n_rep == 1) {
        return {key, value, 1};
    }
    return {key->RepeatInterleave(n_rep, 1), value->RepeatInterleave(n_rep, 1), n_rep};
}

std::shared_ptr<Tensor> ReduceExpandedGradientForGQA(const std::shared_ptr<Tensor> &grad_expanded, int64_t kv_heads,
                                                     int64_t n_rep) {
    if (n_rep == 1) {
        return grad_expanded;
    }

    const auto &dims = grad_expanded->Dims();
    CHECK_EQ(dims.size(), 4);
    CHECK_EQ(dims[1], kv_heads * n_rep);
    return grad_expanded->View({dims[0], kv_heads, n_rep, dims[2], dims[3]})->Sum(2, false);
}

inline cudaStream_t GetCudaStream(const Device &device) {
    return dynamic_cast<infini_train::core::cuda::CudaStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

inline bool IsFlashSupportedDtype(const std::shared_ptr<Tensor> &tensor) {
    if (!tensor) {
        return false;
    }
    return tensor->Dtype() == DataType::kFLOAT32 || tensor->Dtype() == DataType::kBFLOAT16;
}

inline std::shared_ptr<Tensor> ToFloatTensor(const std::shared_ptr<Tensor> &tensor) {
    if (!tensor || tensor->Dtype() == DataType::kFLOAT32) {
        return tensor;
    }
    return std::make_shared<Tensor>(tensor->To(DataType::kFLOAT32));
}

inline std::shared_ptr<Tensor> CastTensorTo(const std::shared_ptr<Tensor> &tensor, DataType dtype) {
    if (!tensor || tensor->Dtype() == dtype) {
        return tensor;
    }
    return std::make_shared<Tensor>(tensor->To(dtype));
}

inline bool CanUseFusedPath(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                            const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask) {
    // Current fused kernel is only enabled for GQA-like layouts to avoid
    // regressions observed on standard MHA shapes.
    if (attn_mask) {
        return false;
    }
    if (!IsFlashSupportedDtype(query) || !IsFlashSupportedDtype(key) || !IsFlashSupportedDtype(value)) {
        return false;
    }
    if (query->Dtype() != key->Dtype() || query->Dtype() != value->Dtype()) {
        return false;
    }
    if (query->Dims().size() != 4 || key->Dims().size() != 4 || value->Dims().size() != 4) {
        return false;
    }
    if (query->Dims()[1] == key->Dims()[1]) {
        return false;
    }
    if (query->Dims()[3] > 256 || value->Dims()[3] > 256) {
        return false;
    }
    return true;
}

inline int SelectFusedBlockSize(int64_t head_dim, int64_t value_dim, int64_t kv_len) {
    // A100 (SM80) generally benefits from higher occupancy for small head/value dims.
    // Use a smaller block when math per row is light; otherwise keep larger block.
    if (head_dim <= 64 && value_dim <= 64 && kv_len <= 2048) {
        return kFusedBlockSizeSmall;
    }
    return kFusedBlockSizeLarge;
}

template <int BLOCK_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionForwardKernel(const T *query, const T *key, const T *value, T *output, float *lse,
                                            int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                            int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                            bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                            uint64_t rng_offset);

template <int BLOCK_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionBackwardKernel(const T *query, const T *key, const T *value, const T *grad_output,
                                             const float *lse, float *grad_query, float *grad_key, float *grad_value,
                                             int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                             int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                             bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                             uint64_t rng_offset);

template <int BLOCK_SIZE, bool USE_DROPOUT, typename T>
void LaunchFusedForwardKernel(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                              const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &output,
                              const std::shared_ptr<Tensor> &lse, int64_t batch, int64_t query_heads, int64_t key_heads,
                              int64_t q_len, int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                              bool is_causal, float scale, float dropout_p, uint64_t rng_seed, uint64_t rng_offset,
                              cudaStream_t cuda_stream) {
    const int64_t rows = batch * query_heads * q_len;
    const size_t shared_mem = static_cast<size_t>(head_dim + value_dim) * sizeof(float);
    FusedAttentionForwardKernel<BLOCK_SIZE, USE_DROPOUT, T><<<rows, BLOCK_SIZE, shared_mem, cuda_stream>>>(
        static_cast<const T *>(query->DataPtr()), static_cast<const T *>(key->DataPtr()),
        static_cast<const T *>(value->DataPtr()), static_cast<T *>(output->DataPtr()),
        static_cast<float *>(lse->DataPtr()), batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
        group_size, is_causal, scale, dropout_p, rng_seed, rng_offset);
}

template <int BLOCK_SIZE, bool USE_DROPOUT, typename T>
void LaunchFusedBackwardKernel(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                               const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &grad_output,
                               const std::shared_ptr<Tensor> &lse, const std::shared_ptr<Tensor> &grad_query,
                               const std::shared_ptr<Tensor> &grad_key, const std::shared_ptr<Tensor> &grad_value,
                               int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len, int64_t kv_len,
                               int64_t head_dim, int64_t value_dim, int64_t group_size, bool is_causal, float scale,
                               float dropout_p, uint64_t rng_seed, uint64_t rng_offset, cudaStream_t cuda_stream) {
    const int64_t rows = batch * query_heads * q_len;
    const size_t shared_mem = static_cast<size_t>(head_dim + value_dim) * sizeof(float);
    FusedAttentionBackwardKernel<BLOCK_SIZE, USE_DROPOUT, T><<<rows, BLOCK_SIZE, shared_mem, cuda_stream>>>(
        static_cast<const T *>(query->DataPtr()), static_cast<const T *>(key->DataPtr()),
        static_cast<const T *>(value->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
        static_cast<const float *>(lse->DataPtr()), static_cast<float *>(grad_query->DataPtr()),
        static_cast<float *>(grad_key->DataPtr()), static_cast<float *>(grad_value->DataPtr()), batch, query_heads,
        key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, scale, dropout_p, rng_seed, rng_offset);
}

__device__ inline float DeterministicUniform(uint64_t seed, uint64_t offset, uint64_t idx) {
    uint64_t x = seed ^ (offset + idx + 0x9e3779b97f4a7c15ULL);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x *= 0x2545F4914F6CDD1DULL;
    return static_cast<float>((x >> 40) & 0xFFFFFF) / static_cast<float>(1 << 24);
}

__device__ inline float WarpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_down_sync(0xFFFFFFFF, value, offset); }
    return value;
}

template <int BLOCK_SIZE> __device__ inline float BlockReduceSum(float value, float *shared_warp_sums) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size.");
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = BLOCK_SIZE / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    value = WarpReduceSum(value);
    if (lane == 0) {
        shared_warp_sums[warp_id] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = lane < kWarpsPerBlock ? shared_warp_sums[lane] : 0.0f;
        block_sum = WarpReduceSum(block_sum);
        if (lane == 0) {
            shared_warp_sums[0] = block_sum;
        }
    }
    __syncthreads();
    return shared_warp_sums[0];
}

template <typename T> __device__ inline float ConvertToFloat(T value) { return static_cast<float>(value); }

template <> __device__ inline float ConvertToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T> __device__ inline T ConvertFromFloat(float value) { return static_cast<T>(value); }

template <> __device__ inline __nv_bfloat16 ConvertFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <int BLOCK_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionForwardKernel(const T *query, const T *key, const T *value, T *output, float *lse,
                                            int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                            int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                            bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                            uint64_t rng_offset) {
    const int64_t row = blockIdx.x;
    if (row >= batch * query_heads * q_len) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t q_pos = row % q_len;
    const int64_t q_head = (row / q_len) % query_heads;
    const int64_t batch_idx = row / (query_heads * q_len);
    const int64_t kv_head = q_head / group_size;
    const float keep_prob = USE_DROPOUT ? (1.0f - dropout_p) : 1.0f;

    __shared__ float s_reduce[BLOCK_SIZE / 32];
    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_alpha;
    __shared__ float s_beta;
    extern __shared__ float s_mem[];
    float *s_q = s_mem;
    float *s_acc = s_q + head_dim;

    const int64_t q_base = (((batch_idx * query_heads + q_head) * q_len + q_pos) * head_dim);
    for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) { s_q[d] = ConvertToFloat(query[q_base + d]); }

    for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) { s_acc[dv] = 0.0f; }
    if (tid == 0) {
        s_m = -INFINITY;
        s_l = 0.0f;
    }
    __syncthreads();

    for (int64_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
        if (is_causal && kv_pos > q_pos) {
            continue;
        }

        float partial = 0.0f;
        for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
            const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
            partial = fmaf(s_q[d], ConvertToFloat(key[k_idx]), partial);
        }

        const float qk = BlockReduceSum<BLOCK_SIZE>(partial, s_reduce);

        if (tid == 0) {
            const float score = qk * scale;
            const uint64_t dropout_idx = ((row * kv_len) + kv_pos);
            const bool keep = !USE_DROPOUT || (DeterministicUniform(rng_seed, rng_offset, dropout_idx) < keep_prob);

            if (!keep) {
                s_alpha = 1.0f;
                s_beta = 0.0f;
            } else {
                const float m_new = fmaxf(s_m, score);
                s_alpha = __expf(s_m - m_new);
                s_beta = __expf(score - m_new);
                s_l = s_l * s_alpha + s_beta;
                s_m = m_new;
                if (USE_DROPOUT) {
                    s_beta /= keep_prob;
                }
            }
        }
        __syncthreads();

        for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
            const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
            s_acc[dv] = fmaf(s_beta, ConvertToFloat(value[v_idx]), s_acc[dv] * s_alpha);
        }
        __syncthreads();
    }

    for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
        const int64_t out_idx = (((batch_idx * query_heads + q_head) * q_len + q_pos) * value_dim + dv);
        output[out_idx] = ConvertFromFloat<T>(s_l > 0.0f ? s_acc[dv] / s_l : 0.0f);
    }
    if (tid == 0) {
        lse[row] = s_l > 0.0f ? (__logf(s_l) + s_m) : -INFINITY;
    }
}

template <int BLOCK_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionBackwardKernel(const T *query, const T *key, const T *value, const T *grad_output,
                                             const float *lse, float *grad_query, float *grad_key, float *grad_value,
                                             int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                             int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                             bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                             uint64_t rng_offset) {
    const int64_t row = blockIdx.x;
    if (row >= batch * query_heads * q_len) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t q_pos = row % q_len;
    const int64_t q_head = (row / q_len) % query_heads;
    const int64_t batch_idx = row / (query_heads * q_len);
    const int64_t kv_head = q_head / group_size;
    const float keep_prob = USE_DROPOUT ? (1.0f - dropout_p) : 1.0f;
    const float lse_row = lse[row];
    const int64_t q_base = (((batch_idx * query_heads + q_head) * q_len + q_pos) * head_dim);
    const int64_t go_base = (((batch_idx * query_heads + q_head) * q_len + q_pos) * value_dim);

    __shared__ float s_reduce[BLOCK_SIZE / 32];
    __shared__ float s_sum_term;
    extern __shared__ float s_mem[];
    float *s_q = s_mem;
    float *s_go = s_q + head_dim;

    for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) { s_q[d] = ConvertToFloat(query[q_base + d]); }
    for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) { s_go[dv] = ConvertToFloat(grad_output[go_base + dv]); }
    __syncthreads();

    float local_sum_term = 0.0f;
    for (int64_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
        if (is_causal && kv_pos > q_pos) {
            continue;
        }

        float qk_partial = 0.0f;
        for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
            const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
            qk_partial = fmaf(s_q[d], ConvertToFloat(key[k_idx]), qk_partial);
        }
        const float qk = BlockReduceSum<BLOCK_SIZE>(qk_partial, s_reduce);
        const float score = qk * scale;
        const float prob_logit = fminf(score - lse_row, 0.0f);
        const float prob = __expf(prob_logit);

        float dprob_partial = 0.0f;
        for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
            const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
            dprob_partial = fmaf(s_go[dv], ConvertToFloat(value[v_idx]), dprob_partial);
        }
        const float dprob_sum = BlockReduceSum<BLOCK_SIZE>(dprob_partial, s_reduce);

        if (tid == 0) {
            float dprob = dprob_sum;
            if (!isfinite(dprob)) {
                dprob = 0.0f;
            }
            if (USE_DROPOUT) {
                const uint64_t dropout_idx = ((row * kv_len) + kv_pos);
                const bool keep = DeterministicUniform(rng_seed, rng_offset, dropout_idx) < keep_prob;
                dprob = keep ? (dprob / keep_prob) : 0.0f;
            }
            const float contrib = dprob * prob;
            local_sum_term += isfinite(contrib) ? contrib : 0.0f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        // local_sum_term is accumulated only by lane 0 in this block.
        s_sum_term = local_sum_term;
    }
    __syncthreads();

    for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
        grad_query[(((batch_idx * query_heads + q_head) * q_len + q_pos) * head_dim + d)] = 0.0f;
    }
    __syncthreads();

    for (int64_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
        if (is_causal && kv_pos > q_pos) {
            continue;
        }

        float qk_partial = 0.0f;
        for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
            const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
            qk_partial = fmaf(s_q[d], ConvertToFloat(key[k_idx]), qk_partial);
        }
        const float qk = BlockReduceSum<BLOCK_SIZE>(qk_partial, s_reduce);
        const float score = qk * scale;
        const float prob_logit = fminf(score - lse_row, 0.0f);
        const float prob = __expf(prob_logit);

        float dprob_partial = 0.0f;
        for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
            const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
            dprob_partial = fmaf(s_go[dv], ConvertToFloat(value[v_idx]), dprob_partial);
        }
        float dprob = BlockReduceSum<BLOCK_SIZE>(dprob_partial, s_reduce);
        if (!isfinite(dprob)) {
            dprob = 0.0f;
        }
        float keep_scale = 1.0f;
        if (USE_DROPOUT) {
            const uint64_t dropout_idx = ((row * kv_len) + kv_pos);
            const bool keep = DeterministicUniform(rng_seed, rng_offset, dropout_idx) < keep_prob;
            keep_scale = keep ? (1.0f / keep_prob) : 0.0f;
            dprob *= keep_scale;
        }

        float ds = prob * (dprob - s_sum_term);
        if (!isfinite(ds)) {
            ds = 0.0f;
        }
        for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
            const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
            grad_query[q_base + d] += ds * scale * ConvertToFloat(key[k_idx]);
            atomicAdd(&grad_key[k_idx], ds * scale * s_q[d]);
        }
        for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
            const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
            atomicAdd(&grad_value[v_idx], prob * keep_scale * s_go[dv]);
        }
        __syncthreads();
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, uint64_t, uint64_t>
RunFallbackForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                   const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask, double dropout_p,
                   bool is_causal, double scale, bool enable_gqa) {
    // Fallback keeps the mathematical behavior of SDPA and is used as the
    // numerically stable path when fused-kernel constraints are not satisfied.
    CHECK_EQ(dropout_p, 0.0) << "Fallback path supports dropout_p == 0 only.";
    auto query_math = ToFloatTensor(query);
    auto key_math = ToFloatTensor(key);
    auto value_math = ToFloatTensor(value);
    auto attn_mask_math = ToFloatTensor(attn_mask);

    if (enable_gqa && query_math->Dims()[1] > key_math->Dims()[1]) {
        const int64_t query_heads = query_math->Dims()[1];
        const int64_t key_heads = key_math->Dims()[1];
        CHECK_EQ(query_heads % key_heads, 0);
        const int64_t n_rep = query_heads / key_heads;

        std::vector<std::shared_ptr<Tensor>> output_groups;
        output_groups.reserve(static_cast<size_t>(key_heads));

        for (int64_t kv_head = 0; kv_head < key_heads; ++kv_head) {
            const int64_t q_head_start = kv_head * n_rep;
            const int64_t q_head_end = q_head_start + n_rep;

            auto q_group = SliceHeadRange(query_math, q_head_start, q_head_end);
            auto k_single = SliceHeadRange(key_math, kv_head, kv_head + 1);
            auto v_single = SliceHeadRange(value_math, kv_head, kv_head + 1);
            auto k_group = k_single->RepeatInterleave(n_rep, 1);
            auto v_group = v_single->RepeatInterleave(n_rep, 1);
            auto group_mask = SelectMaskForHeadRange(attn_mask_math, q_head_start, q_head_end);

            auto probs = RecomputeAttentionProbabilities(q_group, k_group, group_mask, is_causal, scale);
            output_groups.push_back(probs->Matmul(v_group));

            probs.reset();
            group_mask.reset();
            v_group.reset();
            k_group.reset();
            v_single.reset();
            k_single.reset();
            q_group.reset();
        }
        auto output_math = nn::function::Concat(output_groups, 1);
        return {CastTensorTo(output_math, query->Dtype()), nullptr, 0, 0};
    }

    auto [key_used, value_used, _] = MaybeExpandKeyValueForGQA(query_math, key_math, value_math, enable_gqa);
    (void)_;

    auto probs = RecomputeAttentionProbabilities(query_math, key_used, attn_mask_math, is_causal, scale);
    auto output = CastTensorTo(probs->Matmul(value_used), query->Dtype());
    return {output, nullptr, 0, 0};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RunFallbackBackward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                    const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                    const std::shared_ptr<Tensor> &grad_output, double dropout_p, bool is_causal, double scale,
                    bool enable_gqa) {
    // Fallback backward avoids custom-kernel atomic contention and supports
    // both MHA and GQA while preserving numerical behavior.
    CHECK_EQ(dropout_p, 0.0) << "Fallback path supports dropout_p == 0 only.";

    auto query_math = ToFloatTensor(query);
    auto key_math = ToFloatTensor(key);
    auto value_math = ToFloatTensor(value);
    auto grad_output_math = ToFloatTensor(grad_output);
    auto attn_mask_math = ToFloatTensor(attn_mask);

    if (enable_gqa && query_math->Dims()[1] > key_math->Dims()[1]) {
        const int64_t query_heads = query_math->Dims()[1];
        const int64_t key_heads = key_math->Dims()[1];
        CHECK_EQ(query_heads % key_heads, 0);
        const int64_t n_rep = query_heads / key_heads;

        std::vector<std::shared_ptr<Tensor>> grad_query_groups;
        std::vector<std::shared_ptr<Tensor>> grad_key_groups;
        std::vector<std::shared_ptr<Tensor>> grad_value_groups;
        grad_query_groups.reserve(static_cast<size_t>(key_heads));
        grad_key_groups.reserve(static_cast<size_t>(key_heads));
        grad_value_groups.reserve(static_cast<size_t>(key_heads));

        for (int64_t kv_head = 0; kv_head < key_heads; ++kv_head) {
            const int64_t q_head_start = kv_head * n_rep;
            const int64_t q_head_end = q_head_start + n_rep;

            auto q_group = SliceHeadRange(query_math, q_head_start, q_head_end);
            auto k_single = SliceHeadRange(key_math, kv_head, kv_head + 1);
            auto v_single = SliceHeadRange(value_math, kv_head, kv_head + 1);
            auto go_group = SliceHeadRange(grad_output_math, q_head_start, q_head_end);
            auto k_group = k_single->RepeatInterleave(n_rep, 1);
            auto v_group = v_single->RepeatInterleave(n_rep, 1);
            auto group_mask = SelectMaskForHeadRange(attn_mask_math, q_head_start, q_head_end);

            auto probs = RecomputeAttentionProbabilities(q_group, k_group, group_mask, is_causal, scale);
            auto grad_value_expanded = probs->Transpose(-2, -1)->Matmul(go_group);

            auto grad_probs = go_group->Matmul(v_group->Transpose(-2, -1));
            auto sum_term = (grad_probs * probs)->Sum(-1, true);
            auto grad_scores = (grad_probs - sum_term) * probs;
            grad_probs.reset();
            sum_term.reset();
            probs.reset();

            if (group_mask) {
                grad_scores = grad_scores->MaskedFill(group_mask, 0.0f);
            }
            if (is_causal) {
                grad_scores = grad_scores->MaskedFill(BuildCausalMask(grad_scores), 0.0f);
            }

            grad_query_groups.push_back(grad_scores->Matmul(k_group) * static_cast<float>(scale));
            auto grad_key_expanded = grad_scores->Transpose(-2, -1)->Matmul(q_group) * static_cast<float>(scale);
            grad_scores.reset();

            grad_key_groups.push_back(grad_key_expanded
                                          ->View({grad_key_expanded->Dims()[0], 1, n_rep, grad_key_expanded->Dims()[2],
                                                  grad_key_expanded->Dims()[3]})
                                          ->Sum(2, false));
            grad_key_expanded.reset();

            grad_value_groups.push_back(grad_value_expanded
                                            ->View({grad_value_expanded->Dims()[0], 1, n_rep,
                                                    grad_value_expanded->Dims()[2], grad_value_expanded->Dims()[3]})
                                            ->Sum(2, false));
            grad_value_expanded.reset();

            group_mask.reset();
            v_group.reset();
            k_group.reset();
            go_group.reset();
            v_single.reset();
            k_single.reset();
            q_group.reset();
        }

        auto grad_query = CastTensorTo(nn::function::Concat(grad_query_groups, 1), query->Dtype());
        auto grad_key = CastTensorTo(nn::function::Concat(grad_key_groups, 1), key->Dtype());
        auto grad_value = CastTensorTo(nn::function::Concat(grad_value_groups, 1), value->Dtype());
        return {grad_query, grad_key, grad_value};
    }

    auto [key_used, value_used, n_rep] = MaybeExpandKeyValueForGQA(query_math, key_math, value_math, enable_gqa);
    auto probs = RecomputeAttentionProbabilities(query_math, key_used, attn_mask_math, is_causal, scale);
    auto grad_value_expanded = probs->Transpose(-2, -1)->Matmul(grad_output_math);

    auto grad_probs = grad_output_math->Matmul(value_used->Transpose(-2, -1));
    auto sum_term = (grad_probs * probs)->Sum(-1, true);
    auto grad_scores = (grad_probs - sum_term) * probs;
    grad_probs.reset();
    sum_term.reset();
    probs.reset();

    if (attn_mask_math) {
        grad_scores = grad_scores->MaskedFill(attn_mask_math, 0.0f);
    }
    if (is_causal) {
        grad_scores = grad_scores->MaskedFill(BuildCausalMask(grad_scores), 0.0f);
    }

    auto grad_query = CastTensorTo(grad_scores->Matmul(key_used) * static_cast<float>(scale), query->Dtype());
    auto grad_key_expanded = grad_scores->Transpose(-2, -1)->Matmul(query_math) * static_cast<float>(scale);
    grad_scores.reset();

    auto grad_key = CastTensorTo(ReduceExpandedGradientForGQA(grad_key_expanded, key->Dims()[1], n_rep), key->Dtype());
    grad_key_expanded.reset();

    auto grad_value
        = CastTensorTo(ReduceExpandedGradientForGQA(grad_value_expanded, value->Dims()[1], n_rep), value->Dtype());
    grad_value_expanded.reset();

    return {grad_query, grad_key, grad_value};
}
} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, uint64_t, uint64_t>
FlashAttentionForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                      const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask, double dropout_p,
                      bool is_causal, double scale, bool enable_gqa) {
    CHECK(query);
    CHECK(key);
    CHECK(value);

    CHECK(query->GetDevice().IsCUDA());
    CHECK(key->GetDevice().IsCUDA());
    CHECK(value->GetDevice().IsCUDA());
    if (attn_mask) {
        CHECK(attn_mask->GetDevice().IsCUDA());
    }

    CHECK_GE(dropout_p, 0.0);
    CHECK_LT(dropout_p, 1.0);
    CHECK_GT(scale, 0.0);
    CHECK_EQ(query->Dims().size(), 4);
    CHECK_EQ(key->Dims().size(), 4);
    CHECK_EQ(value->Dims().size(), 4);
    CHECK_EQ(query->Dims()[0], key->Dims()[0]);
    CHECK_EQ(query->Dims()[0], value->Dims()[0]);
    CHECK_EQ(query->Dims()[2], key->Dims()[2]);
    CHECK_EQ(query->Dims()[2], value->Dims()[2]);
    CHECK_EQ(query->Dims()[3], key->Dims()[3]);

    if (!CanUseFusedPath(query, key, value, attn_mask)) {
        return RunFallbackForward(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
    }

    const int64_t batch = query->Dims()[0];
    const int64_t query_heads = query->Dims()[1];
    const int64_t q_len = query->Dims()[2];
    const int64_t head_dim = query->Dims()[3];
    const int64_t key_heads = key->Dims()[1];
    const int64_t kv_len = key->Dims()[2];
    const int64_t value_dim = value->Dims()[3];

    int64_t group_size = 1;
    if (enable_gqa) {
        CHECK_EQ(query_heads % key_heads, 0);
        group_size = query_heads / key_heads;
    } else {
        CHECK_EQ(query_heads, key_heads);
    }

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch, query_heads, q_len, value_dim}, query->Dtype(),
                                           query->GetDevice());
    auto lse = std::make_shared<Tensor>(std::vector<int64_t>{batch, query_heads, q_len}, DataType::kFLOAT32,
                                        query->GetDevice());

    const uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const uint64_t offset = 0;

    auto cuda_stream = GetCudaStream(query->GetDevice());
    const int block_size = SelectFusedBlockSize(head_dim, value_dim, kv_len);
    const bool use_dropout = dropout_p > 0.0;
    const bool use_bf16 = query->Dtype() == DataType::kBFLOAT16;

    if (block_size == kFusedBlockSizeSmall) {
        if (use_dropout) {
            if (use_bf16) {
                LaunchFusedForwardKernel<kFusedBlockSizeSmall, true, __nv_bfloat16>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            } else {
                LaunchFusedForwardKernel<kFusedBlockSizeSmall, true, float>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            }
        } else {
            if (use_bf16) {
                LaunchFusedForwardKernel<kFusedBlockSizeSmall, false, __nv_bfloat16>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            } else {
                LaunchFusedForwardKernel<kFusedBlockSizeSmall, false, float>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            }
        }
    } else {
        if (use_dropout) {
            if (use_bf16) {
                LaunchFusedForwardKernel<kFusedBlockSizeLarge, true, __nv_bfloat16>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            } else {
                LaunchFusedForwardKernel<kFusedBlockSizeLarge, true, float>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            }
        } else {
            if (use_bf16) {
                LaunchFusedForwardKernel<kFusedBlockSizeLarge, false, __nv_bfloat16>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            } else {
                LaunchFusedForwardKernel<kFusedBlockSizeLarge, false, float>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            }
        }
    }
    const auto forward_launch_error = cudaGetLastError();
    CHECK_EQ(forward_launch_error, cudaSuccess)
        << "FusedAttentionForwardKernel launch failed: " << cudaGetErrorString(forward_launch_error);
    return {output, lse, seed, offset};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FlashAttentionBackward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                       const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                       const std::shared_ptr<Tensor> &lse, const std::shared_ptr<Tensor> &grad_output, double dropout_p,
                       bool is_causal, double scale, bool enable_gqa, uint64_t rng_seed, uint64_t rng_offset) {
    CHECK(query);
    CHECK(key);
    CHECK(value);
    CHECK(grad_output);

    CHECK(query->GetDevice().IsCUDA());
    CHECK(key->GetDevice().IsCUDA());
    CHECK(value->GetDevice().IsCUDA());
    CHECK(grad_output->GetDevice().IsCUDA());
    if (attn_mask) {
        CHECK(attn_mask->GetDevice().IsCUDA());
    }

    CHECK_GE(dropout_p, 0.0);
    CHECK_LT(dropout_p, 1.0);
    CHECK_GT(scale, 0.0);

    // Use the refactored fallback backward for all shapes. This avoids the
    // atomicAdd-heavy fused backward hotspot and keeps memory usage predictable.
    (void)lse;
    (void)rng_seed;
    (void)rng_offset;
    if (true) {
        return RunFallbackBackward(query, key, value, attn_mask, grad_output, dropout_p, is_causal, scale, enable_gqa);
    }

    const int64_t batch = query->Dims()[0];
    const int64_t query_heads = query->Dims()[1];
    const int64_t q_len = query->Dims()[2];
    const int64_t head_dim = query->Dims()[3];
    const int64_t key_heads = key->Dims()[1];
    const int64_t kv_len = key->Dims()[2];
    const int64_t value_dim = value->Dims()[3];

    int64_t group_size = 1;
    if (enable_gqa) {
        CHECK_EQ(query_heads % key_heads, 0);
        group_size = query_heads / key_heads;
    } else {
        CHECK_EQ(query_heads, key_heads);
    }

    auto grad_query = std::make_shared<Tensor>(query->Dims(), DataType::kFLOAT32, query->GetDevice());
    auto grad_key = std::make_shared<Tensor>(key->Dims(), DataType::kFLOAT32, key->GetDevice());
    auto grad_value = std::make_shared<Tensor>(value->Dims(), DataType::kFLOAT32, value->GetDevice());
    FillTensor(grad_key, 0.0f, "CUDA FlashAttentionBackward");
    FillTensor(grad_value, 0.0f, "CUDA FlashAttentionBackward");

    auto cuda_stream = GetCudaStream(query->GetDevice());
    const int block_size = SelectFusedBlockSize(head_dim, value_dim, kv_len);
    const bool use_dropout = dropout_p > 0.0;
    const bool use_bf16 = query->Dtype() == DataType::kBFLOAT16;

    if (block_size == kFusedBlockSizeSmall) {
        if (use_dropout) {
            if (use_bf16) {
                LaunchFusedBackwardKernel<kFusedBlockSizeSmall, true, __nv_bfloat16>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            } else {
                LaunchFusedBackwardKernel<kFusedBlockSizeSmall, true, float>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            }
        } else {
            if (use_bf16) {
                LaunchFusedBackwardKernel<kFusedBlockSizeSmall, false, __nv_bfloat16>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            } else {
                LaunchFusedBackwardKernel<kFusedBlockSizeSmall, false, float>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            }
        }
    } else {
        if (use_dropout) {
            if (use_bf16) {
                LaunchFusedBackwardKernel<kFusedBlockSizeLarge, true, __nv_bfloat16>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            } else {
                LaunchFusedBackwardKernel<kFusedBlockSizeLarge, true, float>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            }
        } else {
            if (use_bf16) {
                LaunchFusedBackwardKernel<kFusedBlockSizeLarge, false, __nv_bfloat16>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            } else {
                LaunchFusedBackwardKernel<kFusedBlockSizeLarge, false, float>(
                    query, key, value, grad_output, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            }
        }
    }
    const auto backward_launch_error = cudaGetLastError();
    CHECK_EQ(backward_launch_error, cudaSuccess)
        << "FusedAttentionBackwardKernel launch failed: " << cudaGetErrorString(backward_launch_error);
    return {CastTensorTo(grad_query, query->Dtype()), CastTensorTo(grad_key, key->Dtype()),
            CastTensorTo(grad_value, value->Dtype())};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ATTENTION_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_ATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_ATTENTION_KERNEL
