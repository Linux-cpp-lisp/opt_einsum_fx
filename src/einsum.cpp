// Parts of this file adapted from the CUDALibrarySamples:
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR

#include <tuple>
#include <algorithm>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.hpp>

#include "einsum.h"

using namespace torch::autograd;


template<>
struct CuTensorTypeTraits<at::Half> {
  static const cudaDataType_t cudaType = CUDA_R_16F;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_32F;
  typedef float ScalarType;
};

template<>
struct CuTensorTypeTraits<at::BFloat16> {
  static const cudaDataType_t cudaType = CUDA_R_16BF;
  static const cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_32F;
  typedef float ScalarType;
};

constexpr int kMaxNumModes_ = 40; // maximal number of modes supported by cuTENSOR

std::tuple<torch::Tensor, std::string, std::string, std::string> einsum_fw(
    std::string einstr,
    torch::Tensor op0,
    torch::Tensor op1
) {
    TORCH_CHECK(op0.is_cuda(), "both operands must be on a CUDA device");
    TORCH_CHECK(op0.device() == op1.device(), "both operands must be on the same device");

    torch::Tensor output_tensor;
    std::string modesA, modesB, modesC;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        op0.scalar_type(),
        "einsum", 
        [&] {
            Einsum<scalar_t, int64_t, kMaxNumModes_> myEinsum(
                einstr,
                op0.sizes().vec(),
                op0.strides().vec(),
                op1.sizes().vec(),
                op1.strides().vec()
            );
            if (!myEinsum.isInitialized()) {
                throw std::runtime_error("cutensor: Initialization failed.");
            }

            output_tensor = torch::empty(myEinsum.getOutputShape(), op0.options());

            size_t worksize = myEinsum.getWorksize();
            at::Tensor workspace = at::empty({static_cast<int>(worksize)}, at::CUDA(at::kByte));

            auto stream = at::cuda::getCurrentCUDAStream().stream();
            auto ret = myEinsum.execute(GetCuTensorHandle(),
                                        op0.data_ptr<scalar_t>(),
                                        op1.data_ptr<scalar_t>(),
                                        output_tensor.data_ptr<scalar_t>(),
                                        workspace.data_ptr<uint8_t>(),
                                        stream);

            if (! ret) throw std::runtime_error("cutensor: Launch failed.");

            modesA = std::string(myEinsum.modesA().begin(), myEinsum.modesA().end());
            modesB = std::string(myEinsum.modesB().begin(), myEinsum.modesB().end());
            modesC = std::string(myEinsum.modesC().begin(), myEinsum.modesC().end());
    });
    return {output_tensor, modesA, modesB, modesC};
}


class EinsumFunction : public torch::autograd::Function<EinsumFunction> {
    public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        std::string einstr,
        torch::Tensor op0,
        torch::Tensor op1
    ) {
        // - save for backward -
        ctx->save_for_backward({op0, op1});

        torch::Tensor out;
        std::string modesA, modesB, modesC;
        std::tie(out, modesA, modesB, modesC) = einsum_fw(einstr, op0, op1);
        ctx->saved_data["modesA"] = modesA;
        ctx->saved_data["modesB"] = modesB;
        ctx->saved_data["modesC"] = modesC;
        return out;
    }

    static tensor_list backward(
        AutogradContext *ctx,
        tensor_list grad_outputs
    ) {
        // see https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py#L587
        auto grad_out = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        torch::Tensor op0 = saved[0];
        torch::Tensor op1 = saved[1];
        std::string modesA, modesB, modesC;
        modesA = ctx->saved_data["modesA"].toString();
        modesB = ctx->saved_data["modesB"].toString();
        modesC = ctx->saved_data["modesC"].toString();
        // the original code in einsum.h uses the equation characters as mode ints,
        // so we can use them directly as those too:
        torch::Tensor gradA, gradB;
        std::string einstr;
        bool dims_to_expand = false;
        std::vector<int64_t> reshape_to;
        std::vector<int64_t> expand_to;
        size_t i;
        char tmp_c;
        if ( op0.requires_grad() ) {
            // First we have the output grad
            for ( char c : modesC ) {
                einstr.push_back(c);
            }
            einstr.push_back(',');
            // Then op1
            for ( char c : modesB ) {
                einstr.push_back(c);
            }
            einstr += "->";
            // Then the output indexes we get by contraction ---
            // Ones that get summed out in the original einsum we put back through duplication
            for ( i = 0; i < op0.dim(); i++ ) {
                tmp_c = modesA[i];
                if (
                    ( std::find(modesB.begin(), modesB.end(), tmp_c) != std::end(modesB) ) \
                    or \
                    ( std::find(modesC.begin(), modesC.end(), tmp_c) != std::end(modesC) )
                ) {
                    // We found the mode in the other op and the grad, so its in the contraction
                    einstr.push_back(tmp_c);
                    reshape_to.push_back(op0.sizes()[i]);
                    expand_to.push_back(op0.sizes()[i]);
                }
                else {
                    // We don't add this to the contraction, but get it by expansion
                    reshape_to.push_back(1);
                    expand_to.push_back(op0.sizes()[i]);
                    dims_to_expand = true;
                }
            }
            // Run the backward
            gradA = EinsumFunction::apply(einstr, grad_out, op1);
            if ( dims_to_expand ) {
                gradA = gradA.view(reshape_to).expand(expand_to);
            }
        }
        einstr.clear();
        reshape_to.clear();
        expand_to.clear();
        dims_to_expand = false;
        if ( op1.requires_grad() ) {
            // First we have the output grad
            for ( char c : modesC ) {
                einstr.push_back(static_cast<char>(c));
            }
            einstr.push_back(',');
            // Then op0
            for ( char c : modesA ) {
                einstr.push_back(static_cast<char>(c));
            }
            einstr += "->";
            // Then the output indexes we get by contraction ---
            // Ones that get summed out in the original einsum we put back through duplication
            for ( i = 0; i < op1.dim(); i++ ) {
                tmp_c = modesB[i];
                if (
                    ( std::find(modesA.begin(), modesA.end(), tmp_c) != std::end(modesA) ) \
                    or \
                    ( std::find(modesC.begin(), modesC.end(), tmp_c) != std::end(modesC) )
                ) {
                    // We found the mode in the other op and the grad, so its in the contraction
                    einstr.push_back(tmp_c);
                    reshape_to.push_back(op1.sizes()[i]);
                    expand_to.push_back(op1.sizes()[i]);
                }
                else {
                    // We don't add this to the contraction, but get it by expansion
                    reshape_to.push_back(1);
                    expand_to.push_back(op1.sizes()[i]);
                    dims_to_expand = true;
                }
            }
            // Run the backward
            gradB = EinsumFunction::apply(einstr, grad_out, op0);
            if ( dims_to_expand ) {
                gradB = gradB.view(reshape_to).expand(expand_to);
            }
        }
        return {
            op0.requires_grad() ? gradA : torch::Tensor(),
            op1.requires_grad() ? gradB : torch::Tensor()
        };
    }
};


// https://pytorch.org/cppdocs/api/function_namespaceat_1a9279dd932c0f6bebf3807a34ccbe364c.htm
// This function dispatches to the cuTENSOR GPU implementation for CUDA tensors, or the built-in einsum for others.
torch::Tensor einsum(
    std::string einstr,
    torch::Tensor op0,
    torch::Tensor op1
) {
    if (op0.is_cuda()) {
        return EinsumFunction::apply(einstr, op0, op1);
    }
    else {
        return torch::einsum(einstr, {op0, op1});
    }
}

static auto registry = torch::RegisterOperators().op("_opt_einsum_fx::einsum", &einsum);
