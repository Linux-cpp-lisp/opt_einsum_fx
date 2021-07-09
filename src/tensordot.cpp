#include <vector>
#include <string>
#include <assert.h>

#include <torch/extension.h>

using namespace torch::autograd;


// actual computation using cuTENSOR
torch::Tensor tensordot_fw(
    torch::Tensor self,
    torch::Tensor other,
    at::IntArrayRef dims_self,
    at::IntArrayRef dims_other
) {
    return at::tensordot(self, other, dims_self, dims_other);
}


class TensordotFunction : public torch::autograd::Function<TensordotFunction> {
    public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor self,
        torch::Tensor other,
        at::IntArrayRef dims_self,
        at::IntArrayRef dims_other
    ) {
        ctx->save_for_backward({self, other});
        ctx->saved_data["dims_self"] = dims_self;
        ctx->saved_data["dims_other"] = dims_other;
        return tensordot_fw(self, other, dims_self, dims_other);
    }

    static tensor_list backward(
        AutogradContext *ctx,
        tensor_list grad_outputs
    ) {
        // see https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py#L468
        return {grad_outputs[0], grad_outputs[0]};
    }
};

// https://pytorch.org/cppdocs/api/function_namespaceat_1a9279dd932c0f6bebf3807a34ccbe364c.html
torch::Tensor tensordot(
    torch::Tensor self,
    torch::Tensor other,
    at::IntArrayRef dims_self,
    at::IntArrayRef dims_other
) {
    return TensordotFunction::apply(self, other, dims_self, dims_other);
}

static auto registry = torch::RegisterOperators().op("_opt_einsum_fx::tensordot", &tensordot);
