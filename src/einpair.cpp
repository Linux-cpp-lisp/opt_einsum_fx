#include <vector>
#include <tuple>
#include <string>
#include <assert.h>

#include <torch/extension.h>

using namespace torch::autograd;

// Forward declaration from CUDA code
void contract(
    torch::Tensor A,
    int32_t modeA[],
    torch::Tensor B,
    int32_t modeB[],
    torch::Tensor C,
    int32_t modeC[],
    torch::Tensor D,
    int32_t modeD[]
);

// TODO: do two tensor einsum instead, is fully general
// TODO: can cuTENSOR handle that?

class EinpairFunction : public torch::autograd::Function<EinpairFunction> {
    public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        std::string einstr,
        torch::Tensor op0,
        torch::Tensor op1,
    ) {
        TORCH_CHECK(op0.is_cuda(), "both operands must be on a CUDA device");
        TORCH_CHECK(op0.device() == op1.device(), "both operands must be on the same device");
        
        // Parse einstr
        std::vector<int32_t> subs0;
        std::vector<int32_t> subs1;
        std::vector<int32_t> subs_out;
        auto iter = einstr.begin();
        for (; *iter != ','; iter++) {
            TORCH_CHECK(std::isalpha(*iter), "invalid letter in einpair string");
            subs0.push_back(*iter);
        }
        // Consume the comma
        TORCH_CHECK(*iter == ',', "comma missing");
        iter++;
        // Fill subs1
        for (; *iter != '-'; iter++) {
            TORCH_CHECK(std::isalpha(*iter), "invalid letter in einpair string");
            subs1.push_back(*iter);
        }
        // Consume ->
        TORCH_CHECK(*iter == '-', "-> missing or wrong");
        TORCH_CHECK(*(iter++) == '>', "-> missing or wrong");
        iter++;
        // Fill output
        for (; *iter != '-'; iter++) {
            TORCH_CHECK(std::isalpha(*iter), "invalid letter in einpair string");
            subs1.push_back(*iter);
        }

        // - Make and save for backward -
        // TODO
        std::string bw_einstr0;
        std::string bw_einstr1;
        ctx->save_for_backward({op0, op1});
        ctx->saved_data["bw_einstr0"] = bw_einstr0;
        ctx->saved_data["bw_einstr1"] = bw_einstr1;
        
        // Allocate output
        torch::Tensor out;

        // - Do actual computation -
        void contract(
            op0,
            subs0.data(),
            op1,
            subs1.data(),
            out,
            ,
            torch::Tensor D,
            int32_t modeD[]
        );
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
torch::Tensor einpair(
    std::string einstr,
    torch::Tensor op0,
    torch::Tensor op1,
) {
    if (op0.is_cuda()) {
        return EinpairFunction::apply(einstr, op0, op1);
    }
    else {
        return torch::einsum(einstr, op0, op1);
    }
}

static auto registry = torch::RegisterOperators().op("_opt_einsum_fx::einpair", &einpair);
