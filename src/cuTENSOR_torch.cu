#include <vector>
#include <tuple>
#include <string>

#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cutensor.h>


#define HANDLE_ERROR(x) {
    const auto err = x;
    TORCH_CHECK( err == CUTENSOR_STATUS_SUCCESS, cutensorGetErrorString(err) );
}

static cutensorHandle_t handle = nullptr;

// Convert PyTorch dtype to CUDA dtype
inline cudaDataType_t torch_to_CUDA_dtype(caffe2::TypeMeta dtype) {
    switch (dtype) {
        case torch::kFloat32:
            return CUDA_R_32F;
        case torch::kFloat64:
            return CUDA_R_64F;
        default:
            TORCH_CHECK(False, "cannot handle provided dtype");
    }
}

// make cuTENSOR descriptor and alignment from a Tensor
// TODO device
inline std::tuple<cutensorTensorDescriptor_t, uint32_t> torch_to_cuTENSOR(torch::Tensor tensor) {
    cutensorTensorDescriptor_t desc;
    HANDLE_ERROR( cutensorInitTensorDescriptor(
        &handle,
        &desc,
        tensor.ndimension(), // num modes
        tensor.sizes(),  // extent of modes
        tensor.strides(),
        torch_to_CUDA_dtype(tensor.dtype()),
        CUTENSOR_OP_IDENTITY // the elementwise op to apply while loading
    ) );

    uint32_t alignmentRequirement;
    HANDLE_ERROR( cutensorGetAlignmentRequirement(
        &handle,
        tensor.data_ptr(), // data ptr
        &desc,
        &alignmentRequirement
    ) );

    return {desc, alignmentRequirement};
}

void contract(
    torch::Tensor A,
    int32_t modeA[],
    torch::Tensor B,
    int32_t modeB[],
    torch::Tensor C,
    int32_t modeC[],
    torch::Tensor D,
    int32_t modeD[]
) {
    if (handle == nullptr) {
        cutensorInit(&handle);
    }

    // Make descriptors from torch tensors
    cutensorTensorDescriptor_t descA, descB, descC, descD;
    uint32_t alignmentRequirementA, alignmentRequirementB, alignmentRequirementC, alignmentRequirementD;
    std::tie(descA, alignmentRequirementA) = torch_to_cuTENSOR(A);
    std::tie(descB, alignmentRequirementB) = torch_to_cuTENSOR(B);
    std::tie(descC, alignmentRequirementC) = torch_to_cuTENSOR(C);
    std::tie(descD, alignmentRequirementD) = torch_to_cuTENSOR(D);

    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR( cutensorInitContractionDescriptor(
        &handle,
        &desc,
        &descA, modeA, alignmentRequirementA,
        &descB, modeB, alignmentRequirementB,
        &descC, modeC, alignmentRequirementC,
        &descD, modeD, alignmentRequirementD,
        CUTENSOR_COMPUTE_32F // TODO
    ) );

    // Set the algorithm to use
    cutensorContractionFind_t find;
    HANDLE_ERROR( cutensorInitContractionFind(
        &handle,
        &find,
        CUTENSOR_ALGO_DEFAULT
    ) );

    // From https://docs.nvidia.com/cuda/cutensor/getting_started.html#determine-algorithm-and-workspace
    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR( cutensorContractionGetWorkspace(
        &handle,
        &desc,
        &find,
        CUTENSOR_WORKSPACE_RECOMMENDED,
        &worksize
    ) );

    // Allocate workspace
    // TODO: use pytorch to alloc?
    void *work = nullptr;
    if(worksize > 0)
    {
        if( cudaSuccess != cudaMalloc(&work, worksize) ) // TODO: This is optional!
        {
            work = nullptr;
            worksize = 0;
        }
    }

    // Run compute
    cutensorContractionPlan_t plan;
    HANDLE_ERROR( cutensorInitContractionPlan(
        &handle,
        &plan,
        &desc,
        &find,
        worksize
    ) );

    cutensorStatus_t err;

    // Execute the tensor contraction
    // TODO!
    const float32_t alpha = 1.0;
    const float32_t beta = 0.0;
    err = cutensorContraction(
        &handle,
        &plan,
        (void*)&alpha,
        A.data_ptr(),
        B.data_ptr(),
        (void*)&beta,
        C.data_ptr(),
        D.data_ptr(),
        work,
        worksize,
        0 /* stream */
    );
    cudaDeviceSynchronize(); // TODO??

    // Check for errors
    TORCH_CHECK(err == CUTENSOR_STATUS_SUCCESS, cutensorGetErrorString(err));

    // Free working memory
    if ( work ) cudaFree( work );

    return;
}