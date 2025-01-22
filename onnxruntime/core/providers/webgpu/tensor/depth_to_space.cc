#include "depth_to_space.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

// DepthToSpace operator declarations
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DepthToSpace,
    kOnnxDomain,
    11, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    DepthToSpace);

ONNX_OPERATOR_KERNEL_EX(
    DepthToSpace,
    kOnnxDomain,
    13,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    DepthToSpace);

DepthToSpace::DepthToSpace(const OpKernelInfo& info) : WebGpuKernel(info) {
  // Parse attributes
  int64_t blocksize;
  ORT_THROW_IF_ERROR(info.GetAttr("blocksize", &blocksize));
  attributes_.blocksize = blocksize;

  // Get mode attribute (default to "DCR")
  ORT_THROW_IF_ERROR(info.GetAttr("mode", &attributes_.mode));
#if 0
  attributes_.mode = info.GetAttrOrDefault("mode", std::string("DCR"));
  auto format = info.GetAttrOrDefault<std::string>("format", "NCHW");
  attributes_.nchw = (format == "NCHW");
#endif

  // Get format attribute (default to NCHW)
  std::string format = info.GetAttrOrDefault("format", std::string("NCHW"));
  attributes_.nchw = (format == "NCHW");
}

void DepthToSpace::ValidateInputs(const ComputeContext& context) const {
  // Check input count
  ORT_ENFORCE(context.InputCount() == 1, "DepthToSpace requires 1 input.");

  // Get input tensor
  const Tensor* input = context.Input<Tensor>(0);
  ORT_ENFORCE(input != nullptr, "Input tensor is null");

  // Check input dimensions
  auto input_shape = input->Shape();
  ORT_ENFORCE(input_shape.NumDimensions() == 4, "DepthToSpace requires 4D input.");

  // Validate blocksize
  const int64_t block_size_squared = attributes_.blocksize * attributes_.blocksize;
  const int64_t channels = attributes_.nchw ? input_shape[1] : input_shape[3];
  ORT_ENFORCE(channels % block_size_squared == 0,
              "Input tensor's channels dimension must be divisible by (blocksize * blocksize)");
}

void DepthToSpace::CalculateShapeAndPerm(
    const TensorShape& input_shape,
    std::vector<int64_t>& reshaped_shape,
    std::vector<size_t>& perm) const {
  const auto& dims = input_shape.GetDims();
  const int64_t n = dims[0];
  const int64_t blocksize = attributes_.blocksize;
  const bool is_dcr_mode = attributes_.mode == "DCR";

  if (attributes_.nchw) {
    const int64_t c = dims[1];
    const int64_t h = dims[2];
    const int64_t w = dims[3];

    if (is_dcr_mode) {
      reshaped_shape = {n, blocksize, blocksize, c / (blocksize * blocksize), h, w};
      perm = {0, 3, 4, 1, 5, 2};
    } else {
      reshaped_shape = {n, c / (blocksize * blocksize), blocksize, blocksize, h, w};
      perm = {0, 1, 4, 2, 5, 3};
    }
  } else {
    const int64_t h = dims[1];
    const int64_t w = dims[2];
    const int64_t c = dims[3];

    if (is_dcr_mode) {
      reshaped_shape = {n, h, w, blocksize, blocksize, c / (blocksize * blocksize)};
      perm = {0, 1, 3, 2, 4, 5};
    } else {
      reshaped_shape = {n, h, w, c / (blocksize * blocksize), blocksize, blocksize};
      perm = {0, 1, 4, 2, 5, 3};
    }
  }
}

Status DepthToSpace::ComputeInternal(ComputeContext& context) const {
  // Validate inputs
  ValidateInputs(context);

  // Get input tensor
  const Tensor* input_tensor = context.Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape();

  // Calculate reshaped shape and permutation
  std::vector<int64_t> reshaped_shape;
  std::vector<size_t> perm;
  CalculateShapeAndPerm(input_shape, reshaped_shape, perm);

  // Calculate output shape
  int components;
  if (attributes_.nchw) {
    components = 1;
  } else {
    components = 4;
  }

  // Create and setup the program
  auto program = std::make_unique<DepthToSpaceProgram>();

  // Set program attributes
  program->attributes_.blocksize = attributes_.blocksize;
  program->attributes_.is_dcr_mode = (attributes_.mode == "DCR");
  program->attributes_.is_channel_last = !attributes_.nchw;
  program->attributes_.shape = reshaped_shape;
  program->attributes_.perm = perm;
  program->attributes_.input_dims = std::vector<int64_t>(input_shape.GetDims().begin(), input_shape.GetDims().end());

  // Configure inputs/outputs
  program->AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, TensorShape(reshaped_shape), components}});
  program->AddOutput({context.Output(0, TensorShape(reshaped_shape)), ProgramTensorMetadataDependency::None});

  // Set dispatch size
  program->SetDispatchGroupSize(
      static_cast<uint32_t>((TensorShape(reshaped_shape).Size() + 63) / 64)  // Workgroup size of 64
  );

  // Add uniform variables
  program->AddUniformVariables({{static_cast<uint32_t>(TensorShape(reshaped_shape).Size())}});

  // Run the program
  return context.RunProgram(*program);
}

Status DepthToSpaceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Setup input/output variables
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);

  // Generate permutation function
  OStringStream& perm_function = shader.AdditionalImplementation();
  perm_function << "fn perm(i: " << "output_indices_t" << ") -> " << "x_indices_t" << " {\n"
                << "  var a: " << "x_indices_t" << ";\n";

  // perm's rank and shape are the same as the input tensor's rank and shape
  for (size_t i = 0; i < attributes_.perm.size(); ++i) {
    perm_function << "  " << input.IndicesSet("a", attributes_.perm[i], "i[" + std::to_string(i) + "]") << "\n";
  }
  perm_function << "  return a;\n}\n\n";

  // Generate main computation
  OStringStream& code = shader.MainFunctionBody();
  code << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size") << "\n\n"
       << "let indices = " << output.OffsetToIndices("global_idx") << ";\n"
       << "let aIndices = perm(indices);\n"
       << output.SetByOffset("global_idx", input.GetByIndices("aIndices")) << ";\n";

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
