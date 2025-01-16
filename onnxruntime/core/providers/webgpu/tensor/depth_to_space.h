#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include <vector>
#include <string>

namespace onnxruntime {
namespace webgpu {

struct DepthToSpaceAttributes {
  int64_t blocksize;
  std::string mode;  // "DCR" or "CRD"
  bool nchw;         // true for NCHW, false for NHWC
};

class DepthToSpaceProgram : public Program<DepthToSpaceProgram> {
 public:
  explicit DepthToSpaceProgram() : Program("DepthToSpaceProgram") {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  struct Attributes {
    int64_t blocksize;
    bool is_dcr_mode;                 // true for DCR mode, false for CRD mode
    bool is_channel_last;             // true for NHWC, false for NCHW
    std::vector<int64_t> shape;       // Shape after reshape but before permutation
    std::vector<size_t> perm;         // Permutation indices
    std::vector<int64_t> input_dims;  // Original input dimensions
  };

  Attributes attributes_;

  // Define uniform variables
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({
      "output_size",
      ProgramUniformVariableDataType::Uint32,
  });
};

class DepthToSpace : public WebGpuKernel {
 public:
  explicit DepthToSpace(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  DepthToSpaceAttributes attributes_;

  void ValidateInputs(const ComputeContext& context) const;
  void CalculateShapeAndPerm(
      const TensorShape& input_shape,
      std::vector<int64_t>& reshaped_shape,
      std::vector<size_t>& perm) const;
};

}  // namespace webgpu
}  // namespace onnxruntime
