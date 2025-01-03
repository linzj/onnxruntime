#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include <vector>
#include <string>
#include <optional>

namespace onnxruntime {
namespace webgpu {

enum class InternalActivationKind {
  kRelu,
  kSigmoid,
  kTanh,
  kHardSigmoid,
  kClip,
  kLeakyRelu,
  kUndef,
};

enum class AutoPadKind : uint8_t {
  kNotSet,
  kValid,
  kSameUpper,
  kSameLower,
};

struct InternalActivationAttributes {
  InternalActivationKind activationAttributes = InternalActivationKind::kUndef;
  std::optional<float> clipMin, clipMax, alpha, beta;
};

// Struct to hold convolution attributes
struct ConvAttributes : InternalActivationAttributes {
  AutoPadKind autoPad;
  TensorShapeVector dilations;
  bool nchw;  // NHWC or NCHW
  int64_t group;
  TensorShapeVector kernelShape;
  TensorShapeVector pads;
  TensorShapeVector strides;
  mutable std::optional<Tensor*> wT;
  bool wIsConst;
};

// Conv1D2DProgram class for Conv1D and Conv2D
class Conv1D2DProgram : public Program<Conv1D2DProgram> {
 public:
  explicit Conv1D2DProgram();

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  ConvAttributes attributes_;

  // Define uniform variables specific to Conv1D/2D
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"dim_a_outer", ProgramUniformVariableDataType::Int32},
      {"dim_b_outer", ProgramUniformVariableDataType::Int32},
      {"dim_inner", ProgramUniformVariableDataType::Int32},
      {"pad", ProgramUniformVariableDataType::Int32},
      {"stride", ProgramUniformVariableDataType::Int32},
      {"dilation", ProgramUniformVariableDataType::Int32});

  // Helper functions
  static std::string ConvAttributesToWGSL(const ConvAttributes& attributes);
};

// Conv3DProgram class for Conv3D
class Conv3DProgram : public Program<Conv3DProgram> {
 public:
  explicit Conv3DProgram();

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  ConvAttributes attributes_;

  // Define uniform variables specific to Conv3D
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"filter_dims", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32});

  // Helper functions
  static std::string ConvAttributesToWGSL(const ConvAttributes& attributes);
};

// Conv Kernel class for Conv operations
class Conv : public WebGpuKernel {
 public:
  Conv(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  ConvAttributes attributes_;

  // Helper functions
  void ValidateInputs(const ComputeContext& context, const ConvAttributes& attributes) const;
  void CalculateOutputShape(const ComputeContext& context, const ConvAttributes& attributes, std::vector<int64_t>& outputShape) const;
  Status HandleConv1D(ComputeContext& context, const ConvAttributes& attributes) const;
  Status HandleConv2D(ComputeContext& context, const ConvAttributes& adjusted_attributes) const;
  Status HandleConv3D(ComputeContext& context, const ConvAttributes& attributes) const;
};

}  // namespace webgpu
}  // namespace onnxruntime
