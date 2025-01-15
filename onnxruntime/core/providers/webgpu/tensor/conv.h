#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include <vector>
#include <string>
#include <optional>

namespace onnxruntime {
namespace webgpu {
class ShaderIndicesHelper;
class ShaderVariableHelper;

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
  std::vector<uint32_t> dilations;
  bool nchw;  // NHWC or NCHW
  int64_t group;
  TensorShapeVector kernelShape;
  TensorShapeVector pads;
  TensorShapeVector strides;
  mutable std::optional<Tensor*> wT;
  bool wIsConst;
};

// Add after the TransposeShared program class:
class GroupedConvVectorizeProgram : public Program<GroupedConvVectorizeProgram> {
 public:
  explicit GroupedConvVectorizeProgram();

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  struct Attributes {
    ConvAttributes convAttributes;
    uint32_t components;
    uint32_t output_number;
    uint32_t x_number;
    std::vector<uint32_t> x_shape;
    std::vector<uint32_t> w_shape;
    std::vector<uint32_t> output_shape;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    bool has_bias;
  };

  Attributes attributes_;

  // Define uniform variables
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Int32},
      {"pads", ProgramUniformVariableDataType::Int32},
      {"clipMax", ProgramUniformVariableDataType::Float32},
      {"clipMin", ProgramUniformVariableDataType::Float32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32});
};

class GroupedConvProgram : public Program<GroupedConvProgram> {
 public:
  explicit GroupedConvProgram() : Program("GroupedConvProgram") {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  struct Attributes {
    ConvAttributes convAttributes;
    uint32_t components;
    uint32_t output_channels_per_group;
    bool has_bias;
    bool is_channel_last;
    std::vector<uint32_t> x_shape;
    std::vector<uint32_t> w_shape;
    std::vector<uint32_t> output_shape;
  };

  Attributes attributes_;

  // Define uniform variables
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"output_channels_per_group", ProgramUniformVariableDataType::Uint32},
      {"clipMax", ProgramUniformVariableDataType::Float32},
      {"clipMin", ProgramUniformVariableDataType::Float32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32});
};

class NaiveMatmulProgram : public Program<NaiveMatmulProgram> {
 public:
  explicit NaiveMatmulProgram() : Program("NaiveMatmulProgram") {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  struct Attributes {
    uint32_t components;
    uint32_t a_components;
    uint32_t output_number;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t batchSize;
    uint32_t outputShapeSize;
    bool has_bias;
    bool isChannelsLast;
    InternalActivationAttributes activationAttributes;
  };

  Attributes attributes_;

  // Define uniform variables
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"M", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32}, );
};

class MatmulProgram : public Program<MatmulProgram> {
 public:
  explicit MatmulProgram() : Program("MatmulProgram") {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  struct Attributes {
    uint32_t components;
    uint32_t batch_size;
    uint32_t dim_a_outer;
    uint32_t dim_inner;
    uint32_t dim_b_outer;
    bool has_bias;
    bool is_channels_last;
    InternalActivationAttributes activationAttributes;
    std::vector<uint32_t> outer_dims;
    std::vector<uint32_t> a_shape;
    std::vector<uint32_t> b_shape;
    std::array<uint32_t, 3> elements_per_thread;  // Add this
    std::array<uint32_t, 3> workgroup_size;       // Add this
  };

  Attributes attributes_;

  // Define uniform variables
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"dim_a_outer", ProgramUniformVariableDataType::Int32},
      {"dim_b_outer", ProgramUniformVariableDataType::Int32},
      {"dim_inner", ProgramUniformVariableDataType::Int32},
      {"clipMax", ProgramUniformVariableDataType::Float32},
      {"clipMin", ProgramUniformVariableDataType::Float32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32});

 private:
  std::string GenerateMatMulReadWriteFnSource(const std::vector<const ShaderVariableHelper*>& variables) const;
};

class Conv2DMatMulProgram : public Program<Conv2DMatMulProgram> {
 public:
  explicit Conv2DMatMulProgram() : Program("Conv2DMatMulProgram") {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  struct Attributes {
    uint32_t components;
    bool has_bias;
    bool is_channels_last;
    InternalActivationAttributes activationAttributes;
    std::vector<uint32_t> input_shape;
    std::vector<uint32_t> weight_shape;
    std::vector<uint32_t> output_shape;
    std::array<uint32_t, 3> elements_per_thread;  // [4,4,1] or [4,1,1]
    std::array<uint32_t, 3> workgroup_size;       // [8,8,1]
    uint32_t in_channels;
    uint32_t dim_a_outer;
    uint32_t dim_b_outer;
    uint32_t dim_inner;
  };

  Attributes attributes_;

  // Define uniform variables
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"dim_a_outer", ProgramUniformVariableDataType::Int32},
      {"dim_b_outer", ProgramUniformVariableDataType::Int32},
      {"dim_inner", ProgramUniformVariableDataType::Int32},
      {"pad", ProgramUniformVariableDataType::Int32},
      {"stride", ProgramUniformVariableDataType::Int32},
      {"dilation", ProgramUniformVariableDataType::Int32},
      {"clipMax", ProgramUniformVariableDataType::Float32},
      {"clipMin", ProgramUniformVariableDataType::Float32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32});

 private:
  std::string GenerateConv2DCommonSnippet(bool is_channels_last,
                                          bool fit_a_outer,
                                          bool fit_b_outer,
                                          bool fit_inner,
                                          bool add_bias,
                                          const InternalActivationAttributes&,
                                          const std::string& data_type,
                                          uint32_t inner_element_size_x,
                                          uint32_t inner_element_size_w,
                                          uint32_t inner_element_size) const;

  std::string GetXSnippet(uint32_t inner_element_size, const std::string& data_type) const;
  std::string GetWSnippet(uint32_t inner_element_size) const;
  std::string GenerateTypeSnippet(uint32_t size, const std::string& data_type) const;
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
  void CalculateOutputShape(const ComputeContext& context, const ConvAttributes& attributes, TensorShapeVector& outputShape) const;
  Status HandleConv1D(ComputeContext& context, const ConvAttributes& attributes) const;
  Status HandleConv2D(ComputeContext& context, const ConvAttributes& adjusted_attributes) const;
  Status HandleConv3D(ComputeContext& context, const ConvAttributes& attributes) const;

  // Adjust convolution attributes based on input tensors
  ConvAttributes GetAdjustedConvAttributes(const ConvAttributes& attributes, const ComputeContext& context) const;

  // Adjust padding values based on autoPad
  void AdjustPadsBasedOnAutoPad(
      const TensorShapeVector& inputDims,
      const TensorShapeVector& strides,
      const TensorShapeVector& dilations,
      const TensorShapeVector& kernelShape,
      TensorShapeVector& pads,
      bool isChannelsLast,
      AutoPadKind autoPad) const;

  // Helper function to adjust padding for a specific dimension
  int64_t AdjustPadAndReturnShape(
      int64_t inputSize,
      int64_t stride,
      int64_t dilation,
      int64_t kernelSize,
      TensorShapeVector& pads,
      size_t padBeginIndex,
      size_t padEndIndex,
      AutoPadKind autoPad) const;
};

}  // namespace webgpu
}  // namespace onnxruntime
