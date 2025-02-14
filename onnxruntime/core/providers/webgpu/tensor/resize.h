// resize.h
#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"

#include <vector>
#include <string>

namespace onnxruntime {
namespace webgpu {

// Enumerations corresponding to CoordinateTransformMode
enum class CoordinateTransformMode {
  HalfPixel,
  Asymmetric,
  PyTorchHalfPixel,
  TfHalfPixelForNN,
  AlignCorners,
  TfCropAndResize,
  HalfPixelSymmetric
};

// Enumerations corresponding to KeepAspectRatioPolicy
enum class KeepAspectRatioPolicy {
  Stretch,
  NotSmaller,
  NotLarger
};

// Enumerations corresponding to Mode
enum class Mode {
  Nearest,
  Linear,
  Cubic
};

// Enumerations corresponding to NearestMode
enum class NearestMode {
  RoundPreferFloor,
  RoundPreferCeil,
  Floor,
  Ceil,
  Simple
};

// Structure to hold Resize attributes
struct ResizeAttributes {
  int antialias;
  int opset;
  TensorShapeVector axes;
  std::vector<float> roi;
  std::vector<float> scales;
  CoordinateTransformMode coordinateTransformMode;
  float cubicCoeffA;
  bool excludeOutside;
  float extrapolationValue;
  KeepAspectRatioPolicy keepAspectRatioPolicy;
  Mode mode;
  NearestMode nearestMode;
  TensorShape output_shape;
  TensorShape input_shape;
  bool input_is_fp16;
};

// Resize Program Class
class ResizeProgram : public Program<ResizeProgram> {
 public:
  explicit ResizeProgram();

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  // Attributes
  ResizeAttributes attributes_;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"scales", ProgramUniformVariableDataType::Float32},
      {"roi", ProgramUniformVariableDataType::Float32});

  // Helper methods for shader code generation
  static std::string CoordinateTransformModeToWGSL(CoordinateTransformMode mode);
  static std::string KeepAspectRatioPolicyToWGSL(KeepAspectRatioPolicy policy);
  static std::string ModeToWGSL(Mode mode);
  static std::string NearestModeToWGSL(NearestMode mode, int opset_version);
};

// Resize Kernel Class
class Resize : public WebGpuKernel {
 public:
  Resize(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  ResizeAttributes attributes_;

  // Helper functions
  void ValidateInputs(const ComputeContext& context,
                      const ResizeAttributes& attributes,
                      int opsetVersion,
                      std::vector<float>& scales,
                      std::vector<int64_t>& sizes,
                      std::vector<float>& roi) const;
  void ValidateScales(const std::vector<float>& scales, const ResizeAttributes& attributes) const;
};

}  // namespace webgpu
}  // namespace onnxruntime
