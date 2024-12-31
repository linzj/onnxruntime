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
  std::vector<int64_t> axes;
  CoordinateTransformMode coordinateTransformMode;
  float cubicCoeffA;
  bool excludeOutside;
  float extrapolationValue;
  KeepAspectRatioPolicy keepAspectRatioPolicy;
  Mode mode;
  NearestMode nearestMode;
};

// Resize Program Class
class ResizeProgram : public Program<ResizeProgram> {
 public:
  ResizeProgram() : Program("Resize", ProgramMetadata()) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  // Attributes
  ResizeAttributes attributes_;

 private:
  // Helper methods for shader code generation
  std::string CoordinateTransformModeToWGSL(CoordinateTransformMode mode) const;
  std::string KeepAspectRatioPolicyToWGSL(KeepAspectRatioPolicy policy) const;
  std::string ModeToWGSL(Mode mode) const;
  std::string NearestModeToWGSL(NearestMode mode, int opset_version) const;

  // Additional helper functions can be added here
};

// Resize Kernel Class
class Resize : public WebGpuKernel {
 public:
  Resize(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  ResizeAttributes attributes_;
  int opset_;

  // Helper functions
  void ValidateInputs(const ComputeContext& context,
                      const ResizeAttributes& attributes,
                      int opsetVersion,
                      std::vector<float>& scales,
                      std::vector<int64_t>& sizes,
                      std::vector<float>& roi) const;
  std::vector<float> UpdateScales(const std::vector<float>& scales, const std::vector<int64_t>& axes, int rank) const;
  void ValidateScales(const std::vector<float>& scales, const ResizeAttributes& attributes) const;
};

}  // namespace webgpu
}  // namespace onnxruntime
