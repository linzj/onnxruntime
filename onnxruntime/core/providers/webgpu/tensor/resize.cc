// resize.cc
#include "core/providers/webgpu/tensor/resize.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace std {
template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(const std::vector<T>& vec) const {
    size_t seed = vec.size();  // Start with the size of the vector
    for (const auto& element : vec) {
      // Combine the hash of each element with the seed
      seed ^= std::hash<T>{}(element) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
}  // namespace std

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_RESIZE_VERSIONED_KERNEL(start, end)            \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                          \
      Resize,                                                 \
      kOnnxDomain,                                            \
      start,                                                  \
      end,                                                    \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T1", WebGpuSupportedNumberTypes()) \
          .TypeConstraint("T2", WebGpuSupportedNumberTypes()) \
          .InputMemoryType(OrtMemTypeCPU, 1)                  \
          .InputMemoryType(OrtMemTypeCPU, 2)                  \
          .InputMemoryType(OrtMemTypeCPU, 3),                 \
      Resize);

#define WEBGPU_RESIZE_KERNEL(version)                         \
  ONNX_OPERATOR_KERNEL_EX(                                    \
      Resize,                                                 \
      kOnnxDomain,                                            \
      version,                                                \
      kWebGpuExecutionProvider,                               \
      (*KernelDefBuilder::Create())                           \
          .TypeConstraint("T1", WebGpuSupportedNumberTypes()) \
          .TypeConstraint("T2", WebGpuSupportedNumberTypes()) \
          .InputMemoryType(OrtMemTypeCPU, 1)                  \
          .InputMemoryType(OrtMemTypeCPU, 2)                  \
          .InputMemoryType(OrtMemTypeCPU, 3),                 \
      Resize);

WEBGPU_RESIZE_VERSIONED_KERNEL(10, 10)
WEBGPU_RESIZE_VERSIONED_KERNEL(11, 12)
WEBGPU_RESIZE_VERSIONED_KERNEL(13, 17)
WEBGPU_RESIZE_VERSIONED_KERNEL(18, 18)
WEBGPU_RESIZE_KERNEL(19)

namespace {
Printable SetChannelAndBatchIndices(
    const ShaderIndicesHelper& input,
    int channel_idx,
    int batch_idx,
    int special_dims) {
  return Printable([&](std::ostream& code) {
    if (input.Rank() > special_dims) {
      // Assuming indicesSet is a function in ShaderIndicesHelper
      code << input.IndicesSet("input_indices", channel_idx, "channel") << ";\n";
      code << input.IndicesSet("input_indices", batch_idx, "batch") << ";\n";
    }
  });
}

// Helper functions to convert strings to enums
CoordinateTransformMode ParseCoordinateTransformMode(const std::string& mode_str) {
  if (mode_str == "half_pixel") {
    return CoordinateTransformMode::HalfPixel;
  } else if (mode_str == "asymmetric") {
    return CoordinateTransformMode::Asymmetric;
  } else if (mode_str == "pytorch_half_pixel") {
    return CoordinateTransformMode::PyTorchHalfPixel;
  } else if (mode_str == "tf_half_pixel_for_nn") {
    return CoordinateTransformMode::TfHalfPixelForNN;
  } else if (mode_str == "align_corners") {
    return CoordinateTransformMode::AlignCorners;
  } else if (mode_str == "tf_crop_and_resize") {
    return CoordinateTransformMode::TfCropAndResize;
  } else if (mode_str == "half_pixel_symmetric") {
    return CoordinateTransformMode::HalfPixelSymmetric;
  } else {
    ORT_THROW("Unsupported CoordinateTransformMode: ", mode_str);
  }
}

KeepAspectRatioPolicy ParseKeepAspectRatioPolicy(const std::string& policy_str) {
  if (policy_str == "stretch") {
    return KeepAspectRatioPolicy::Stretch;
  } else if (policy_str == "not_smaller") {
    return KeepAspectRatioPolicy::NotSmaller;
  } else if (policy_str == "not_larger") {
    return KeepAspectRatioPolicy::NotLarger;
  } else {
    ORT_THROW("Unsupported KeepAspectRatioPolicy: ", policy_str);
  }
}

Mode ParseMode(const std::string& mode_str) {
  if (mode_str == "nearest") {
    return Mode::Nearest;
  } else if (mode_str == "linear") {
    return Mode::Linear;
  } else if (mode_str == "cubic") {
    return Mode::Cubic;
  } else {
    ORT_THROW("Unsupported Mode: ", mode_str);
  }
}

NearestMode ParseNearestMode(const std::string& mode_str) {
  if (mode_str == "round_prefer_floor") {
    return NearestMode::RoundPreferFloor;
  } else if (mode_str == "round_prefer_ceil") {
    return NearestMode::RoundPreferCeil;
  } else if (mode_str == "floor") {
    return NearestMode::Floor;
  } else if (mode_str == "ceil") {
    return NearestMode::Ceil;
  } else if (mode_str == "simple" || mode_str.empty()) {
    return NearestMode::Simple;
  } else {
    ORT_THROW("Unsupported NearestMode: ", mode_str);
  }
}

// Function to parse Resize attributes from a map
ResizeAttributes ParseResizeAttributes(const OpKernelInfo& info) {
  ResizeAttributes attrs;

  // Parse antialias
  attrs.antialias = info.GetAttrOrDefault("antialias", 0.0f);

  // Parse optset
  attrs.opset = info.node().SinceVersion();

  // Parse axes
  attrs.axes = info.GetAttrsOrDefault("axes");

  // Parse coordinateTransformMode
  attrs.coordinateTransformMode = ParseCoordinateTransformMode(info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel"));

  // Parse cubicCoeffA
  attrs.cubicCoeffA = info.GetAttrOrDefault<float>("cubic_coeff_a", -0.75f);

  // Parse excludeOutside
  attrs.excludeOutside = info.GetAttrOrDefault("exclude_outside", 0.0f) != 0.0f;

  // Parse extrapolationValue
  attrs.extrapolationValue = info.GetAttrOrDefault<float>("extrapolation_value", 0.0f);

  // Parse keepAspectRatioPolicy
  attrs.keepAspectRatioPolicy = ParseKeepAspectRatioPolicy(info.GetAttrOrDefault<std::string>("keep_aspect_ratio_policy", "stretch"));

  // Parse mode
  attrs.mode = ParseMode(info.GetAttrOrDefault<std::string>("mode", "nearest"));

  // Parse nearestMode
  attrs.nearestMode = ParseNearestMode(info.GetAttrOrDefault<std::string>("nearest_mode", "simple"));

  return attrs;
}

std::vector<float> UpdateScales(const std::vector<float>& scales, const gsl::span<const int64_t>& axes, int rank) {
  // Validate axes
  for (const auto& axis : axes) {
    if (axis < 0 || axis >= rank) {
      ORT_THROW("Resize requires axes input values to be positive and less than rank");
    }
  }

  // Initialize newScales with 1.0
  std::vector<float> newScales(rank, 1.0f);
  for (size_t i = 0; i < axes.size(); ++i) {
    newScales[axes[i]] = scales[i];
  }
  return newScales;
}

std::vector<float> UpdateRoI(const std::vector<float>& roi, const gsl::span<const int64_t>& axes, size_t rank) {
  std::vector<float> roi_temp(rank * 2, 0);                     // Create a vector with 'rank' 0's followed by 'rank' 1's
  std::vector<float> roi_local = roi.empty() ? roi_temp : roi;  // If roi is empty, use roiTmp, otherwise copy roi

  if (!axes.empty()) {
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t v = axes[i];
      roi_temp[v] = roi_local[i];
      roi_temp[i + rank] = roi_local[axes.size() + i];
    }
    return roi_temp;
  }

  return roi_local;
}

std::vector<int64_t> InitOutputShape(const gsl::span<const int64_t>& input_shape,
                                     const std::vector<float>& scales,
                                     const std::vector<int64_t>& sizes,
                                     const gsl::span<const int64_t>& axes) {
  std::vector<int64_t> output_shape;

  if (!sizes.empty()) {
    if (!axes.empty()) {
      // Copy inputShape into outputShape
      output_shape.insert(output_shape.end(), input_shape.begin(), input_shape.end());

      // Check that axes is not out of bounds
      if (static_cast<size_t>(*std::max_element(axes.begin(), axes.end())) > input_shape.size()) {
        ORT_THROW("axes is out of bound");
      }

      // Set values in outputShape based on axes and sizes
      for (size_t i = 0; i < axes.size(); ++i) {
        output_shape[axes[i]] = sizes[i];
      }
    } else {
      // Just copy sizes into outputShape
      output_shape = sizes;
    }
  } else {
    if (scales.empty()) {
      ORT_THROW("Resize requires either scales or sizes.");
    } else {
      // Scale inputShape using scales and round the values
      for (size_t i = 0; i < input_shape.size(); ++i) {
        output_shape.push_back(std::round(input_shape[i] * scales[i]));
      }
    }
  }

  return output_shape;
}

std::vector<int64_t> AdjustOutputShape(const gsl::span<const int64_t>& input_shape,
                                       std::vector<float>& scales,
                                       const ResizeAttributes& attributes) {
  // Define a lambda for scale in policy based on the provided `keepAspectRatioPolicy`
  float scale_in_policy = [&]() -> float {
    if (attributes.keepAspectRatioPolicy == KeepAspectRatioPolicy::NotLarger) {
      if (!attributes.axes.empty()) {
        return *std::min_element(attributes.axes.begin(), attributes.axes.end(),
                                 [&](int i, int j) { return scales[i] < scales[j]; });
      } else {
        return *std::min_element(scales.begin(), scales.end());
      }
    } else if (attributes.keepAspectRatioPolicy == KeepAspectRatioPolicy::NotSmaller) {
      if (!attributes.axes.empty()) {
        return *std::max_element(attributes.axes.begin(), attributes.axes.end(),
                                 [&](int i, int j) { return scales[i] < scales[j]; });
      } else {
        return *std::max_element(scales.begin(), scales.end());
      }
    } else {
      ORT_THROW("Keep aspect ratio policy ", ResizeProgram::KeepAspectRatioPolicyToWGSL(attributes.keepAspectRatioPolicy), " is not supported");
    }
  }();

  // Reset scales to 1.0 for all dimensions
  std::fill(scales.begin(), scales.end(), 1.0f);

  std::vector<int64_t> adjustedOutputShape(input_shape.begin(), input_shape.end());

  if (!attributes.axes.empty()) {
    // Set scale in policy for each axis in axes
    for (size_t i = 0; i < attributes.axes.size(); ++i) {
      scales[attributes.axes[i]] = scale_in_policy;
      adjustedOutputShape[attributes.axes[i]] = std::round(input_shape[attributes.axes[i]] * scales[attributes.axes[i]]);
    }
  } else {
    // Set all scales to scaleInPolicy if no axes are provided
    std::fill(scales.begin(), scales.end(), scale_in_policy);
    for (size_t i = 0; i < input_shape.size(); ++i) {
      adjustedOutputShape[i] = std::round(input_shape[i] * scales[i]);
    }
  }

  return adjustedOutputShape;
};

}  // namespace

ResizeProgram::ResizeProgram() : Program("Resize") {}

// Helper function to validate scales
void Resize::ValidateScales(const std::vector<float>& scales, const ResizeAttributes& attributes) const {
  // All scale values must be positive
  for (const auto& scale : scales) {
    if (scale <= 0.0f) {
      ORT_THROW("Resize requires scales input values to be positive");
    }
  }

  // Validate scales dimensions based on mode
  if (!scales.empty()) {
    if (attributes.mode == Mode::Linear) {
      bool valid = (scales.size() == 2 || scales.size() == 3 ||
                    (scales.size() == 4 && scales[0] == 1.0f && scales[1] == 1.0f) ||
                    (scales.size() == 4 && scales[0] == 1.0f && scales[3] == 1.0f) ||
                    (scales.size() == 5 && scales[0] == 1.0f && scales[1] == 1.0f));
      if (!valid) {
        ORT_THROW("For linear mode, Resize requires scales to be 2D, 3D, 4D with either two outermost or one innermost and one outermost scale values equal to 1, or 5D with two outermost scale values equal to 1");
      }
    } else if (attributes.mode == Mode::Cubic) {
      bool valid = (scales.size() == 2 ||
                    (scales.size() == 4 && scales[0] == 1.0f && scales[1] == 1.0f) ||
                    (scales.size() == 4 && scales[0] == 1.0f && scales[3] == 1.0f));
      if (!valid) {
        ORT_THROW("Reize requires scales input size to be 2 or 4 for cubic mode");
      }
    }
  }
}

// Helper function to update scales based on axes
// Helper function to validate inputs
void Resize::ValidateInputs(const ComputeContext& context,
                            const ResizeAttributes& attributes,
                            int opset_version,
                            std::vector<float>& scales,
                            std::vector<int64_t>& sizes,
                            std::vector<float>& roi) const {
  // Determine input indices based on opsetVersion
  int roi_input_index, scales_input_index, size_input_index;
  if (opset_version > 10) {
    roi_input_index = 1;
    scales_input_index = 2;
    size_input_index = 3;
  } else {
    roi_input_index = -1;
    scales_input_index = (context.InputCount() > 1) ? 1 : -1;
    size_input_index = -1;
  }

  const auto input_tensor = context.Input(0);
  int rank = static_cast<int>(input_tensor->Shape().GetDims().size());

  // Handle roi input
  if (roi_input_index >= 0 && static_cast<size_t>(context.InputCount()) > static_cast<size_t>(roi_input_index) &&
      context.Input(roi_input_index)->Shape().GetDims().size() > 0) {
    const auto& roi_tensor = context.Input(roi_input_index);
    auto roi_data = roi_tensor->Data<float>();
    size_t roi_size = roi_tensor->Shape().Size();
    roi.assign(roi_data, roi_data + roi_size);
  } else if (attributes.coordinateTransformMode == CoordinateTransformMode::TfCropAndResize) {
    ORT_THROW("Resize requires RoI input to be specified when coordinateTransformMode is tfCropAndResize");
  }

  // Handle scales input
  if (scales_input_index >= 0 && static_cast<size_t>(context.InputCount()) > static_cast<size_t>(scales_input_index) &&
      context.Input(scales_input_index)->Shape().GetDims().size() == 1 &&
      context.Input(scales_input_index)->Shape().GetDims()[0] > 0) {
    const auto& scales_tensor = context.Input(scales_input_index);
    const float* scales_data = scales_tensor->Data<float>();
    size_t scales_size = scales_tensor->Shape().Size();
    scales.assign(scales_data, scales_data + scales_size);

    if (!scales.empty() &&
        scales.size() != static_cast<size_t>(rank) &&
        opset_version >= 18 &&
        scales.size() != attributes.axes.size()) {
      ORT_THROW("Resize requires scales input size to be same as input rank or axes size for opset 18 and up");
    }

    ValidateScales(scales, attributes);

    if (!attributes.axes.empty()) {
      scales = UpdateScales(scales, attributes.axes, rank);
    }
  }

  // Handle sizes input
  if (size_input_index >= 0 && static_cast<size_t>(context.InputCount()) > static_cast<size_t>(size_input_index) &&
      context.Input(size_input_index)->Shape().GetDims().size() == 1 &&
      context.Input(size_input_index)->Shape().GetDims()[0] > 0) {
    const auto& sizes_tensor = context.Input(size_input_index);
    const int64_t* sizes_data = sizes_tensor->Data<int64_t>();
    size_t sizes_size = sizes_tensor->Shape().Size();
    sizes.assign(sizes_data, sizes_data + sizes_size);

    if (!sizes.empty() &&
        sizes.size() != static_cast<size_t>(rank) &&
        opset_version >= 18 &&
        sizes.size() != attributes.axes.size()) {
      ORT_THROW("Resize requires sizes input size to be same as input rank or axes size for opset 18 and up");
    }
  }

  // Additional checks
  if (!attributes.axes.empty()) {
    if (!scales.empty() && scales.size() != attributes.axes.size()) {
      ORT_THROW("Resize requires 'scales' input size to be of axes rank when axes attributes is specified");
    }
    if (!sizes.empty() && sizes.size() != attributes.axes.size()) {
      ORT_THROW("Resize requires 'sizes' input size to be of axes rank when axes attributes is specified");
    }
  }

  if (!scales.empty() && !sizes.empty() && scales.size() > static_cast<size_t>(input_tensor->Shape().GetDims().size())) {
    ORT_THROW("Resize requires only one of scales or sizes to be specified");
  }
}

// ResizeProgram: Utility function implementations
std::string ResizeProgram::CoordinateTransformModeToWGSL(CoordinateTransformMode mode) {
  switch (mode) {
    case CoordinateTransformMode::HalfPixel:
      return "half_pixel";
    case CoordinateTransformMode::Asymmetric:
      return "asymmetric";
    case CoordinateTransformMode::PyTorchHalfPixel:
      return "pytorch_half_pixel";
    case CoordinateTransformMode::TfHalfPixelForNN:
      return "tf_half_pixel_for_nn";
    case CoordinateTransformMode::AlignCorners:
      return "align_corners";
    case CoordinateTransformMode::TfCropAndResize:
      return "tf_crop_and_resize";
    case CoordinateTransformMode::HalfPixelSymmetric:
      return "half_pixel_symmetric";
    default:
      ORT_THROW("Unsupported CoordinateTransformMode enum");
  }
}

std::string ResizeProgram::KeepAspectRatioPolicyToWGSL(KeepAspectRatioPolicy policy) {
  switch (policy) {
    case KeepAspectRatioPolicy::Stretch:
      return "stretch";
    case KeepAspectRatioPolicy::NotSmaller:
      return "not_smaller";
    case KeepAspectRatioPolicy::NotLarger:
      return "not_larger";
    default:
      ORT_THROW("Unsupported KeepAspectRatioPolicy enum");
  }
}

std::string ResizeProgram::ModeToWGSL(Mode mode) {
  switch (mode) {
    case Mode::Nearest:
      return "nearest";
    case Mode::Linear:
      return "linear";
    case Mode::Cubic:
      return "cubic";
    default:
      ORT_THROW("Unsupported Mode enum");
  }
}

std::string ResizeProgram::NearestModeToWGSL(NearestMode mode, int opset_version) {
  switch (mode) {
    case NearestMode::RoundPreferFloor:
      return "round_prefer_floor";
    case NearestMode::RoundPreferCeil:
      return "round_prefer_ceil";
    case NearestMode::Floor:
      return "floor";
    case NearestMode::Ceil:
      return "ceil";
    case NearestMode::Simple:
      if (opset_version < 11) {
        return "simple";
      }
      ORT_THROW("Nearest mode 'simple' not supported for opset >= 11");
    default:
      ORT_THROW("Unsupported NearestMode enum");
  }
}

// ResizeProgram: GenerateShaderCode implementation
Status ResizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Declare input and output variables using ShaderVariableHelper with appropriate usage flags
  const ShaderVariableHelper& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const ShaderVariableHelper& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);

  // Initialize a string stream to build additional WGSL implementations
  OStringStream& additional_impl = shader.AdditionalImplementation();

  // Define getOriginalCoordinateFromResizedCoordinate function
  additional_impl << "fn getOriginalCoordinateFromResizedCoordinate(xResized: u32, xScale: f32, lengthResized: u32, "
                  << "lengthOriginal: u32, roiStart: f32, roiEnd: f32) -> f32 {\n";  // Returning f32 as per input_value_t
  switch (attributes_.coordinateTransformMode) {
    case CoordinateTransformMode::Asymmetric:
      additional_impl << "  return f32(xResized) / f32(xScale);\n";
      break;
    case CoordinateTransformMode::PyTorchHalfPixel:
      additional_impl << "  if (lengthResized > 1) {\n"
                      << "    return (f32(xResized) + 0.5) / f32(xScale) - 0.5;\n"
                      << "  } else {\n"
                      << "    return 0.0;\n"
                      << "  }\n";
      break;
    case CoordinateTransformMode::TfHalfPixelForNN:
      additional_impl << "  return (f32(xResized) + 0.5) / f32(xScale);\n";
      break;
    case CoordinateTransformMode::AlignCorners:
      additional_impl << "  if (lengthResized == 1) {\n"
                      << "    return 0.0;\n"
                      << "  } else {\n"
                      << "    let whole = f32(xResized * (lengthOriginal - 1) / (lengthResized - 1));\n"
                      << "    let fract = f32(xResized * (lengthOriginal - 1) % (lengthResized - 1)) / f32(lengthResized - 1);\n"
                      << "    return whole + fract;\n"
                      << "  }\n";
      break;
    case CoordinateTransformMode::TfCropAndResize:
      additional_impl << "  if (lengthResized > 1) {\n"
                      << "    return f32(roiStart) * f32(lengthOriginal - 1) +\n"
                      << "           (f32(xResized) * f32(roiEnd - roiStart) * f32(lengthOriginal - 1)) /\n"
                      << "           f32(lengthResized - 1);\n"
                      << "  } else {\n"
                      << "    return 0.5 * f32(roiStart + roiEnd) * f32(lengthOriginal - 1);\n"
                      << "  }\n";
      break;
    case CoordinateTransformMode::HalfPixelSymmetric:
      additional_impl << "  let outputWidth = f32(xScale) * f32(lengthResized);\n"
                      << "  let adjustment = f32(lengthResized) / outputWidth;\n"
                      << "  let center = f32(lengthOriginal) / 2.0;\n"
                      << "  let offset = center * (1.0 - adjustment);\n"
                      << "  return offset + ((f32(xResized) + 0.5) / f32(xScale)) - 0.5;\n";
      break;
    case CoordinateTransformMode::HalfPixel:
      additional_impl << "  return ((f32(xResized) + 0.5) / f32(xScale)) - 0.5;\n";
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported CoordinateTransformMode in shader generation.");
  }
  additional_impl << "}\n\n";

  // Define getNearestPixelFromOriginal function
  additional_impl << "fn getNearestPixelFromOriginal(xOriginal: f32, isDownSample: bool) -> f32 {\n";
  switch (attributes_.nearestMode) {
    case NearestMode::RoundPreferFloor:
      additional_impl << "  if (fract(xOriginal) == 0.5) {\n"
                      << "    return floor(xOriginal);\n"  // Corrected to 'floor' as per user feedback
                      << "  } else {\n"
                      << "    return round(xOriginal);\n"
                      << "  }\n";
      break;
    case NearestMode::RoundPreferCeil:
      additional_impl << "  if (fract(xOriginal) == 0.5) {\n"
                      << "    return ceil(xOriginal);\n"
                      << "  } else {\n"
                      << "    return round(xOriginal);\n"
                      << "  }\n";
      break;
    case NearestMode::Floor:
      additional_impl << "  return floor(xOriginal);\n";
      break;
    case NearestMode::Ceil:
      additional_impl << "  return ceil(xOriginal);\n";
      break;
    case NearestMode::Simple:
      if (attributes_.opset < 11) {
        additional_impl << "  if (isDownSample) {\n"
                        << "    return ceil(xOriginal);\n"
                        << "  } else {\n"
                        << "    return xOriginal;\n"
                        << "  }\n";
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Nearest mode 'simple' not supported for opset >= 11.");
      }
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported NearestMode in shader generation.");
  }
  additional_impl << "}\n\n";

  // Define calculateOriginalIndicesFromOutputIndices function
  size_t output_shape_length = attributes_.output_shape.NumDimensions();
  size_t input_shape_length = attributes_.input_shape.NumDimensions();
  bool is_f16 = attributes_.input_is_fp16;
  bool use_extrapolation = attributes_.coordinateTransformMode == CoordinateTransformMode::TfCropAndResize;
  additional_impl << "fn calculateOriginalIndicesFromOutputIndices(output_indices: output_indices_t) -> array<output_value_t, "
                  << output_shape_length << "> {\n"
                  << "  var original_indices: array<output_value_t, " << output_shape_length << ">;\n"
                  << "  for (var i: u32 = 0u; i < " << output_shape_length << "u; i = i + 1u) {\n"
                  << "    let output_index = " << output.IndicesGet("output_indices", "i") << ";\n"
                  << "    let scale = " << GetElementAt("uniforms.scales", "i", attributes_.scales.size(), is_f16) << ";\n"
                  << "    let roi_low = " << GetElementAt("uniforms.roi", "i", attributes_.roi.size(), is_f16) << ";\n"
                  << "    let roi_hi = " << GetElementAt("uniforms.roi", "i + " + std::to_string(input_shape_length), attributes_.roi.max_size(), is_f16) << ";\n"
                  << "    if (scale == 1.0) {\n"
                  << "      original_indices[i] = output_value_t(output_index);\n"
                  << "    } else {\n"
                  << "      let input_shape_i = " << GetElementAt("uniforms.input_shape", "i", input_shape_length, is_f16) << ";\n"
                  << "      let output_shape_i = " << GetElementAt("uniforms.output_shape", "i", output_shape_length, is_f16) << ";\n"
                  << "      original_indices[i] = getOriginalCoordinateFromResizedCoordinate(output_index, scale, output_shape_i,\n"
                  << "                                                                       input_shape_i, roi_low, roi_hi);\n"
                  << "    }\n"
                  << "  }\n"
                  << "  return original_indices;\n"
                  << "}\n";

  // Define calculateInputIndicesFromOutputIndices function
  additional_impl << "fn calculateInputIndicesFromOutputIndices(output_indices: output_indices_t) -> input_indices_t {\n"
                  << "  var input_indices: input_indices_t;\n"
                  << "  for (var i: u32 = 0u; i < " << output_shape_length << "; i++) {\n"
                  << "    var output_index = " << output.IndicesGet("output_indices", "i") << ";\n"
                  << "    var input_index: u32;\n"
                  << "    var scale = " << GetElementAt("uniforms.scales", "i", attributes_.scales.size()) << ";\n"
                  << "    if (scale == 1.0) {" << "\n"
                  << "      input_index = output_index;\n"
                  << "    } else {\n"
                  << "      var roi_low = " << GetElementAt("uniforms.roi", "i", attributes_.roi.size()) << ";\n"
                  << "      var roi_hi = " << GetElementAt("uniforms.roi", MakeString("i + ", input_shape_length), attributes_.roi.size()) << ";\n"
                  << "      var input_shape_i = " << GetElementAt("uniforms.input_shape", "i", input_shape_length) << ";\n"
                  << "      var output_shape_i = " << GetElementAt("uniforms.output_shape", "i", output_shape_length) << ";\n"
                  << "      var original_idx = getOriginalCoordinateFromResizedCoordinate(output_index, scale, output_shape_i," << "\n"
                  << "                                                                    input_shape_i, roi_low, roi_hi);\n"
                  << "      if (!" << (use_extrapolation ? "true" : "false") << " || (original_idx >= 0 && original_idx < output_value_t(input_shape_i))) {\n"
                  << "        if (original_idx < 0) {\n"
                  << "          input_index = 0;\n"
                  << "        } else if (original_idx > output_value_t(input_shape_i - 1)) {\n"
                  << "          input_index = input_shape_i - 1;\n"
                  << "        } else {\n"
                  << "          input_index = u32(getNearestPixelFromOriginal(original_idx, scale < 1));\n"
                  << "        }\n"
                  << "      } else {\n"
                  << "        input_index = u32(original_idx);\n"
                  << "      }\n"
                  << "    }\n"
                  << "    " << input.IndicesSet("input_indices", "i", "input_index") << ";\n"
                  << "  }\n"
                  << "  return input_indices;\n"
                  << "}\n\n";

  // Define checkInputIndices function
  additional_impl << "fn checkInputIndices(input_indices: input_indices_t) -> bool {\n"
                  << "  for (var i: u32 = 0u; i < " << output_shape_length << "u; i = i + 1u) {\n"
                  << "  var input_index = " << input.IndicesGet("input_indices", "i") << ";\n"
                  << "  if (input_index < 0 || input_index >= " << GetElementAt("uniforms.input_shape", "i", input_shape_length) << ") {\n"
                  << "    return false;\n"
                  << "  }\n"
                  << "}\n"
                  << "return true;\n"
                  << "}\n\n";

  // Define interpolation functions based on mode
  if (attributes_.mode == Mode::Linear) {
    if (input_shape_length == 2 || input_shape_length == 4) {
      constexpr const bool is_nchw = true;
      auto [batchIdx, heightIdx, widthIdx, channelIdx] =
          input_shape_length == 2 ? std::make_tuple(-1, 0, 1, -1) : is_nchw ? std::make_tuple(0, 2, 3, 1)
                                                                            : std::make_tuple(0, 1, 2, 3);
      // Bilinear interpolation
      additional_impl
          << "    fn getInputValue(batch: u32, channel: u32, row: u32, col: u32) -> input_value_t {\n"
          << "      var input_indices: input_indices_t;\n"
          << "      " << input.IndicesSet("input_indices", heightIdx, MakeString("max(0, min(row, ", attributes_.input_shape[heightIdx] - 1, "))")) << "\n"
          << "      " << input.IndicesSet("input_indices", widthIdx, MakeString("max(0, min(col, ", attributes_.input_shape[widthIdx] - 1, "))")) << "\n"
          << "      " << SetChannelAndBatchIndices(input, channelIdx, batchIdx, 2) << "\n"
          << "      return " << input.GetByIndices("input_indices") << ";\n"
          << "    }\n"
          << "\n"
          << "    fn bilinearInterpolation(output_indices: output_indices_t) -> input_value_t {\n"
          << "      var originalIndices = calculateOriginalIndicesFromOutputIndices(output_indices);\n"
          << "      var row:input_value_t = originalIndices[" << heightIdx << "];\n"
          << "      var col:input_value_t = originalIndices[" << widthIdx << "];\n";

      if (use_extrapolation) {
        additional_impl
            << "      if (row < 0 || row > (" << attributes_.input_shape[heightIdx] << " - 1) || col < 0 || col > (" << attributes_.input_shape[widthIdx] << " - 1)) {\n"
            << "        return " << attributes_.extrapolationValue << ";\n"
            << "      }\n";
      }
      additional_impl << "      row = max(0, min(row, " << attributes_.input_shape[heightIdx] << " - 1));\n"
                      << "      col = max(0, min(col, " << attributes_.input_shape[widthIdx] << " - 1));\n"
                      << "      var row1: u32 = u32(row);\n"
                      << "      var col1: u32 = u32(col);\n"
                      << "      var row2: u32 = u32(row + 1);\n"
                      << "      var col2: u32 = u32(col + 1);\n"
                      << "      var channel: u32 = " << (input_shape_length > 2 ? MakeString("u32(originalIndices[", channelIdx, "])") : "0") << ";\n"
                      << "      var batch: u32 =  " << (input_shape_length > 2 ? MakeString("u32(originalIndices[", batchIdx, "])") : "0") << ";\n"
                      << "      var x11: input_value_t = getInputValue(batch, channel, row1, col1);\n"
                      << "      var x12: input_value_t = getInputValue(batch, channel, row1, col2);\n"
                      << "      var x21: input_value_t = getInputValue(batch, channel, row2, col1);\n"
                      << "      var x22: input_value_t = getInputValue(batch, channel, row2, col2);\n"
                      << "      var dx1: input_value_t = abs(row - input_value_t(row1));\n"
                      << "      var dx2: input_value_t = abs(input_value_t(row2) - row);\n"
                      << "      var dy1: input_value_t = abs(col - input_value_t(col1));\n"
                      << "      var dy2: input_value_t = abs(input_value_t(col2) - col);\n"
                      << "      if (row1 == row2) {\n"
                      << "        dx1 = 0.5;\n"
                      << "        dx2 = 0.5;\n"
                      << "      }\n"
                      << "      if (col1 == col2) {\n"
                      << "        dy1 = 0.5;\n"
                      << "        dy2 = 0.5;\n"
                      << "      }\n"
                      << "      return (x11 * dx2 * dy2 + x12 * dx2 * dy1 + x21 * dx1 * dy2 + x22 * dx1 * dy1);\n"
                      << "    }\n";
    } else if (input_shape_length == 3 || input_shape_length == 5) {
      constexpr const bool is_nchw = true;
      auto [batchIdx, depthIdx, heightIdx, widthIdx, channelIdx] =
          input_shape_length == 3 ? std::make_tuple(-1, 0, 1, 2, -1) : is_nchw ? std::make_tuple(0, 2, 3, 4, 1)
                                                                               : std::make_tuple(0, 1, 2, 3, 4);
      // Trilinear interpolation
      additional_impl
          << "    fn getInputValue(batch: u32, channel: u32, depth:u32, height: u32, width: u32) -> input_value_t {\n"
          << "      var input_indices: input_indices_t;\n"
          << "      " << input.IndicesSet("input_indices", depthIdx, MakeString("max(0, min(depth, ", attributes_.input_shape[depthIdx] - 1, "))")) << "\n"
          << "      " << input.IndicesSet("input_indices", heightIdx, MakeString("")) << ";\n"
          << "      " << input.IndicesSet("input_indices", widthIdx, MakeString("max(0, min(width, ", attributes_.input_shape[widthIdx] - 1, "))")) << "\n"
          << "      " << SetChannelAndBatchIndices(input, channelIdx, batchIdx, 3) << "\n"
          << "      return " << input.GetByIndices("input_indices") << ";\n"
          << "    }\n"
          << "\n"
          << "    fn trilinearInterpolation(output_indices: output_indices_t) -> input_value_t {\n"
          << "      var originalIndices = calculateOriginalIndicesFromOutputIndices(output_indices);\n"
          << "      var depth:input_value_t = originalIndices[" << depthIdx << "];\n"
          << "      var height:input_value_t = originalIndices[" << heightIdx << "];\n"
          << "      var width:input_value_t = originalIndices[" << widthIdx << "];\n";
      if (use_extrapolation) {
        additional_impl << "      if (depth < 0 || depth > " << (attributes_.input_shape[depthIdx] - 1) << " || height < 0 || height > " << (attributes_.input_shape[heightIdx] - 1) << " || width < 0 || width > " << (attributes_.input_shape[widthIdx] - 1) << ") {\n"
                        << "        return " << attributes_.extrapolationValue << ";\n"
                        << "     }\n";
      }
      additional_impl
          << "      depth = max(0, min(depth, " << (attributes_.input_shape[depthIdx] - 1) << "));\n"
          << "      height = max(0, min(height, " << (attributes_.input_shape[heightIdx] - 1) << "));\n"
          << "      width = max(0, min(width, " << (attributes_.input_shape[widthIdx] - 1) << "));\n"
          << "      var depth1: u32 = u32(depth);\n"
          << "      var height1: u32 = u32(height);\n"
          << "      var width1: u32 = u32(width);\n"
          << "      var depth2: u32 = u32(depth + 1);\n"
          << "      var height2: u32 = u32(height + 1);\n"
          << "      var width2: u32 = u32(width + 1);\n"
          << "      var channel: u32 = ";
      if (input_shape_length > 3) {
        additional_impl << "u32(originalIndices[" << channelIdx << "])";
      } else {
        additional_impl << "0";
      }
      additional_impl << ";\n"
                      << "      var batch: u32 =  ";
      if (input_shape_length > 3) {
        additional_impl << "u32(originalIndices[" << batchIdx << "])";
      } else {
        additional_impl << "0";
      }
      additional_impl << ";\n"
                      << "      var x111: input_value_t = getInputValue(batch, channel, depth1, height1, width1);\n"
                      << "      var x112: input_value_t = getInputValue(batch, channel, depth1, height1, width2);\n"
                      << "      var x121: input_value_t = getInputValue(batch, channel, depth1, height2, width1);\n"
                      << "      var x122: input_value_t = getInputValue(batch, channel, depth1, height2, width2);\n"
                      << "      var x211: input_value_t = getInputValue(batch, channel, depth2, height1, width1);\n"
                      << "      var x212: input_value_t = getInputValue(batch, channel, depth2, height1, width2);\n"
                      << "      var x221: input_value_t = getInputValue(batch, channel, depth2, height2, width1);\n"
                      << "      var x222: input_value_t = getInputValue(batch, channel, depth2, height2, width2);\n"
                      << "      var dx1: input_value_t = abs(depth - input_value_t(depth1));\n"
                      << "      var dx2: input_value_t = abs(input_value_t(depth2) - depth);\n"
                      << "      var dy1: input_value_t = abs(height - input_value_t(height1));\n"
                      << "      var dy2: input_value_t = abs(input_value_t(height2) - height);\n"
                      << "      var dz1: input_value_t = abs(width - input_value_t(width1));\n"
                      << "      var dz2: input_value_t = abs(input_value_t(width2) - width);\n"
                      << "      if (depth1 == depth2) {\n"
                      << "        dx1 = 0.5;\n"
                      << "        dx2 = 0.5;\n"
                      << "      }\n"
                      << "      if (height1 == height2) {\n"
                      << "        dy1 = 0.5;\n"
                      << "        dy2 = 0.5;\n"
                      << "      }\n"
                      << "      if (width1 == width2) {\n"
                      << "        dz1 = 0.5;\n"
                      << "        dz2 = 0.5;\n"
                      << "      }\n"
                      << "      return (x111 * dx2 * dy2 * dz2 + x112 * dx2 * dy2 * dz1 + x121 * dx2 * dy1 *dz2 + x122 * dx2 * dy1 * dz1 +\n"
                      << "              x211 * dx1 * dy2 * dz2 + x212 * dx1 * dy2 * dz1 + x221 * dx1 * dy1 *dz2 + x222 * dx1 * dy1 * dz1);\n"
                      << "    }";
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Linear mode only supports input dims 2, 3, 4 and 5 are supported in linear mode.");
    }
  } else if (attributes_.mode == Mode::Cubic) {
    const bool is_2d = input_shape_length == 2;
    constexpr const bool is_nchw = true;
    int heightIdx, widthIdx;
    std::tie(heightIdx, widthIdx) = is_2d ? std::make_tuple(0, 1) : is_nchw ? std::make_tuple(2, 3)
                                                                            : std::make_tuple(1, 2);
    // Define the lambda for creating cubic interpolation functions
    auto createCubicInterpolationFunction = [&](std::ostream& func, int idx) {
      std::string direction = (idx == heightIdx) ? "row" : "col";

      func << "fn " << direction << "CubicInterpolation(input_indices: input_indices_t, output_indices: output_indices_t) -> input_value_t {\n"
           << "  var output_index = " << output.IndicesGet("output_indices", idx) << ";\n"
           << "  var originalIdx: input_value_t = getOriginalCoordinateFromResizedCoordinate(output_index, "
           << attributes_.scales[idx] << ", " << attributes_.output_shape[idx] << ", " << attributes_.input_shape[idx] << ", "
           << attributes_.roi[idx] << ", " << (attributes_.roi[idx] + input_shape_length) << ");\n"
           << "  var fractOriginalIdx: input_value_t = originalIdx - floor(originalIdx);\n"
           << "  var coefs = getCubicInterpolationCoefs(fractOriginalIdx);\n\n"
           << "  if (" << (use_extrapolation ? "true" : "false") << " && (originalIdx < 0.0 || originalIdx > ("
           << (attributes_.input_shape[idx] - 1) << ".0))) {\n"
           << "    return " << attributes_.extrapolationValue << ";\n"
           << "  }\n"
           << "  var data: array<input_value_t, 4> = array<input_value_t, 4>(0.0, 0.0, 0.0, 0.0);\n"
           << "  for (var i: i32 = -1; i < 3; i = i + 1) {\n"
           << "    var " << direction << ": input_value_t = originalIdx + input_value_t(i);\n"
           << "    if (" << direction << " < 0.0 || " << direction << " >= " << (attributes_.input_shape[idx] - 1) << ".0) {\n";

      if (attributes_.excludeOutside) {
        func << "      coefs[i + 1] = 0.0;\n"
             << "      continue;\n";
      } else if (use_extrapolation) {
        func << "      return " << attributes_.extrapolationValue << ";\n";
      } else {
        func << "      " << direction << " = max(0.0, min(" << direction << ", " << (attributes_.input_shape[idx] - 1) << ".0));\n";
      }

      func << "    }\n"
           << "    var input_indices_copy: input_indices_t = input_indices;\n"
           << "    " << input.IndicesSet("input_indices_copy", idx, "u32(" + direction + ")") << ";\n"
           << "    data[i + 1] = "
           << ((idx == heightIdx) ? input.GetByIndices("input_indices_copy") : "rowCubicInterpolation(input_indices_copy, output_indices)") << ";\n"
           << "  }\n"
           << "  return cubicInterpolation1D(data, coefs);\n"
           << "}\n\n";
    };
    // Append cubic interpolation functions for height and width indices
    createCubicInterpolationFunction(additional_impl, heightIdx);
    createCubicInterpolationFunction(additional_impl, widthIdx);
    // Define getCubicInterpolationCoefs function
    additional_impl << "fn getCubicInterpolationCoefs(s: input_value_t) -> array<input_value_t, 4> {\n"
                    << "  var absS = abs(s);\n"
                    << "  var coeffs: array<input_value_t, 4> = array<input_value_t, 4>(0.0, 0.0, 0.0, 0.0);\n"
                    << "  var oneMinusAbsS: input_value_t = 1.0 - absS;\n"
                    << "  var twoMinusAbsS: input_value_t = 2.0 - absS;\n"
                    << "  var onePlusAbsS: input_value_t = 1.0 + absS;\n"
                    << "  coeffs[0] = ((" << attributes_.cubicCoeffA << " * onePlusAbsS - 5.0 * " << attributes_.cubicCoeffA
                    << ") * onePlusAbsS + 8.0 * " << attributes_.cubicCoeffA
                    << ") * onePlusAbsS - 4.0 * " << attributes_.cubicCoeffA << ";\n"
                    << "  coeffs[1] = ((" << attributes_.cubicCoeffA << " + 2.0) * absS - (" << attributes_.cubicCoeffA
                    << " + 3.0)) * absS * absS + 1.0;\n"
                    << "  coeffs[2] = ((" << attributes_.cubicCoeffA << " + 2.0) * oneMinusAbsS - (" << attributes_.cubicCoeffA
                    << " + 3.0)) * oneMinusAbsS * oneMinusAbsS + 1.0;\n"
                    << "  coeffs[3] = ((" << attributes_.cubicCoeffA << " * twoMinusAbsS - 5.0 * " << attributes_.cubicCoeffA
                    << ") * twoMinusAbsS + 8.0 * " << attributes_.cubicCoeffA
                    << ") * twoMinusAbsS - 4.0 * " << attributes_.cubicCoeffA << ";\n"
                    << "  return coeffs;\n"
                    << "}\n\n";

    // Define cubicInterpolation1D function
    additional_impl << "fn cubicInterpolation1D(x: array<input_value_t, 4>, coefs: array<input_value_t, 4>) -> input_value_t {\n"
                    << "  var coefsSum: input_value_t = coefs[0] + coefs[1] + coefs[2] + coefs[3];\n"
                    << "  return (x[0] * coefs[0] + x[1] * coefs[1] + x[2] * coefs[2] + x[3] * coefs[3]) / coefsSum;\n"
                    << "}\n\n";

    // Define bicubicInterpolation function
    additional_impl << "fn bicubicInterpolation(output_indices: output_indices_t) -> input_value_t {\n"
                    << "  var input_indices: input_indices_t = output_indices;\n"
                    << "  return colCubicInterpolation(input_indices, output_indices);\n"
                    << "}\n\n";
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported interpolation mode.");
  }

  // Start building the main shader function body
  OStringStream& main_body = shader.MainFunctionBody();

  // Inject guard against out-of-bounds workgroup sizes
  main_body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size") << "\n";

  // Determine if scaling is required
  bool no_scale = attributes_.input_shape == attributes_.output_shape;

  if (no_scale) {
    // No scaling; direct copy
    main_body << "output[global_idx] = input[global_idx];\n";
  } else {
    // Scaling required
    main_body << "  let output_indices = " << output.OffsetToIndices("global_idx") << ";\n"
              << "  var input_indices: input_indices_t;\n";

    if (attributes_.mode == Mode::Nearest) {
      main_body << "input_indices = calculateInputIndicesFromOutputIndices(output_indices);\n"
                << "  if (checkInputIndices(input_indices)) {\n"
                << "    output[global_idx] = " << input.GetByIndices("input_indices") << ";\n"
                << "  } else {\n"
                << "    output[global_idx] = " << attributes_.extrapolationValue << ";\n"
                << "  }\n";
    } else if (attributes_.mode == Mode::Linear) {
      // Bilinear interpolation
      std::string selected_function = (input_shape_length == 2 || input_shape_length == 4 ? "bilinearInterpolation" : "trilinearInterpolation");
      main_body << "  output[global_idx] = " << selected_function << "(output_indices);\n";
    } else if (attributes_.mode == Mode::Cubic) {
      main_body << "  output[global_idx] = bicubicInterpolation(output_indices);\n";
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported resize mode.");
    }
  }

  return Status::OK();
}

Resize::Resize(const OpKernelInfo& info) : WebGpuKernel(info) {
  // Parse attributes
  attributes_ = ParseResizeAttributes(info);
}

// Resize::ComputeInternal implementation
Status Resize::ComputeInternal(ComputeContext& context) const {
  // Retrieve input tensors
  const Tensor* input_tensor = context.Input(0);
  if (input_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor is null");
  }

  std::vector<float> scales;
  std::vector<int64_t> sizes;
  std::vector<float> roi;
  std::vector<int64_t> output_shape;

  // Extract opset version from custom data buffer
  int opsetVersion;
  opsetVersion = attributes_.opset;

  // Validate inputs and populate scales, sizes, roi
  try {
    ValidateInputs(context, attributes_, opsetVersion, scales, sizes, roi);
    // Determine output shape using the provided lambdas
    const auto& input_dims = input_tensor->Shape().GetDims();

    // Initialize the ROI using the updateRoI lambda
    roi = UpdateRoI(attributes_.roi, attributes_.axes, input_dims.size());

    // Initialize the output shape using the initOutputShape lambda
    output_shape = InitOutputShape(input_dims, scales, sizes, attributes_.axes);

    if (attributes_.scales.empty()) {
      // Calculate scales if not provided
      scales.resize(input_dims.size());
      for (size_t i = 0; i < input_dims.size(); ++i) {
        scales[i] = (input_dims[i] == 0 ? 1.0f : static_cast<float>(output_shape[i]) / input_dims[i]);
      }

      if (attributes_.keepAspectRatioPolicy != KeepAspectRatioPolicy::Stretch) {
        // Adjust output shape based on keepAspectRatioPolicy
        output_shape = AdjustOutputShape(input_dims, scales, attributes_);
      }
    }
  } catch (const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, ex.what());
  }

  // Create output tensor
  TensorShape output_shape_obj(output_shape);
  Tensor* output_tensor = context.Output(0, output_shape_obj);
  if (output_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Output tensor is null");
  }

  // Calculate output size
  size_t output_size = output_shape_obj.Size();

  // Initialize ResizeProgram
  ResizeProgram program;
  program.attributes_ = attributes_;
  program.attributes_.output_shape = output_shape;
  program.attributes_.input_shape = input_tensor->Shape();
  program.attributes_.scales = scales;
  program.attributes_.roi = roi;
  program.attributes_.input_is_fp16 = input_tensor->IsDataType<MLFloat16>();
  program.AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize(static_cast<uint32_t>((output_size + 63) / 64) /* workgroup size */, 1, 1)
      .CacheHint(std::to_string(attributes_.opset) +
                 "|" + std::to_string(std::hash<std::vector<float>>{}(scales)) +
                 "|" + std::to_string(std::hash<std::vector<int64_t>>{}(sizes)))
      .AddUniformVariables({{static_cast<uint32_t>(output_size)}, {std::move(scales)}, {std::move(roi)}});

  // Run the shader program
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
