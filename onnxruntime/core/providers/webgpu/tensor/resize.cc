// resize.cc
#include "resize.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

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
    throw std::invalid_argument("Unsupported CoordinateTransformMode: " + mode_str);
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
    throw std::invalid_argument("Unsupported KeepAspectRatioPolicy: " + policy_str);
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
    throw std::invalid_argument("Unsupported Mode: " + mode_str);
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
    throw std::invalid_argument("Unsupported NearestMode: " + mode_str);
  }
}

// Function to parse Resize attributes from a map
ResizeAttributes ParseResizeAttributes(const OpKernelInfo& info) {
  ResizeAttributes attrs;

  // Parse antialias
  attrs.antialias = info.GetAttrOrDefault<int>("antialias", 0);

  // Parse axes
  attrs.axes = info.GetAttrsOrDefault<int64_t>("axes");

  // Parse coordinateTransformMode
  attrs.coordinateTransformMode = ParseCoordinateTransformMode(info.GetAttrOrDefault<std::string>("coordinateTransformMode", "half_pixel"));

  // Parse cubicCoeffA
  attrs.cubicCoeffA = info.GetAttrOrDefault<float>("cubicCoeffA", -0.75f);

  // Parse excludeOutside
  attrs.excludeOutside = info.GetAttrOrDefault<bool>("excludeOutside", false);

  // Parse extrapolationValue
  attrs.extrapolationValue = info.GetAttrOrDefault<float>("extrapolationValue", 0.0f);

  // Parse keepAspectRatioPolicy
  attrs.keepAspectRatioPolicy = ParseKeepAspectRatioPolicy(info.GetAttrOrDefault<std::string>("keepAspectRatioPolicy", "stretch"));

  // Parse mode
  attrs.mode = ParseMode(info.GetAttrOrDefault<std::string>("mode", "nearest"));

  // Parse nearestMode
  attrs.nearestMode = ParseNearestMode(info.GetAttrOrDefault<std::string>("nearestMode", "simple"));

  return attrs;
}

// Helper function to validate scales
void Resize::ValidateScales(const std::vector<float>& scales, const ResizeAttributes& attributes) const {
  // All scale values must be positive
  for (const auto& scale : scales) {
    if (scale <= 0.0f) {
      throw std::invalid_argument("Resize requires scales input values to be positive");
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
        throw std::invalid_argument("For linear mode, Resize requires scales to be 2D, 3D, 4D with either two outermost or one innermost and one outermost scale values equal to 1, or 5D with two outermost scale values equal to 1");
      }
    } else if (attributes.mode == Mode::Cubic) {
      bool valid = (scales.size() == 2 ||
                    (scales.size() == 4 && scales[0] == 1.0f && scales[1] == 1.0f) ||
                    (scales.size() == 4 && scales[0] == 1.0f && scales[3] == 1.0f));
      if (!valid) {
        throw std::invalid_argument("Resize requires scales input size to be 2 or 4 for cubic mode");
      }
    }
  }
}

// Helper function to update scales based on axes
std::vector<float> Resize::UpdateScales(const std::vector<float>& scales, const std::vector<int64_t>& axes, int rank) const {
  // Validate axes
  for (const auto& axis : axes) {
    if (axis < 0 || axis >= rank) {
      throw std::invalid_argument("Resize requires axes input values to be positive and less than rank");
    }
  }

  // Initialize newScales with 1.0
  std::vector<float> newScales(rank, 1.0f);
  for (size_t i = 0; i < axes.size(); ++i) {
    newScales[axes[i]] = scales[i];
  }
  return newScales;
}

// Helper function to validate inputs
void Resize::ValidateInputs(const ComputeContext& context,
                            const ResizeAttributes& attributes,
                            int opsetVersion,
                            std::vector<float>& scales,
                            std::vector<int64_t>& sizes,
                            std::vector<float>& roi) const {
  // Determine input indices based on opsetVersion
  int roiInputIndex, scalesInputIndex, sizesInputIndex;
  if (opsetVersion > 10) {
    roiInputIndex = 1;
    scalesInputIndex = 2;
    sizesInputIndex = 3;
  } else {
    roiInputIndex = -1;
    scalesInputIndex = (context.InputCount() > 1) ? 1 : -1;
    sizesInputIndex = -1;
  }

  const auto input_tensor = context.Input(0);
  int rank = static_cast<int>(input_tensor->Shape().GetDims().size());

  // Handle roi input
  if (roiInputIndex >= 0 && static_cast<size_t>(context.InputCount()) > static_cast<size_t>(roiInputIndex) &&
      context.Input(roiInputIndex)->Shape().GetDims().size() > 0) {
    const auto& roi_tensor = context.Input(roiInputIndex);
    auto roi_data = roi_tensor->Data<float>();
    size_t roi_size = roi_tensor->Shape().Size();
    roi.assign(roi_data, roi_data + roi_size);
  } else if (attributes.coordinateTransformMode == CoordinateTransformMode::TfCropAndResize) {
    throw std::invalid_argument("Resize requires RoI input to be specified when coordinateTransformMode is tfCropAndResize");
  }

  // Handle scales input
  if (scalesInputIndex >= 0 && static_cast<size_t>(context.InputCount()) > static_cast<size_t>(scalesInputIndex) &&
      context.Input(scalesInputIndex)->Shape().GetDims().size() == 1 &&
      context.Input(scalesInputIndex)->Shape().GetDims()[0] > 0) {
    const auto& scales_tensor = context.Input(scalesInputIndex);
    const float* scales_data = scales_tensor->Data<float>();
    size_t scales_size = scales_tensor->Shape().Size();
    scales.assign(scales_data, scales_data + scales_size);

    if (!scales.empty() &&
        scales.size() != static_cast<size_t>(rank) &&
        opsetVersion >= 18 &&
        scales.size() != attributes.axes.size()) {
      throw std::invalid_argument("Resize requires scales input size to be same as input rank or axes size for opset 18 and up");
    }

    ValidateScales(scales, attributes);

    if (!attributes.axes.empty()) {
      scales = UpdateScales(scales, attributes.axes, rank);
    }
  }

  // Handle sizes input
  if (sizesInputIndex >= 0 && static_cast<size_t>(context.InputCount()) > static_cast<size_t>(sizesInputIndex) &&
      context.Input(sizesInputIndex)->Shape().GetDims().size() == 1 &&
      context.Input(sizesInputIndex)->Shape().GetDims()[0] > 0) {
    const auto& sizes_tensor = context.Input(sizesInputIndex);
    const int64_t* sizes_data = sizes_tensor->Data<int64_t>();
    size_t sizes_size = sizes_tensor->Shape().Size();
    sizes.assign(sizes_data, sizes_data + sizes_size);

    if (!sizes.empty() &&
        sizes.size() != static_cast<size_t>(rank) &&
        opsetVersion >= 18 &&
        sizes.size() != attributes.axes.size()) {
      throw std::invalid_argument("Resize requires sizes input size to be same as input rank or axes size for opset 18 and up");
    }
  }

  // Additional checks
  if (!attributes.axes.empty()) {
    if (!scales.empty() && scales.size() != attributes.axes.size()) {
      throw std::invalid_argument("Resize requires 'scales' input size to be of axes rank when axes attributes is specified");
    }
    if (!sizes.empty() && sizes.size() != attributes.axes.size()) {
      throw std::invalid_argument("Resize requires 'sizes' input size to be of rank axes rank when axes attributes is specified");
    }
  }

  if (!scales.empty() && !sizes.empty() && scales.size() > static_cast<size_t>(input_tensor->Shape().GetDims().size())) {
    throw std::invalid_argument("Resize requires only one of scales or sizes to be specified");
  }
}

// ResizeProgram: Utility function implementations
std::string ResizeProgram::CoordinateTransformModeToWGSL(CoordinateTransformMode mode) const {
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
      throw std::invalid_argument("Unsupported CoordinateTransformMode enum");
  }
}

std::string ResizeProgram::KeepAspectRatioPolicyToWGSL(KeepAspectRatioPolicy policy) const {
  switch (policy) {
    case KeepAspectRatioPolicy::Stretch:
      return "stretch";
    case KeepAspectRatioPolicy::NotSmaller:
      return "not_smaller";
    case KeepAspectRatioPolicy::NotLarger:
      return "not_larger";
    default:
      throw std::invalid_argument("Unsupported KeepAspectRatioPolicy enum");
  }
}

std::string ResizeProgram::ModeToWGSL(Mode mode) const {
  switch (mode) {
    case Mode::Nearest:
      return "nearest";
    case Mode::Linear:
      return "linear";
    case Mode::Cubic:
      return "cubic";
    default:
      throw std::invalid_argument("Unsupported Mode enum");
  }
}

std::string ResizeProgram::NearestModeToWGSL(NearestMode mode, int opset_version) const {
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
      throw std::invalid_argument("Nearest mode 'simple' not supported for opset >= 11");
    default:
      throw std::invalid_argument("Unsupported NearestMode enum");
  }
}

// ResizeProgram: GenerateShaderCode implementation
Status ResizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Register uniforms
  const ShaderVariableHelper& output_size = shader.AddInput("ouput_size");
  const ShaderVariableHelper& scales = shader.AddInput("scales");
  const ShaderVariableHelper& roi = shader.AddInput("roi");

  // Declare input and output variables
  const ShaderVariableHelper& input = shader.AddInput("input");
  const ShaderVariableHelper& output = shader.AddOutput("output");

  // Begin shader code generation
  std::ostringstream ss;

  // Define getOriginalCoordinateFromResizedCoordinate function
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size");
  bool noScale = Inputs()[0].tensor->Shape() == Outputs()[0].tensor->Shape();
  ss << "fn getOriginalCoordinateFromResizedCoordinate(xResized: u32, xScale: f32, lengthResized: u32, lengthOriginal: u32, roiStart: f32, roiEnd: f32) -> f32 {\n";
  switch (attributes_.coordinateTransformMode) {
    case CoordinateTransformMode::Asymmetric:
      ss << "  return f32(xResized) / xScale;\n";
      break;
    case CoordinateTransformMode::PyTorchHalfPixel:
      ss << "  if (lengthResized > 1u) {\n"
         << "    return (f32(xResized) + 0.5) / xScale - 0.5;\n"
         << "  } else {\n"
         << "    return 0.0;\n"
         << "  }\n";
      break;
    case CoordinateTransformMode::TfHalfPixelForNN:
      ss << "  return (f32(xResized) + 0.5) / xScale;\n";
      break;
    case CoordinateTransformMode::AlignCorners:
      ss << "  if (lengthResized == 1u) {\n"
         << "    return 0.0;\n"
         << "  } else {\n"
         << "    let whole = f32(xResized * (lengthOriginal - 1u) / (lengthResized - 1u));\n"
         << "    let fract = f32(xResized * (lengthOriginal - 1u) % (lengthResized - 1u)) / f32(lengthResized - 1u);\n"
         << "    return whole + fract;\n"
         << "  }\n";
      break;
    case CoordinateTransformMode::TfCropAndResize:
      ss << "  if (lengthResized > 1u) {\n"
         << "    return roiStart * f32(lengthOriginal - 1u) + (f32(xResized) * (roiEnd - roiStart) * f32(lengthOriginal - 1u)) / f32(lengthResized - 1u);\n"
         << "  } else {\n"
         << "    return 0.5 * (roiStart + roiEnd) * f32(lengthOriginal - 1u);\n"
         << "  }\n";
      break;
    case CoordinateTransformMode::HalfPixelSymmetric:
      ss << "  let outputWidth = xScale * f32(lengthResized);\n"
         << "  let adjustment = f32(lengthResized) / outputWidth;\n"
         << "  let center = f32(lengthOriginal) / 2.0;\n"
         << "  let offset = center * (1.0 - adjustment);\n"
         << "  return offset + ((f32(xResized) + 0.5) / xScale) - 0.5;\n";
      break;
    case CoordinateTransformMode::HalfPixel:
      ss << "  return ((f32(xResized) + 0.5) / xScale) - 0.5;\n";
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported CoordinateTransformMode");
  }
  ss << "}\n\n";

  // Define getNearestPixelFromOriginal function
  ss << "fn getNearestPixelFromOriginal(xOriginal: f32, isDownSample: bool) -> f32 {\n";
  switch (attributes_.nearestMode) {
    case NearestMode::RoundPreferFloor:
      ss << "  if (fract(xOriginal) == 0.5) {\n"
         << "    return floor(xOriginal);\n"
         << "  } else {\n"
         << "    return round(xOriginal);\n"
         << "  }\n";
      break;
    case NearestMode::RoundPreferCeil:
      ss << "  if (fract(xOriginal) == 0.5) {\n"
         << "    return ceil(xOriginal);\n"
         << "  } else {\n"
         << "    return round(xOriginal);\n"
         << "  }\n";
      break;
    case NearestMode::Floor:
      ss << "  return floor(xOriginal);\n";
      break;
    case NearestMode::Ceil:
      ss << "  return ceil(xOriginal);\n";
      break;
    case NearestMode::Simple:
      if (GetOpsetVersionFromCustomDataBuffer(shader.GetComputeContext()) < 11) {
        ss << "  if (isDownSample) {\n"
           << "    return ceil(xOriginal);\n"
           << "  } else {\n"
           << "    return xOriginal;\n"
           << "  }\n";
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Nearest mode 'simple' not supported for opset >= 11");
      }
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported NearestMode");
  }
  ss << "}\n\n";

  // Define helper functions for interpolation (bilinear, trilinear, bicubic)
  // For brevity, only bilinear is implemented here. Similar implementations are needed for trilinear and bicubic.

  if (attributes_.mode == Mode::Linear) {
    if (shader.GetInputShapeSize() == 2 || shader.GetInputShapeSize() == 4) {  // 2D or 4D
      // Bilinear Interpolation
      ss << "fn bilinearInterpolation(output_indices: array<u32, " << shader.GetInputShapeSize() << ">) -> f32 {\n"
         << "  let row = uniforms.roi[0] + (f32(output_indices[1]) * uniforms.scales[1]);\n"
         << "  let col = uniforms.roi[1] + (f32(output_indices[2]) * uniforms.scales[2]);\n"
         << "  let row1 = u32(floor(row));\n"
         << "  let col1 = u32(floor(col));\n"
         << "  let row2 = row1 + 1u;\n"
         << "  let col2 = col1 + 1u;\n"
         << "  let dx = row - f32(row1);\n"
         << "  let dy = col - f32(col1);\n"
         << "  let val11 = input[output_indices[0], row1, col1, output_indices[3]];\n"
         << "  let val12 = input[output_indices[0], row1, col2, output_indices[3]];\n"
         << "  let val21 = input[output_indices[0], row2, col1, output_indices[3]];\n"
         << "  let val22 = input[output_indices[0], row2, col2, output_indices[3]];\n"
         << "  return (val11 * (1.0 - dx) * (1.0 - dy)) + (val12 * (1.0 - dx) * dy) + (val21 * dx * (1.0 - dy)) + (val22 * dx * dy);\n"
         << "}\n\n";
    }
    // Implement trilinear and other interpolations similarly...
  }

  // Main function
  ss << "fn main() {\n"
     << "  if (global_idx >= uniforms.output_size) { return; }\n"
     << "  let output_indices = offsetToIndices(global_idx);\n";

  // Depending on mode, generate different interpolation logic
  std::string mode_str = ModeToWGSL(attributes_.mode);
  if (attributes_.mode == Mode::Nearest) {
    ss << "  // Nearest mode interpolation\n"
       << "  var input_indices: array<u32, " << shader.GetInputShapeSize() << ">;\n"
       << "  for (var i: u32 = 0u; i < " << shader.GetInputShapeSize() << "; i = i + 1u) {\n"
       << "    let scale = uniforms.scales[i];\n"
       << "    let roi_start = uniforms.roi[i];\n"
       << "    let roi_end = uniforms.roi[" << shader.GetInputShapeSize() << " + i];\n"
       << "    let original_coord = getOriginalCoordinateFromResizedCoordinate(output_indices[i], scale, uniforms.output_shape[i], uniforms.input_shape[i], roi_start, roi_end);\n"
       << "    if (scale == 1.0) {\n"
       << "      input_indices[i] = output_indices[i];\n"
       << "    } else {\n"
       << "      let input_coord = getNearestPixelFromOriginal(original_coord, scale < 1.0);\n"
       << "      input_indices[i] = u32(input_coord);\n"
       << "    }\n"
       << "  }\n"
       << "  // Set the output value from the input tensor\n"
       << "  output[global_idx] = input[input_indices];\n";
  } else if (attributes_.mode == Mode::Linear) {
    if (shader.GetInputShapeSize() == 2 || shader.GetInputShapeSize() == 4) {  // 2D or 4D
      ss << "  // Bilinear interpolation\n"
         << "  let interpolated_val = bilinearInterpolation(output_indices);\n"
         << "  output[global_idx] = interpolated_val;\n";
    }
    // Implement trilinear and other interpolations similarly...
  } else if (attributes_.mode == Mode::Cubic) {
    // Implement bicubic interpolation here...
    ss << "  // Bicubic interpolation not implemented yet.\n";
    // Placeholder
    ss << "  output[global_idx] = 0.0;\n";
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported Mode in shader generation");
  }

  ss << "}\n";

  // Assign the generated shader code to the ShaderHelper
  shader.SetCode(ss.str());

  return Status::OK();
}

Resize::Resize(const OpKernelInfo& info) : WebGpuKernel(info) {
  // Parse attributes
  attributes_ = ParseResizeAttributes(info);
  optset_ = info.node().SinceVersion();
}

// Resize::ComputeInternal implementation
Status Resize::ComputeInternal(ComputeContext& context) const {
  // Retrieve input tensors
  const Tensor* input_tensor = context.Input(0);
  if (input_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input tensor is null");
  }

  std::vector<const Tensor*> inputs = context.inputs;
  std::vector<float> scales;
  std::vector<int64_t> sizes;
  std::vector<float> roi;

  // Extract opset version from custom data buffer
  int opsetVersion;
  opsetVersion = opset_;

  // Validate inputs and populate scales, sizes, roi
  try {
    ValidateInputs(inputs, attributes_, opsetVersion, scales, sizes, roi);
  } catch (const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, ex.what());
  }

  // Determine output shape
  std::vector<int64_t> output_dims;
  const auto& input_dims = input_tensor->Shape().GetDims();
  if (!sizes.empty()) {
    if (!attributes_.axes.empty()) {
      output_dims = input_dims;
      for (size_t i = 0; i < attributes_.axes.size(); ++i) {
        output_dims[attributes_.axes[i]] = sizes[i];
      }
    } else {
      output_dims = sizes;
    }
  } else {
    if (scales.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Resize requires either scales or sizes.");
    } else {
      output_dims.resize(input_dims.size());
      for (size_t i = 0; i < input_dims.size(); ++i) {
        output_dims[i] = static_cast<int64_t>(std::round(input_dims[i] * scales[i]));
      }

      if (attributes_.keepAspectRatioPolicy != KeepAspectRatioPolicy::Stretch) {
        // Adjust output shape based on keepAspectRatioPolicy
        float scale_in_policy;
        if (attributes_.keepAspectRatioPolicy == KeepAspectRatioPolicy::NotLarger) {
          if (!attributes_.axes.empty()) {
            scale_in_policy = *std::min_element(attributes_.axes.begin(), attributes_.axes.end(),
                                                [&](int64_t a, int64_t b) { return scales[a] < scales[b]; });
          } else {
            scale_in_policy = *std::min_element(scales.begin(), scales.end());
          }
        } else if (attributes_.keepAspectRatioPolicy == KeepAspectRatioPolicy::NotSmaller) {
          if (!attributes_.axes.empty()) {
            scale_in_policy = *std::max_element(attributes_.axes.begin(), attributes_.axes.end(),
                                                [&](int64_t a, int64_t b) { return scales[a] > scales[b]; });
          } else {
            scale_in_policy = *std::max_element(scales.begin(), scales.end());
          }
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported KeepAspectRatioPolicy");
        }

        // Update scales to maintain aspect ratio
        std::vector<float> updated_scales = scales;
        std::fill(updated_scales.begin(), updated_scales.end(), 1.0f);

        if (!attributes_.axes.empty()) {
          for (size_t i = 0; i < attributes_.axes.size(); ++i) {
            updated_scales[attributes_.axes[i]] = scale_in_policy;
            output_dims[attributes_.axes[i]] = static_cast<int64_t>(std::round(input_dims[attributes_.axes[i]] * scale_in_policy));
          }
        } else {
          for (size_t i = 0; i < updated_scales.size(); ++i) {
            updated_scales[i] = scale_in_policy;
            output_dims[i] = static_cast<int64_t>(std::round(input_dims[i] * scale_in_policy));
          }
        }

        scales = updated_scales;
      }
    }
  }

  // Create output tensor
  TensorShape output_shape_obj(output_dims);
  Tensor* output_tensor = context.Output(0, output_shape_obj);
  if (output_tensor == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Output tensor is null");
  }

  // Initialize ResizeProgram
  ResizeProgram program;
  program.attributes_ = attributes_;

  // Create ShaderHelper
  ShaderHelper shader(program, /* program_metadata */ ProgramMetadata(), /* device */ context.device(),
                      /* limits */ context.limits(),
                      /* dispatch_group_size_x */ static_cast<uint32_t>((output_shape_obj.Size() + 63) / 64),
                      /* dispatch_group_size_y */ 1,
                      /* dispatch_group_size_z */ 1);

  ORT_RETURN_IF_ERROR(shader.Init());

  // Generate shader code
  ORT_RETURN_IF_ERROR(program.GenerateShaderCode(shader));

  // Create ProgramInfo
  ProgramInfo program_info;
  program_info.name = "Resize";
  program_info.shader_code = shader.GetCode();

  // Calculate output size
  size_t output_size = output_shape_obj.Size();

  // Set dispatch group size (assuming workgroup size of 64)
  program_info.dispatch_group = {static_cast<uint32_t>((output_size + 63) / 64), 1, 1};

  // Set uniforms
  program_info.uniforms.push_back({DataType::UINT32, {static_cast<uint32_t>(output_size)}});
  program_info.uniforms.push_back({DataType::FLOAT, scales});
  program_info.uniforms.push_back({DataType::FLOAT, roi});
  program_info.uniforms.push_back({DataType::UINT32, input_dims});
  program_info.uniforms.push_back({DataType::UINT32, output_dims});

  // Define outputs
  program_info.outputs.push_back({output_shape_obj, input_tensor->DataType()});

  // Run the shader program
  return context.RunProgram(program_info);
}

}  // namespace webgpu
}  // namespace onnxruntime
