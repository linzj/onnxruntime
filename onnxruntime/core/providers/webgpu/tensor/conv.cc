// conv.cc
#include "conv.h"
#include "core/providers/webgpu/shader_helper.h"
#include <cmath>
#include <string>

namespace onnxruntime {
namespace webgpu {

namespace {
Status ParseInternalActivationAttributes(const OpKernelInfo& info, InternalActivationAttributes& attrib) {
  Status status;
  std::string activation_str;
  constexpr const float kMIN_CLIP = -3.4028234663852886e38;
  constexpr const float kMAX_CLIP = 3.4028234663852886e38;

  status = info.GetAttr<std::string>("activation", &activation_str);
  ORT_RETURN_IF_ERROR(status);

  std::vector<float> activation_params;
  if (activation_str == "Relu") {
    attrib.activationAttributes = InternalActivationKind::kRelu;
  } else if (activation_str == "Sigmoid") {
    attrib.activationAttributes = InternalActivationKind::kSigmoid;
  } else if (activation_str == "Tanh") {
    attrib.activationAttributes = InternalActivationKind::kTanh;
  } else if (activation_str == "HardSigmoid") {
    attrib.activationAttributes = InternalActivationKind::kHardSigmoid;
    activation_params = info.GetAttrsOrDefault<float>("activation_params", {0.2f, 0.5f});
    attrib.alpha = activation_params[0];
    attrib.beta = activation_params[0];
  } else if (activation_str == "Clip") {
    attrib.activationAttributes = InternalActivationKind::kClip;
    activation_params = info.GetAttrsOrDefault<float>("activation_params", {kMIN_CLIP, kMAX_CLIP});
    attrib.clipMin = activation_params[0];
    attrib.clipMax = activation_params[1];
  } else if (activation_str == "LeakyRelu") {
    attrib.activationAttributes = InternalActivationKind::kLeakyRelu;
    activation_params = info.GetAttrsOrDefault<float>("activation_params", {0.01});
    attrib.alpha = activation_params[0];
  } else {
    attrib.activationAttributes = InternalActivationKind::kUndef;  // Default to Clip
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Undefined activation kind string");
  }
  return Status();
}

ConvAttributes ParseConvAttributes(const OpKernelInfo& info) {
  ConvAttributes convAttrs;

  // Parse autoPad (default to "NOTSET" if not specified)
  convAttrs.autoPad = static_cast<AutoPadKind>(static_cast<uint8_t>(info.GetAttrOrDefault<float>("autoPad", 0.0f)));

  // Parse dilations
  convAttrs.dilations = info.GetAttrsOrDefault("dilations", {});

  // Parse kernelShape
  convAttrs.kernelShape = info.GetAttrsOrDefault("kernelShape", {});

  // Parse pads
  convAttrs.pads = info.GetAttrsOrDefault("pads", {});

  // Parse strides
  convAttrs.strides = info.GetAttrsOrDefault("strides", {});

  // Parse group (default to 1 for standard convolution)
  convAttrs.group = info.GetAttrOrDefault<float>("group", 1.0f);

  // Parse nchw (default to true for NCHW format)
  auto format = info.GetAttrOrDefault<std::string>("format", "NCHW");
  convAttrs.nchw = format == "NCHW";

  // Parse activation function attributes (if available)
  auto status = ParseInternalActivationAttributes(info, convAttrs);
  (void)status;

  return convAttrs;
}

// The program creator
std::unique_ptr<details::ProgramWrapper> CreateTransposeProgram(
    const Tensor* inputTensor, const std::vector<int64_t>& permAttr) {
  ORT_THROW("CreateTransposeProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateGroupedConvVectorizeProgram(
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const std::vector<int64_t>& outputShape,
    const std::function<std::vector<int64_t>(const std::vector<int64_t>&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateGroupedConvVectorizeProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateGroupedConvProgram(
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const std::vector<int64_t>& outputShape,
    const std::function<std::vector<int64_t>(const std::vector<int64_t>&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateGroupedConvProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateNaiveMatmulProgram(
    const std::vector<const Tensor*>& inputs,
    const InternalActivationAttributes& activationAttributes,
    const std::vector<int64_t>& outputShape,
    const std::vector<int64_t>& reshapedOutputShape,
    bool isChannelsLast,
    const std::function<std::vector<int64_t>(const std::vector<int64_t>&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateNaiveMatmulProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateMatmulProgram(
    const std::vector<const Tensor*>& inputs,
    const InternalActivationAttributes& activationAttributes,
    const std::vector<int64_t>& outputShape,
    const std::vector<int64_t>& reshapedOutputShape,
    bool isChannelsLast,
    const std::function<std::vector<int64_t>(const std::vector<int64_t>&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateMatmulProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateConv2DMatMulProgram(
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const std::vector<int64_t>& outputShape,
    int64_t dimAOuter,
    int64_t dimBOuter,
    int64_t dimInner,
    bool hasBias,
    bool sequentialAccessByThreads,
    const std::function<std::vector<int64_t>(const std::vector<int64_t>&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateConv2DMatMulProgram is not implemented yet.");
}

}  // namespace

// Conv1D2DProgram Implementation

Conv1D2DProgram::Conv1D2DProgram() : Program("Conv1D2DProgram") {
  // Initialize default values or other necessary setup if required
}

Status Conv1D2DProgram::GenerateShaderCode(ShaderHelper& sh) const {
  // Implement shader code generation logic for Conv1D and Conv2D
  // This will include defining the shader source, handling uniforms, and utilizing helper functions
  // Example:
  // sh.AddFunction(...);
  // sh.AddMainFunction(...);

  // Placeholder implementation:
  return Status::OK();
}

std::string Conv1D2DProgram::ConvAttributesToWGSL(const ConvAttributes& attributes) {
  // Convert ConvAttributes to WGSL shader code strings, like padding, dilation, etc.
  // Example:
  // std::ostringstream wgsl_code;
  // wgsl_code << "let stride = vec2<i32>(" << attributes.strides[0] << ", " << attributes.strides[1] << ");\n";
  // ... (additional conversions)
  // return wgsl_code.str();

  // Placeholder implementation:
  return "";
}

// Conv3DProgram Implementation

Conv3DProgram::Conv3DProgram() : Program("Conv3DProgram") {
  // Initialize default values or other necessary setup if required
}

Status Conv3DProgram::GenerateShaderCode(ShaderHelper& sh) const {
  // Implement shader code generation logic for Conv3D
  // This will include defining the shader source, handling uniforms, and utilizing helper functions
  // Example:
  // sh.AddFunction(...);
  // sh.AddMainFunction(...);

  // Placeholder implementation:
  return Status::OK();
}

std::string Conv3DProgram::ConvAttributesToWGSL(const ConvAttributes& attributes) {
  // Convert ConvAttributes to WGSL shader code strings specific to Conv3D
  // Example:
  // std::ostringstream wgsl_code;
  // wgsl_code << "let stride = vec3<i32>(" << attributes.strides[0] << ", " << attributes.strides[1] << ", " << attributes.strides[2] << ");\n";
  // ... (additional conversions)
  // return wgsl_code.str();

  // Placeholder implementation:
  return "";
}

// Conv Kernel Implementation

Conv::Conv(const OpKernelInfo& info) : WebGpuKernel(info) {
  // Initialize attributes from the OpKernelInfo
  // Assuming there's a method to parse attributes from OpKernelInfo
  // Example:
  // attributes_ = ParseConvAttributes(info);
  // For the purpose of this example, we'll assume it's already set or handled elsewhere
  attributes_ = ParseConvAttributes(info);
}

Status Conv::ComputeInternal(ComputeContext& context) const {
  // Get the input tensor and its number of dimensions
  const Tensor* input_data = context.Input(0);
  int64_t input_dims = input_data->Shape().NumDimensions();
  // Dispatch based on the number of dimensions of the input
  if (input_dims == 3) {
    // 1D Convolution (assuming input shape is [batch_size, channels, width])
    return HandleConv1D(context, attributes_);
  } else if (input_dims == 5) {
    // 3D Convolution (assuming input shape is [batch_size, channels, depth, height, width])
    return HandleConv3D(context, attributes_);
  } else if (input_dims == 4) {
    // 2D Convolution (assuming input shape is [batch_size, channels, height, width])
    // Adjust attributes if necessary (for example, for NHWC/NCHW handling)
    ConvAttributes adjusted_attributes = attributes_;
    if (adjusted_attributes.kernelShape.empty()) {
      // Adjust kernelShape or other attributes if needed for Conv2D
      // (you can do any adjustments you need here)
    }
    return HandleConv2D(context, adjusted_attributes);
  } else {
    // Unsupported input shape for convolution
    ORT_THROW("Unsupported input shape: Convolution supports 3D, 4D, or 5D inputs only.");
  }

  return Status::OK();
}

void Conv::ValidateInputs(const ComputeContext& context, const ConvAttributes& attributes) const {
  // Get the input tensors
  int input_count = context.InputCount();

  // Conv requires 2 or 3 inputs (data, weights, and optional bias)
  if (input_count != 2 && input_count != 3) {
    ORT_THROW("Conv requires 2 or 3 inputs, but found ", input_count);
  }

  // Access input and weights tensors
  const Tensor* input_data = context.Input(0);
  const Tensor* weights = context.Input(1);

  if (!input_data || !weights) {
    ORT_THROW("Input data or weights tensor is missing");
  }

  // Check the dimensionality of the input tensor
  if (input_data->Shape().NumDimensions() > 5) {
    ORT_THROW("Input tensor has more than 5 dimensions, which is not supported");
  }

  // Check that the number of dimensions of input data matches the number of dimensions of weights
  if (input_data->Shape().NumDimensions() != weights->Shape().NumDimensions()) {
    ORT_THROW("Filter does not have the same number of dimensions as input");
  }

  // Check if the data channel and filter input channel match
  int64_t data_channel = (attributes.nchw) ? input_data->Shape()[1] : input_data->Shape()[(input_data->Shape().NumDimensions() - 1)];
  int64_t filter_in_channel = weights->Shape()[1] * attributes.group;
  if (data_channel != filter_in_channel) {
    ORT_THROW("FILTER_IN_CHANNEL should be equal to DATA_CHANNEL");
  }

  // If bias is provided (3 inputs), it should be 1D and the number of elements should match the number of feature maps
  if (input_count == 3) {
    const Tensor* bias = context.Input(2);
    if (bias->Shape().NumDimensions() != 1 || bias->Shape()[0] != weights->Shape()[0]) {
      ORT_THROW("Invalid bias: Bias should be 1D and match the number of feature maps in the filter");
    }
  }

  // Determine the spatial rank (dims excluding batch and channels)
  size_t spatial_rank = input_data->Shape().NumDimensions() - 2;

  // Check the dimensions of dilations, strides, and pads
  if (attributes.dilations.size() != spatial_rank) {
    ORT_THROW("Dilations should be ", spatial_rank, "D");
  }

  if (attributes.strides.size() != spatial_rank) {
    ORT_THROW("Strides should be ", spatial_rank, "D");
  }

  if (attributes.pads.size() != spatial_rank * 2) {
    ORT_THROW("Pads should be ", spatial_rank * 2, "D");
  }

  // If kernelShape is specified, its size must be 2 less than the dimensions of the weights tensor
  if (!attributes.kernelShape.empty() && attributes.kernelShape.size() != weights->Shape().NumDimensions() - 2) {
    ORT_THROW("Invalid kernel shape: The kernelShape length should be ", weights->Shape().NumDimensions() - 2);
  }
}

void Conv::CalculateOutputShape(const ComputeContext& context, const ConvAttributes& attributes, std::vector<int64_t>& outputShape) const {
}

// HandleConv1D - Convolution for 1D inputs (e.g., [batch_size, channels, width])
Status Conv::HandleConv1D(ComputeContext& context, const ConvAttributes& attributes) const {
  // Implement the convolution logic for 1D inputs
  // You would need to write the code that actually performs the convolution
  // For example, setting up the shader or calling the relevant WebGPU method
  // that handles 1D convolutions.
  // This is just a placeholder for actual logic.
  return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Conv1D handling not implemented yet.");
}

// HandleConv3D - Convolution for 3D inputs (e.g., [batch_size, channels, depth, height, width])
Status Conv::HandleConv3D(ComputeContext& context, const ConvAttributes& attributes) const {
  // Implement the convolution logic for 3D inputs
  // Similar to Conv1D, you'd need to handle the 3D tensor data and perform
  // the convolution operation, possibly dispatching it to a WebGPU shader.
  // This is a placeholder for the actual 3D logic.
  return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Conv3D handling not implemented yet.");
}

// HandleConv2D - Convolution for 2D inputs (e.g., [batch_size, channels, height, width])
Status Conv::HandleConv2D(ComputeContext& context, const ConvAttributes& attributes) const {
  // Check attributes
  const bool isChannelsLast = attributes.nchw;  // Assuming 'nchw' corresponds to 'NHWC' format

  // Calculate output shape
  std::vector<int64_t> outputShape;
  CalculateOutputShape(context, attributes, outputShape);

  // Get input tensors
  const Tensor* inputTensor = context.Input<Tensor>(0);                                      // Input tensor
  const Tensor* weightTensor = context.Input<Tensor>(1);                                     // Weight tensor
  const Tensor* biasTensor = context.InputCount() > 2 ? context.Input<Tensor>(2) : nullptr;  // Bias tensor (optional)

  if (attributes.group != 1) {
    // Handle grouped convolution
    std::vector<const Tensor*> convInputs = {inputTensor};
    if (isChannelsLast) {
      if (!attributes.wT) {
        // Create and run the transpose program
        auto transposeProgram = CreateTransposeProgram(weightTensor, {1, 0});  // Assuming permAttr = [1, 0]
        ORT_RETURN_IF_ERROR(context.RunProgram(*transposeProgram));

        // Store the transposed weight in attributes.wT
        attributes.wT = transposeProgram->Outputs()[0].tensor;
      }
      convInputs.push_back(*attributes.wT);
    } else {
      convInputs.push_back(weightTensor);
    }
    if (biasTensor) {
      convInputs.push_back(biasTensor);
    }

    // Check if grouped convolution with vectorization is enabled
    std::string_view adaptor_arch = context.AdapterInfo().architecture;
    bool enableGroupedConvVectorize = adaptor_arch != "ampere";
    if (enableGroupedConvVectorize && isChannelsLast &&
        weightTensor->Shape()[0] == attributes.group && weightTensor->Shape()[1] == 1 &&
        attributes.dilations[0] == 1 && attributes.dilations[1] == 1) {
      auto groupedConvVectorizeProgram = CreateGroupedConvVectorizeProgram(
          convInputs, attributes, outputShape, /*squeezeOutputShapeFunction=*/nullptr);
      return context.RunProgram(*groupedConvVectorizeProgram);
    } else {
      auto groupedConvProgram = CreateGroupedConvProgram(
          convInputs, attributes, outputShape, /*squeezeOutputShapeFunction=*/nullptr);
      return context.RunProgram(*groupedConvProgram);
    }
  }

  // Handle non-grouped convolution
  const bool hasBias = biasTensor != nullptr;
  const int64_t inputHeight = inputTensor->Shape()[isChannelsLast ? 1 : 2];
  const int64_t inputWidth = inputTensor->Shape()[isChannelsLast ? 2 : 3];
  const int64_t inputChannels = inputTensor->Shape()[isChannelsLast ? 3 : 1];
  const int64_t weightHeight = weightTensor->Shape()[2];
  const int64_t weightWidth = weightTensor->Shape()[3];

  const int64_t outHeight = outputShape[isChannelsLast ? 1 : 2];
  const int64_t outWidth = outputShape[isChannelsLast ? 2 : 3];
  const int64_t outChannels = outputShape[isChannelsLast ? 3 : 1];

  // Check if convolution can be optimized as matrix multiplication
  const bool sameSize = isChannelsLast &&
                        weightHeight == inputHeight &&
                        weightWidth == inputWidth &&
                        attributes.pads[0] == 0 &&
                        attributes.pads[1] == 0;
  const bool is1x1Conv = weightHeight == 1 && weightWidth == 1 &&
                         attributes.dilations[0] == 1 && attributes.dilations[1] == 1 &&
                         attributes.strides[0] == 1 && attributes.strides[1] == 1 &&
                         attributes.pads[0] == 0 && attributes.pads[1] == 0;

  if (sameSize || is1x1Conv) {
    // Perform convolution using matrix multiplication
    const int64_t batch = outputShape[0];
    Tensor xReshaped;
    Tensor wReshaped;
    std::vector<int64_t> matmulOutputShape;
    std::vector<const Tensor*> matmulInputs;

    if (isChannelsLast) {
      if (!attributes.wT) {
        // Create and run the transpose program
        auto transposeProgram = CreateTransposeProgram(weightTensor, {1, 0});  // Assuming permAttr = [1, 0]
        ORT_RETURN_IF_ERROR(context.RunProgram(*transposeProgram));

        // Store the transposed weight in attributes.wT
        attributes.wT = transposeProgram->Outputs()[0].tensor;
      }

      if (sameSize) {
        const int64_t sharedDim = inputHeight * inputWidth * inputChannels;
        xReshaped = inputTensor->ReshapeToTensor({1, batch, sharedDim});
        wReshaped = attributes.wT.value()->ReshapeToTensor({1, sharedDim, outChannels});
        matmulOutputShape = {1, batch, outChannels};
      } else {
        xReshaped = inputTensor->ReshapeToTensor({batch, inputHeight * inputWidth, inputChannels});
        wReshaped = attributes.wT.value()->ReshapeToTensor({1, inputChannels, outChannels});
        matmulOutputShape = {batch, outHeight * outWidth, outChannels};
      }
      matmulInputs.push_back(&xReshaped);
      matmulInputs.push_back(&wReshaped);
    } else {
      xReshaped = inputTensor->ReshapeToTensor({batch, inputChannels, inputHeight * inputWidth});
      wReshaped = weightTensor->ReshapeToTensor({1, outChannels, inputChannels});
      matmulOutputShape = {batch, outChannels, outHeight * outWidth};
      matmulInputs.push_back(&wReshaped);
      matmulInputs.push_back(&xReshaped);
    }

    if (hasBias) {
      matmulInputs.push_back(biasTensor);
    }

    const int64_t N = matmulOutputShape[2];
    const int64_t K = matmulInputs[0]->Shape().GetDims().back();
    if (N < 8 && K < 8) {
      auto naiveMatmulProgram = CreateNaiveMatmulProgram(
          matmulInputs, attributes, outputShape, matmulOutputShape, isChannelsLast, /*squeezeOutputShapeFunction=*/nullptr);
      return context.RunProgram(*naiveMatmulProgram);
    } else {
      auto matmulProgram = CreateMatmulProgram(
          matmulInputs, attributes, outputShape, matmulOutputShape, isChannelsLast, /*squeezeOutputShapeFunction=*/nullptr);
      return context.RunProgram(*matmulProgram);
    }
  }

  // Default convolution using matrix multiplication
  const bool sequentialAccessByThreads = true;  // Assuming Intel backend

  // STEP 1: Transpose weight
  if (!attributes.wT) {
    auto transposeProgram = CreateTransposeProgram(weightTensor, {1, 0});  // Assuming permAttr = [1, 0]
    ORT_RETURN_IF_ERROR(context.RunProgram(*transposeProgram));

    // Store the transposed weight in attributes.wT
    attributes.wT = transposeProgram->Outputs()[0].tensor;
  }

  // STEP 2: Prepare reshaped inputs
  std::vector<const Tensor*> convInputs = {inputTensor, *attributes.wT};
  if (hasBias) {
    convInputs.push_back(biasTensor);
  }

  // STEP 3: Compute matmul
  const int64_t dimAOuter = isChannelsLast ? outHeight * outWidth : outChannels;
  const int64_t dimBOuter = isChannelsLast ? outChannels : outHeight * outWidth;
  const int64_t dimInner = weightHeight * weightWidth * inputChannels;

  auto conv2DMatMulProgram = CreateConv2DMatMulProgram(
      convInputs, attributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias, sequentialAccessByThreads, /*squeezeOutputShapeFunction=*/nullptr);
  return context.RunProgram(*conv2DMatMulProgram);
}

}  // namespace webgpu
}  // namespace onnxruntime
