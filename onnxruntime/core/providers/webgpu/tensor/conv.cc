// conv.cc
#include "conv.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include <string>

namespace onnxruntime {
namespace webgpu {

namespace {

// Helper function to get the maximum number of vector components
uint32_t GetMaxComponents(int64_t size) {
  // We cannot use vec3 type since it has alignment of 16 bytes
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

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
  gsl::span<const uint32_t> dilations_span;
  if (info.GetAttrsAsSpan("dilations", dilations_span).IsOK()) {
    convAttrs.dilations = std::vector<uint32_t>(dilations_span.begin(), dilations_span.end());
  }

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

// Helper function to get adjusted permutation
TensorShapeVector GetAdjustedPerm(size_t inputRank, const TensorShapeVector& permAttr) {
  if (permAttr.empty() || permAttr.size() != inputRank) {
    TensorShapeVector reversedPerm(inputRank);
    std::iota(reversedPerm.rbegin(), reversedPerm.rend(), 0);
    return reversedPerm;
  }
  return permAttr;
}

// Helper function to get output shape based on permutation
TensorShapeVector GetOutputShape(const TensorShapeVector& inputShape, const TensorShapeVector& perm) {
  TensorShapeVector outputShape(inputShape.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    outputShape[i] = inputShape[perm[i]];
  }
  return outputShape;
}

// Helper function to check if transpose is equivalent to a reshape
bool IsTransposeReshape(const TensorShapeVector& perm, const gsl::span<const int64_t>& shape) {
  int64_t lastPermutedAxis = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (shape[perm[i]] == 1) {
      continue;
    }
    if (perm[i] < lastPermutedAxis) {
      return false;
    }
    lastPermutedAxis = perm[i];
  }
  return true;
}

// Helper function to check if arrays are equal
bool AreEqual(const TensorShapeVector& arr1, const std::vector<int>& arr2) {
  if (arr1.size() != arr2.size()) return false;
  for (size_t i = 0; i < arr1.size(); i++) {
    if (arr1[i] != arr2[i]) return false;
  }
  return true;
}

// Helper function to convert output batch indices to input batch indices
std::string ConvertOutputBatchIndicesToInputBatchIndices(
    const std::string& target_indices_name,
    const ShaderVariableHelper& input_var,
    size_t input_batch_rank,
    size_t output_batch_rank,
    const std::string& batch_indices_name) {
  // Assume output_batch_rank >= input_batch_rank, the first output_batch_rank - input_batch_rank of
  // output_batch_rank should be ignored
  const size_t extending_input_rank = output_batch_rank - input_batch_rank;
  std::stringstream code;

  for (size_t i = 0; i < input_batch_rank; ++i) {
    code << "if (" << GetElementAt(input_var.Shape(), i, input_var.Rank()) << " != 1) {\n"
         << "  " << input_var.IndicesSet(target_indices_name, i, GetElementAt(batch_indices_name, i + extending_input_rank, output_batch_rank)) << "\n"
         << "} else {\n"
         << "  " << input_var.IndicesSet(target_indices_name, i, "0") << "\n"
         << "}\n";
  }

  return code.str();
}

// Helper function to squeeze shape
auto SqueezeShape(const gsl::span<const int64_t>& shape, const gsl::span<const size_t>& adjusted_perm, InlinedVector<int64_t>& new_shape, InlinedVector<int64_t>& new_perm) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != 1) {
      new_shape.push_back(shape[i]);
    }
    if (shape[adjusted_perm[i]] != 1) {
      new_perm.push_back(adjusted_perm[i]);
    }
  }
}

// The program creator
Status RunTransposeProgram(ComputeContext& context, const gsl::span<const size_t>& permutations, Tensor*& output_tensor) {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();

  // Compute output shape based on permutations
  TensorShapeVector output_dims(input_shape.NumDimensions());
  for (size_t i = 0; i < permutations.size(); ++i) {
    output_dims[i] = input_shape[permutations[i]];
  }
  TensorShape output_shape(output_dims);
  output_tensor = context.Output(0, output_shape);

  // Compute use_shared flag based on shape and permutations
  InlinedVector<int64_t> new_shape{};
  InlinedVector<int64_t> new_perm{};
  SqueezeShape(input_shape.GetDims(), permutations, new_shape, new_perm);
  const bool channels_last = new_perm == InlinedVector<int64_t>({2, 3, 1});
  const bool channels_first = new_perm == InlinedVector<int64_t>({3, 1, 2});
  const bool use_shared = (new_shape.size() == 2 && new_perm[0] > new_perm[1]) || channels_last || channels_first;

  // Handle shape transformations for shared memory case
  auto new_input_shape = input_shape;
  TensorShape new_output_shape(output_dims);
  if (use_shared) {
    new_input_shape = channels_last
                          ? TensorShape({new_shape[0], new_shape[1] * new_shape[2]})
                      : channels_first
                          ? TensorShape({new_shape[0] * new_shape[1], new_shape[2]})
                          : new_shape;
    new_output_shape = TensorShape({new_input_shape[1], new_input_shape[0]});
  }

  // Create and configure the transpose program
  TransposeProgram program{permutations, use_shared};

  // Set workgroup sizes based on tile size
  if (use_shared) {
    program.SetWorkgroupSize(Transpose::TILE_SIZE, Transpose::TILE_SIZE, 1);
  }

  // Configure program inputs/outputs
  program
      .CacheHint(absl::StrJoin(permutations, "-"))
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank, new_input_shape, 1}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::None, new_output_shape, 1}})
      .AddUniformVariables({
          {static_cast<uint32_t>(input_tensor->Shape().Size())},
      });

  // Set dispatch group sizes
  if (use_shared) {
    program.SetDispatchGroupSize(
        static_cast<uint32_t>((new_output_shape[1] + Transpose::TILE_SIZE - 1) / Transpose::TILE_SIZE),
        static_cast<uint32_t>((new_output_shape[0] + Transpose::TILE_SIZE - 1) / Transpose::TILE_SIZE));
  } else {
    program.SetDispatchGroupSize(
        (input_tensor->Shape().Size() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  }

  // Run the program
  return context.RunProgram(program);
}

Status RunGroupedConvVectorizeProgram(
    ComputeContext& context,
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const TensorShapeVector& outputShape,
    Tensor*& output_tensor) {
  const bool has_bias = inputs.size() > 2;
  const uint32_t components = GetMaxComponents(outputShape[3]);
  const uint32_t output_number = GetMaxComponents(outputShape[2]);
  const size_t output_size = TensorShape(outputShape).Size() / components / output_number;

  std::vector<uint32_t> x_shape = {
      static_cast<uint32_t>(inputs[0]->Shape()[0]),
      static_cast<uint32_t>(inputs[0]->Shape()[1]),
      static_cast<uint32_t>(inputs[0]->Shape()[2]),
      static_cast<uint32_t>(inputs[0]->Shape()[3] / components)};

  std::vector<uint32_t> w_shape = {
      static_cast<uint32_t>(inputs[1]->Shape()[0]),
      static_cast<uint32_t>(inputs[1]->Shape()[1]),
      static_cast<uint32_t>(inputs[1]->Shape()[2]),
      static_cast<uint32_t>(inputs[1]->Shape()[3] / components)};

  std::vector<uint32_t> output_shape_in_shader = {
      static_cast<uint32_t>(outputShape[0]),
      static_cast<uint32_t>(outputShape[1]),
      static_cast<uint32_t>(outputShape[2]),
      static_cast<uint32_t>(outputShape[3] / components)};

  const uint32_t x_number = (output_number - 1) * static_cast<uint32_t>(attributes.strides[1]) + static_cast<uint32_t>(w_shape[1]);

  // Create program
  auto program = std::make_unique<GroupedConvVectorizeProgram>();

  // Set attributes
  program->attributes_.convAttributes = attributes;
  program->attributes_.components = components;
  program->attributes_.output_number = output_number;
  program->attributes_.x_number = x_number;
  program->attributes_.x_shape = x_shape;
  program->attributes_.w_shape = w_shape;
  program->attributes_.output_shape = output_shape_in_shader;
  program->attributes_.has_bias = has_bias;
  program->attributes_.strides = {
      static_cast<int32_t>(attributes.strides[0]),
      static_cast<int32_t>(attributes.strides[1])};
  program->attributes_.pads = {
      static_cast<int32_t>(attributes.pads[0]),
      static_cast<int32_t>(attributes.pads[1])};

  // Configure inputs/outputs
  program->AddInputs({{inputs[0], ProgramTensorMetadataDependency::TypeAndShape}});
  program->AddInputs({{inputs[1], ProgramTensorMetadataDependency::None}});
  if (has_bias) {
    program->AddInputs({{inputs[2], ProgramTensorMetadataDependency::None}});
  }

  output_tensor = context.Output(0, TensorShape(outputShape));
  program->AddOutput({output_tensor, ProgramTensorMetadataDependency::TypeAndShape});

  // Set dispatch sizes
  program->SetDispatchGroupSize((static_cast<uint32_t>(output_size) + 63) / 64);

  // Add uniforms
  program->AddUniformVariables({{static_cast<uint32_t>(output_size)},
                                {program->attributes_.strides},
                                {program->attributes_.pads},
                                {attributes.clipMax.has_value() ? attributes.clipMax.value() : 3.4028234663852886e38f},
                                {attributes.clipMin.has_value() ? attributes.clipMin.value() : -3.4028234663852886e38f},
                                {attributes.alpha.has_value() ? attributes.alpha.value() : 0.2f},
                                {attributes.beta.has_value() ? attributes.beta.value() : 0.5f}});

  // Run the program
  return context.RunProgram(*program);
}

Status RunGroupedConvProgram(
    ComputeContext& context,
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const TensorShapeVector& output_shape,
    Tensor*& output_tensor) {
  const bool has_bias = inputs.size() > 2;
  const auto* x = inputs[0];
  const auto* w = inputs[1];
  const auto* b = has_bias ? inputs[2] : nullptr;
  const auto& x_shape = x->Shape().AsShapeVector();
  const auto& w_shape = w->Shape().AsShapeVector();

  const bool is_channel_last = !attributes.nchw;
  const int64_t output_channels = is_channel_last ? output_shape[3] : output_shape[1];
  const int64_t output_channels_per_group = output_channels / attributes.group;
  const uint32_t components = (is_channel_last && output_channels_per_group >= 4) ? GetMaxComponents(output_channels) : 1;
  const size_t output_size = TensorShape(output_shape).Size() / components;

  // Create program
  auto program = std::make_unique<GroupedConvProgram>();

  // Set program attributes
  program->attributes_.convAttributes = attributes;
  program->attributes_.components = components;
  program->attributes_.output_channels_per_group = static_cast<uint32_t>(output_channels_per_group);
  program->attributes_.has_bias = has_bias;
  program->attributes_.is_channel_last = is_channel_last;

  // Set tensor shapes
  program->attributes_.x_shape = {
      static_cast<uint32_t>(x_shape[0]),
      static_cast<uint32_t>(x_shape[1]),
      static_cast<uint32_t>(x_shape[2]),
      static_cast<uint32_t>(x_shape[3])};

  program->attributes_.w_shape = {
      static_cast<uint32_t>(w_shape[0]),
      static_cast<uint32_t>(w_shape[1]),
      static_cast<uint32_t>(w_shape[2]),
      static_cast<uint32_t>(w_shape[3] / components)};

  program->attributes_.output_shape = {
      static_cast<uint32_t>(output_shape[0]),
      static_cast<uint32_t>(output_shape[1]),
      static_cast<uint32_t>(output_shape[2]),
      static_cast<uint32_t>(output_shape[3] / components)};

  // Configure program inputs/outputs
  output_tensor = context.Output(0, TensorShape(output_shape));

  program->AddInputs({{x, ProgramTensorMetadataDependency::TypeAndShape}});
  program->AddInputs({{w, ProgramTensorMetadataDependency::TypeAndShape}});
  if (has_bias) {
    program->AddInputs({{b, ProgramTensorMetadataDependency::None}});
  }
  program->AddOutput({output_tensor, ProgramTensorMetadataDependency::None});

  // Set dispatch size
  program->SetDispatchGroupSize((static_cast<uint32_t>(output_size) + 63) / 64);

  // Add uniform variables
  program->AddUniformVariables({{static_cast<uint32_t>(output_size)},
                                {attributes.dilations},
                                {std::vector<uint32_t>{static_cast<uint32_t>(attributes.strides[0]),
                                                       static_cast<uint32_t>(attributes.strides[1])}},
                                {std::vector<uint32_t>{static_cast<uint32_t>(attributes.pads[0]),
                                                       static_cast<uint32_t>(attributes.pads[1])}},
                                {static_cast<uint32_t>(output_channels_per_group)},
                                {attributes.clipMax.has_value() ? attributes.clipMax.value() : 3.4028234663852886e38f},
                                {attributes.clipMin.has_value() ? attributes.clipMin.value() : -3.4028234663852886e38f},
                                {attributes.alpha.has_value() ? attributes.alpha.value() : 0.2f},
                                {attributes.beta.has_value() ? attributes.beta.value() : 0.5f}});

  // Run the program
  return context.RunProgram(*program);
}

Status RunNaiveMatmulProgram(
    ComputeContext& context,
    const std::vector<const Tensor*>& inputs,
    const InternalActivationAttributes& activationAttributes,
    const TensorShapeVector& outputShape,
    const TensorShapeVector& reshapedOutputShape,
    bool isChannelsLast,
    Tensor*& output_tensor) {
  // Extract dimensions
  const auto& a_shape = inputs[0]->Shape().GetDims();
  const auto& b_shape = inputs[1]->Shape().GetDims();
  const bool has_bias = inputs.size() > 2;

  // Calculate M, N, K dimensions and components
  const int64_t M = a_shape[a_shape.size() - 2];
  const int64_t N = b_shape[b_shape.size() - 1];
  const int64_t K = a_shape[a_shape.size() - 1];
  const uint32_t components = GetMaxComponents(N);
  const uint32_t a_components = GetMaxComponents(K);
  const uint32_t output_number = GetMaxComponents(M);

  // Calculate output dimensions
  std::vector<uint32_t> outerDims;
  for (size_t i = 0; i < reshapedOutputShape.size() - 2; ++i) {
    outerDims.push_back(static_cast<uint32_t>(reshapedOutputShape[i]));
  }
  const int64_t batchSize = std::accumulate(outerDims.begin(), outerDims.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  const TensorShapeVector outputShapeInShader{batchSize, M, N};
  const size_t outputSize = TensorShape(outputShape).Size() / components / output_number;

  // Create and setup program
  auto program = std::make_unique<NaiveMatmulProgram>();
  program->attributes_.components = components;
  program->attributes_.a_components = a_components;
  program->attributes_.output_number = output_number;
  program->attributes_.has_bias = has_bias;
  program->attributes_.isChannelsLast = isChannelsLast;
  program->attributes_.activationAttributes = activationAttributes;
  program->attributes_.M = static_cast<uint32_t>(M);
  program->attributes_.N = static_cast<uint32_t>(N);
  program->attributes_.K = static_cast<uint32_t>(K);
  program->attributes_.batchSize = static_cast<uint32_t>(batchSize);
  program->attributes_.outputShapeSize = static_cast<uint32_t>(outputShape.size());

  // Configure program inputs
  program->AddInputs({{inputs[0], ProgramTensorMetadataDependency::TypeAndRank}});
  program->AddInputs({{inputs[1], ProgramTensorMetadataDependency::TypeAndRank}});
  if (has_bias) {
    program->AddInputs({{inputs[2], ProgramTensorMetadataDependency::TypeAndRank}});
  }
  // allocate a Tensor object for outerDims
  std::vector<int64_t> outerDimsInt64(outerDims.begin(), outerDims.end());
  auto outerDimsTensor = context.CreateGPUTensor(inputs[0]->DataType(), TensorShape(outerDimsInt64));
  program->AddInputs({{&outerDimsTensor, ProgramTensorMetadataDependency::TypeAndRank}});

  // Configure program outputs
  output_tensor = context.Output(0, TensorShape(outputShape));
  program->AddOutput({output_tensor, ProgramTensorMetadataDependency::None});

  // Set dispatch size
  program->SetDispatchGroupSize((static_cast<uint32_t>(outputSize) + 63) / 64);

  // Add uniform variables
  program->AddUniformVariables({
      {static_cast<uint32_t>(outputSize)},
      {static_cast<uint32_t>(M)},
      {static_cast<uint32_t>(N)},
      {static_cast<uint32_t>(K)},
  });

  // Cache the program
  program->CacheHint(absl::StrJoin(
      {std::to_string(components), std::to_string(a_components), std::to_string(output_number), std::to_string(isChannelsLast)}, ";"));

  // Run the program
  return context.RunProgram(*program);
}

Status RunMatmulProgram(
    ComputeContext& context,
    const std::vector<const Tensor*>& inputs,
    const InternalActivationAttributes& activationAttributes,
    const TensorShapeVector& outputShape,
    const TensorShapeVector& reshapedOutputShape,
    bool isChannelsLast,
    Tensor*& output_tensor) {
  ORT_THROW("Matmul is not implemented yet.");
}

Status RunConv2DMatMulProgram(
    ComputeContext& context,
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const TensorShapeVector& outputShape,
    int64_t dimAOuter,
    int64_t dimBOuter,
    int64_t dimInner,
    bool hasBias,
    bool sequentialAccessByThreads,
    Tensor*& output_tensor) {
  ORT_THROW("Conv2DMatMul is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateGroupedConvVectorizeProgram(
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const TensorShapeVector& outputShape,
    const std::function<TensorShapeVector(const TensorShapeVector&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateGroupedConvVectorizeProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateGroupedConvProgram(
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const TensorShapeVector& outputShape,
    const std::function<TensorShapeVector(const TensorShapeVector&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateGroupedConvProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateNaiveMatmulProgram(
    const std::vector<const Tensor*>& inputs,
    const InternalActivationAttributes& activationAttributes,
    const TensorShapeVector& outputShape,
    const TensorShapeVector& reshapedOutputShape,
    bool isChannelsLast,
    const std::function<TensorShapeVector(const TensorShapeVector&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateNaiveMatmulProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateMatmulProgram(
    const std::vector<const Tensor*>& inputs,
    const InternalActivationAttributes& activationAttributes,
    const TensorShapeVector& outputShape,
    const TensorShapeVector& reshapedOutputShape,
    bool isChannelsLast,
    const std::function<TensorShapeVector(const TensorShapeVector&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateMatmulProgram is not implemented yet.");
}

std::unique_ptr<details::ProgramWrapper> CreateConv2DMatMulProgram(
    const std::vector<const Tensor*>& inputs,
    const ConvAttributes& attributes,
    const TensorShapeVector& outputShape,
    int64_t dimAOuter,
    int64_t dimBOuter,
    int64_t dimInner,
    bool hasBias,
    bool sequentialAccessByThreads,
    const std::function<TensorShapeVector(const TensorShapeVector&)>& squeezeOutputShapeFunction) {
  ORT_THROW("CreateConv2DMatMulProgram is not implemented yet.");
}

std::string GetActivationSnippet(
    const InternalActivationAttributes& attributes,
    const std::string& valueType,
    const std::string& baseType = "f32") {
  switch (attributes.activationAttributes) {
    case InternalActivationKind::kRelu:
      return "value = max(value, " + valueType + "(0.0));";

    case InternalActivationKind::kSigmoid:
      return "value = (" + valueType + "(1.0) / (" + valueType + "(1.0) + exp(-value)));";

    case InternalActivationKind::kClip:
      return "value = clamp(value, " + valueType + "(" + baseType + "(uniforms.clip_min)), " +
             valueType + "(" + baseType + "(uniforms.clip_max)));";

    case InternalActivationKind::kHardSigmoid:
      return "value = max(" + valueType + "(0.0), min(" + valueType + "(1.0), " +
             baseType + "(uniforms.alpha) * value + " + baseType + "(uniforms.beta)));";

    case InternalActivationKind::kLeakyRelu:
      return "value = select(" + baseType + "(uniforms.alpha) * value, value, value >= " + valueType + "(0.0));";

    case InternalActivationKind::kTanh:
      return "let e2x = exp(-2.0 * abs(value));\n"
             "          value = sign(value) * (1.0 - e2x) / (1.0 + e2x);";

    case InternalActivationKind::kUndef:
      return "";

    default:
      ORT_THROW("Unsupported activation");
  }
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
    ConvAttributes adjusted_attributes = GetAdjustedConvAttributes(attributes_, context);
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

void Conv::CalculateOutputShape(const ComputeContext& context, const ConvAttributes& attributes, TensorShapeVector& outputShape) const {
  // Get input tensor
  const Tensor* inputTensor = context.Input<Tensor>(0);
  const auto& inputShape = inputTensor->Shape().GetDims();

  // Extract attributes
  const auto& kernelShape = attributes.kernelShape;
  const auto& dilations = attributes.dilations;
  const auto& adjustPads = attributes.pads;
  const auto& strides = attributes.strides;
  const bool isChannelsLast = !attributes.nchw;

  // Extract batch size
  const int64_t batchSize = inputShape[0];

  // Extract input spatial shape (height, width)
  const size_t spatialRank = inputShape.size() - 2;  // Rank of spatial dimensions (2 for 2D convolution)
  TensorShapeVector inputSpatialShape;
  if (isChannelsLast) {
    inputSpatialShape = TensorShapeVector(inputShape.begin() + 1, inputShape.end() - 1);
  } else {
    inputSpatialShape = TensorShapeVector(inputShape.begin() + 2, inputShape.end());
  }

  // Extract output channels
  const int64_t outChannels = kernelShape[0];

  // Extract kernel spatial shape (height, width)
  TensorShapeVector kernelSpatialShape(kernelShape.begin() + 2, kernelShape.end());

  // Calculate dilated kernel shape
  TensorShapeVector dilatedKernelShape;
  for (size_t i = 0; i < kernelSpatialShape.size(); ++i) {
    dilatedKernelShape.push_back(kernelSpatialShape[i] + (kernelSpatialShape[i] - 1) * (dilations[i] - 1));
  }

  // Calculate input spatial shape with padding
  TensorShapeVector inputSpatialShapeWithPad;
  for (size_t i = 0; i < inputSpatialShape.size(); ++i) {
    inputSpatialShapeWithPad.push_back(inputSpatialShape[i] + adjustPads[i] + adjustPads[i + spatialRank]);
  }

  // Calculate output spatial shape
  TensorShapeVector outputSpatialShape;
  for (size_t i = 0; i < inputSpatialShapeWithPad.size(); ++i) {
    outputSpatialShape.push_back(
        (inputSpatialShapeWithPad[i] - dilatedKernelShape[i] + strides[i]) / strides[i]);
  }

  // Construct final output shape
  outputShape.clear();
  outputShape.push_back(batchSize);  // Batch size
  if (isChannelsLast) {
    outputShape.insert(outputShape.end(), outputSpatialShape.begin(), outputSpatialShape.end());  // Spatial dimensions
    outputShape.push_back(outChannels);                                                           // Output channels
  } else {
    outputShape.push_back(outChannels);
    outputShape.insert(outputShape.end(), outputSpatialShape.begin(), outputSpatialShape.end());  // Spatial dimensions
  }
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
  TensorShapeVector outputShape;
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
        Tensor* output_tensor;
        std::vector<size_t> perm = {1, 0};                                       // Define permutation
        ORT_RETURN_IF_ERROR(RunTransposeProgram(context, perm, output_tensor));  // Use the new interface

        // Store the transposed weight in attributes.wT
        attributes.wT = output_tensor;
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
    bool enable_grouped_conv_vectorize = adaptor_arch != "ampere";
    if (enable_grouped_conv_vectorize && isChannelsLast &&
        weightTensor->Shape()[0] == attributes.group && weightTensor->Shape()[1] == 1 &&
        attributes.dilations[0] == 1 && attributes.dilations[1] == 1) {
      Tensor* output_tensor;
      ORT_RETURN_IF_ERROR(RunGroupedConvVectorizeProgram(
          context, convInputs, attributes, outputShape, output_tensor));
      return Status::OK();
    } else {
      Tensor* output_tensor;
      ORT_RETURN_IF_ERROR(RunGroupedConvProgram(
          context, convInputs, attributes, outputShape, output_tensor));
      return Status::OK();
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
    TensorShapeVector xReshaped;
    TensorShapeVector wReshaped;
    TensorShapeVector matmulOutputShape;
    std::vector<TensorShapeVector> matmulInputs;

    if (isChannelsLast) {
      if (!attributes.wT) {
        // Create and run the transpose program
        std::vector<size_t> perm = {1, 0};  // Define permutation
        Tensor* output_tensor;

        ORT_THROW_IF_ERROR(RunTransposeProgram(context, perm, output_tensor));  // Assuming permAttr = [1, 0]

        // Store the transposed weight in attributes.wT
        attributes.wT = output_tensor;
      }

      if (sameSize) {
        const int64_t sharedDim = inputHeight * inputWidth * inputChannels;
        xReshaped = {1, batch, sharedDim};
        wReshaped = {1, sharedDim, outChannels};
        matmulOutputShape = {1, batch, outChannels};
      } else {
        xReshaped = {batch, inputHeight * inputWidth, inputChannels};
        wReshaped = {1, inputChannels, outChannels};
        matmulOutputShape = {batch, outHeight * outWidth, outChannels};
      }
      matmulInputs.push_back(xReshaped);
      matmulInputs.push_back(wReshaped);
    } else {
      xReshaped = {batch, inputChannels, inputHeight * inputWidth};
      wReshaped = {1, outChannels, inputChannels};
      matmulOutputShape = {batch, outChannels, outHeight * outWidth};
      matmulInputs.push_back(wReshaped);
      matmulInputs.push_back(xReshaped);
    }

    if (hasBias) {
      matmulInputs.push_back(biasTensor->Shape().AsShapeVector());
    }

    const int64_t N = matmulOutputShape[2];
    const int64_t K = matmulInputs[0].back();
    if (N < 8 && K < 8) {
      Tensor* output_tensor;
      ORT_RETURN_IF_ERROR(RunNaiveMatmulProgram(
          context, matmulInputs, attributes, outputShape, matmulOutputShape, isChannelsLast, output_tensor));
      return Status::OK();
    } else {
      Tensor* output_tensor;
      ORT_RETURN_IF_ERROR(RunMatmulProgram(
          context, matmulInputs, attributes, outputShape, matmulOutputShape, isChannelsLast, output_tensor));
      return Status::OK();
    }
  }

  // Default convolution using matrix multiplication
  const bool sequentialAccessByThreads = true;  // Assuming Intel backend

  // STEP 1: Transpose weight
  if (!attributes.wT) {
    Tensor* output_tensor;
    ORT_THROW_IF_ERROR(RunTransposeProgram(context, std::vector<std::size_t>({1, 0}), output_tensor));  // Assuming permAttr = [1, 0]

    // Store the transposed weight in attributes.wT
    attributes.wT = output_tensor;
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

  Tensor* output_tensor;
  ORT_RETURN_IF_ERROR(RunConv2DMatMulProgram(
      context, convInputs, attributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias, sequentialAccessByThreads, output_tensor));
  return Status::OK();
}

ConvAttributes Conv::GetAdjustedConvAttributes(const ConvAttributes& attributes, const ComputeContext& context) const {
  ConvAttributes adjustedAttributes = attributes;

  // Adjust kernelShape if not well specified
  auto kernelShape = attributes.kernelShape;
  if (kernelShape.size() < context.Input(1)->Shape().NumDimensions() - 2) {
    // Fill missing dimensions with 0
    kernelShape.resize(context.Input(1)->Shape().NumDimensions() - 2, 0);
  }

  // Infer kernelShape from the weight tensor dimensions
  for (size_t i = 2; i < context.Input(1)->Shape().NumDimensions(); ++i) {
    if (kernelShape[i - 2] == 0) {
      kernelShape[i - 2] = context.Input(1)->Shape()[i];
    }
  }

  // Adjust pads based on autoPad
  TensorShapeVector pads = attributes.pads;
  AdjustPadsBasedOnAutoPad(
      context.Input(0)->Shape().AsShapeVector(),
      attributes.strides,
      TensorShapeVector(attributes.dilations.begin(), attributes.dilations.end()),
      kernelShape,
      pads,
      attributes.nchw,  // Assuming 'nchw' corresponds to 'NHWC' format
      attributes.autoPad);

  // Update adjustedAttributes with new kernelShape and pads
  adjustedAttributes.kernelShape = kernelShape;
  adjustedAttributes.pads = pads;

  return adjustedAttributes;
}

void Conv::AdjustPadsBasedOnAutoPad(
    const TensorShapeVector& inputDims,
    const TensorShapeVector& strides,
    const TensorShapeVector& dilations,
    const TensorShapeVector& kernelShape,
    TensorShapeVector& pads,
    bool isChannelsLast,
    AutoPadKind autoPad) const {
  if (autoPad == AutoPadKind::kNotSet) {
    return;
  }

  // Validate input dimensions
  if (pads.size() != 2 * (inputDims.size() - 2)) {
    ORT_THROW("Length of pads should be twice the length of data dimensions.");
  }

  if (strides.size() != inputDims.size() - 2) {
    ORT_THROW("Length of strides should be the length of data dimensions.");
  }

  if (kernelShape.size() != inputDims.size() - 2) {
    ORT_THROW("Length of kernel shapes should be the length of data dimensions.");
  }

  // Adjust pads for each dimension
  for (size_t dim = 0; dim < inputDims.size() - 2; ++dim) {
    AdjustPadAndReturnShape(
        inputDims[dim + (isChannelsLast ? 1 : 2)],
        strides[dim],
        dilations[dim],
        kernelShape[dim],
        pads,
        dim,
        dim + inputDims.size() - 2,
        autoPad);
  }
}

int64_t Conv::AdjustPadAndReturnShape(
    int64_t inputSize,
    int64_t stride,
    int64_t dilation,
    int64_t kernelSize,
    TensorShapeVector& pads,
    size_t padBeginIndex,
    size_t padEndIndex,
    AutoPadKind autoPad) const {
  const int64_t dkernel = dilation * (kernelSize - 1) + 1;

  if (autoPad != AutoPadKind::kNotSet) {
    switch (autoPad) {
      case AutoPadKind::kValid:
        // No padding
        pads[padBeginIndex] = 0;
        pads[padEndIndex] = 0;
        return (inputSize - dkernel) / stride + 1;

      case AutoPadKind::kSameUpper:
      case AutoPadKind::kSameLower:
        if (dilation != 1) {
          ORT_THROW("Dilation not supported for SAME_UPPER or SAME_LOWER.");
        } else {
          const int64_t legacyTargetSize = (inputSize + stride - 1) / stride;
          const int64_t padNeeded = (legacyTargetSize - 1) * stride + kernelSize - inputSize;

          if (autoPad == AutoPadKind::kSameLower) {
            pads[padBeginIndex] = (padNeeded + 1) / 2;
          } else {
            pads[padBeginIndex] = padNeeded / 2;
          }
          pads[padEndIndex] = padNeeded - pads[padBeginIndex];

          return (inputSize + padNeeded - kernelSize) / stride + 1;
        }

      default:
        ORT_THROW("Unsupported AutoPad value.");
    }
  } else {
    return (inputSize + pads[padBeginIndex] + pads[padEndIndex] - dkernel) / stride + 1;
  }
}

Status GroupedConvVectorizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Declare input and output variables with appropriate usage flags
  const auto& input = shader.AddInput("x", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  const auto& weights = shader.AddInput("w", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);

  shader.AddInput("b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);

  const std::string process_bias = attributes_.has_bias ? "value += b[output_channel];" : "";
  const std::string_view base_type = output.StorageType();
  const std::string apply_activation = GetActivationSnippet(attributes_.convAttributes, std::string(output.ValueType()), std::string(base_type));

  // Build shader code
  std::stringstream code;

  // Add guard against out-of-bounds workgroup sizes
  code << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size") << "\n";

  // Calculate indices and positions
  code << "let width0 = uniforms.output_shape[3];\n"
       << "let output_channel = global_idx % width0;\n"
       << "var index1 = global_idx / width0;\n"
       << "let width1 = uniforms.output_shape[2] / " << attributes_.output_number << "u;\n"
       << "let col = (index1 % width1) * " << attributes_.output_number << "u;\n"
       << "index1 = index1 / width1;\n"
       << "let row = index1 % uniforms.output_shape[1];\n"
       << "let batch = index1 / uniforms.output_shape[1];\n\n";

  // Calculate corner position
  code << "let x_corner = vec2<i32>(i32(row), i32(col)) * uniforms.strides - uniforms.pads;\n\n";

  // Declare arrays for intermediate values
  code << "var x_vals: array<" << "x_value_t" << ", " << attributes_.x_number << ">;\n"
       << "var values: array<" << "output_value_t" << ", " << attributes_.output_number << ">;\n"
       << "let input_channel = output_channel;\n\n";

  // Main computation loop
  code << "for (var w_height: u32 = 0u; w_height < " << attributes_.w_shape[0] << "u; w_height++) {\n"
       << "  let x_height = x_corner.x + i32(w_height);\n"
       << "  if (x_height >= 0 && u32(x_height) < uniforms.x_shape[1]) {\n"
       << "    for (var i = 0; i < " << attributes_.x_number << "; i++) {\n"
       << "      let x_width = x_corner.y + i;\n"
       << "      if (x_width >= 0 && u32(x_width) < uniforms.x_shape[2]) {\n"
       << "        x_vals[i] = " << input.GetByIndices("x_indices_t(batch, u32(x_height), u32(x_width), input_channel)") << ";\n"
       << "      } else {\n"
       << "        x_vals[i] = " << "x_value_t" << "(0);\n"
       << "      }\n"
       << "    }\n"
       << "    for (var w_width: u32 = 0u; w_width < " << attributes_.w_shape[1] << "u; w_width++) {\n"
       << "      let w_val = " << weights.GetByIndices("w_indices_t(w_height, w_width, 0, output_channel)") << ";\n"
       << "      for (var i = 0u; i < " << attributes_.output_number << "u; i++) {\n"
       << "        values[i] = fma(x_vals[i * u32(uniforms.strides[1]) + w_width], w_val, values[i]);\n"
       << "      }\n"
       << "    }\n"
       << "  }\n"
       << "}\n\n";

  // Write output values
  code << "for (var i = 0u; i < " << attributes_.output_number << "u; i++) {\n"
       << "  var value = values[i];\n"
       << "  " << process_bias << "\n"
       << "  " << apply_activation << "\n"
       << "  " << output.SetByIndices("output_indices_t(batch, row, col + i, output_channel)", "value") << ";\n"
       << "}\n";

  shader.MainFunctionBody() << code.str();
  return Status::OK();
}

Status GroupedConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Declare input and output variables
  const auto& x = shader.AddInput("x", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  const auto& w = shader.AddInput("w", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform);

  if (attributes_.has_bias) {
    shader.AddInput("b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  }

  const std::string process_bias = attributes_.has_bias ? "value += b[output_channel];" : "";
  const std::string_view base_type = output.StorageType();
  const std::string apply_activation = GetActivationSnippet(attributes_.convAttributes, std::string(output.ValueType()), std::string(base_type));

  // Build the main calculation logic based on layout
  std::string calculate_result;
  if (attributes_.is_channel_last) {
    calculate_result = R"(
      for (var wHeight: u32 = 0u; wHeight < uniforms.w_shape[0]; wHeight++) {
        let xHeight = xRCCorner.x + wHeight * uniforms.dilations[0];

        if (xHeight < 0u || xHeight >= uniforms.x_shape[1]) {
          continue;
        }

        for (var wWidth: u32 = 0u; wWidth < uniforms.w_shape[1]; wWidth++) {
          let xWidth = xRCCorner.y + wWidth * uniforms.dilations[1];
          if (xWidth < 0u || xWidth >= uniforms.x_shape[2]) {
            continue;
          }

          for (var wInChannel: u32 = 0u; wInChannel < uniforms.w_shape[2]; wInChannel++) {
            let input_channel = in_channel_offset + wInChannel;
            let xVal = )" +
                       x.GetByIndices("x_indices_t(batch, xHeight, xWidth, input_channel)") + R"(;
            let wVal = )" +
                       w.GetByIndices("w_indices_t(wHeight, wWidth, wInChannel, output_channel)") + R"(;
            value += xVal * wVal;
          }
        }
      }
    )";
  } else {
    calculate_result = R"(
      for (var wInChannel: u32 = 0u; wInChannel < uniforms.w_shape[1]; wInChannel++) {
        let input_channel = in_channel_offset + wInChannel;
        for (var wHeight: u32 = 0u; wHeight < uniforms.w_shape[2]; wHeight++) {
          let xHeight = xRCCorner.x + wHeight * uniforms.dilations[0];

          if (xHeight < 0u || xHeight >= uniforms.x_shape[2]) {
            continue;
          }

          for (var wWidth: u32 = 0u; wWidth < uniforms.w_shape[3]; wWidth++) {
            let xWidth = xRCCorner.y + wWidth * uniforms.dilations[1];
            if (xWidth < 0u || xWidth >= uniforms.x_shape[3]) {
              continue;
            }

            let xVal = )" +
                       x.GetByIndices("x_indices_t(batch, input_channel, xHeight, xWidth)") + R"(;
            let wVal = )" +
                       w.GetByIndices("w_indices_t(output_channel, wInChannel, wHeight, wWidth)") + R"(;
            value += xVal * wVal;
          }
        }
      }
    )";
  }

  // Build the complete shader code
  std::stringstream code;
  code << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size") << "\n\n"
       << "let outputIndices = " << output.OffsetToIndices("global_idx") << ";\n"
       << "let batch: u32 = outputIndices[0];\n"
       << "let output_channel: u32 = outputIndices[" << (attributes_.is_channel_last ? "3" : "1") << "];\n"
       << "let xRCCorner: vec2<u32> = vec2<u32>(outputIndices[" << (attributes_.is_channel_last ? "1" : "2")
       << "], outputIndices[" << (attributes_.is_channel_last ? "2" : "3") << "]) * uniforms.strides - uniforms.pads;\n"
       << "let group_id: u32 = output_channel * " << attributes_.components << "u / uniforms.output_channels_per_group;\n"
       << "var in_channel_offset = group_id * uniforms.w_shape[" << (attributes_.is_channel_last ? "2" : "1") << "];\n\n"
       << "var value: " << "output_value_t" << " = " << "output_value_t" << "(0);\n"
       << calculate_result << "\n"
       << process_bias << "\n"
       << apply_activation << "\n"
       << output.SetByOffset("global_idx", "value") << ";\n";

  shader.MainFunctionBody() << code.str();
  return Status::OK();
}

Status NaiveMatmulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Setup input/output variables
  const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);

  if (attributes_.has_bias) {
    shader.AddInput("bias", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  }
  const auto& batch_dims = shader.AddInput("batchDims", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseOffsetToIndices);

  const std::string_view base_type = "output_value_t";
  const std::string apply_activation = GetActivationSnippet(attributes_.activationAttributes, "output_value_t", std::string(base_type));

  // Generate bias processing code
  std::string process_bias;
  if (attributes_.has_bias) {
    const uint32_t biasComponents = attributes_.isChannelsLast ? attributes_.components : 1;
    process_bias = attributes_.isChannelsLast
                       ? absl::StrCat("value += bias[col / ", biasComponents, "];")
                       : absl::StrCat("value += ", "output_value_t", "(bias[row + i]);");
  }

  // Helper function to generate calculation result
  auto generate_calc_result = [&]() {
    std::stringstream cal_str;
    cal_str << "var a_data: " << "a_value_t" << ";\n";

    // Generate b_data loads
    for (uint32_t i = 0; i < attributes_.a_components; i++) {
      cal_str << "let b_data" << i << " = b[(b_offset + (k + " << i << ") * uniforms.N + col) / "
              << attributes_.components << "];\n";
    }

    // Generate calculation for each output
    for (uint32_t i = 0; i < attributes_.output_number; i++) {
      cal_str << "a_data = a[(a_offset + (row + " << i << ") * uniforms.K + k) / "
              << attributes_.a_components << "];\n";

      for (uint32_t j = 0; j < attributes_.a_components; j++) {
        cal_str << "values[" << i << "] = fma("
                << "b_value_t" << "(a_data" << (attributes_.a_components == 1 ? "" : "[" + std::to_string(j) + "]")
                << "), b_data" << j << ", values[" << i << "]);\n";
      }
    }
    return cal_str.str();
  };

  // Build the main shader code
  std::stringstream code;
  code << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size") << "\n\n"
       << "let col = (global_idx % (uniforms.N / " << attributes_.components << ")) * " << attributes_.components << ";\n"
       << "var index1 = global_idx / (uniforms.N / " << attributes_.components << ");\n"
       << "let stride1 = uniforms.M / " << attributes_.output_number << ";\n"
       << "let row = (index1 % stride1) * " << attributes_.output_number << ";\n"
       << "let batch = index1 / stride1;\n\n";

  // Add batch indices calculation if needed
  if (attributes_.outputShapeSize != 2) {
    code << "let batch_indices = " << batch_dims.OffsetToIndices("batch") << ";\n";
  }

  // Setup indices and offsets
  code << "var a_indices: " << "a_indices_t" << ";\n"
       << ConvertOutputBatchIndicesToInputBatchIndices("a_indices", a, a.Rank() - 2, batch_dims.Rank(), "batch_indices") << "\n"
       << a.IndicesSet("a_indices", a.Rank() - 2, "0") << "\n"
       << a.IndicesSet("a_indices", a.Rank() - 1, "0") << "\n"
       << "let a_offset = " << a.IndicesToOffset("a_indices") << ";\n\n"
       << "var b_indices: " << "b_indices_t" << ";\n"
       << ConvertOutputBatchIndicesToInputBatchIndices("b_indices", b, b.Rank() - 2, batch_dims.Rank(), "batch_indices") << "\n"
       << b.IndicesSet("b_indices", b.Rank() - 2, "0") << "\n"
       << b.IndicesSet("b_indices", b.Rank() - 1, "0") << "\n"
       << "let b_offset = " << b.IndicesToOffset("b_indices") << ";\n\n"
       << "var values: array<" << "output_value_t" << ", " << attributes_.output_number << ">;\n\n"
       << "for (var k: u32 = 0u; k < uniforms.K; k = k + " << attributes_.a_components << ") {\n"
       << generate_calc_result()
       << "}\n\n"
       << "for (var i = 0u; i < " << attributes_.output_number << "u; i++) {\n"
       << "  var value = values[i];\n"
       << "  " << process_bias << "\n"
       << "  " << apply_activation << "\n"
       << "  let cur_indices = " << "output_indices_t" << "(batch, row + i, col);\n"
       << "  let offset = " << output.IndicesToOffset("cur_indices") << ";\n"
       << "  " << output.SetByOffset("offset / " + std::to_string(attributes_.components), "value") << ";\n"
       << "}\n";

  shader.MainFunctionBody() << code.str();
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
