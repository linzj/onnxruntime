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

#if 0
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
#endif

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
    const TensorShapeVector& output_shape,
    Tensor*& output_tensor) {
  const bool has_bias = inputs.size() > 2;
  const uint32_t components = GetMaxComponents(output_shape[3]);
  const uint32_t output_number = GetMaxComponents(output_shape[2]);
  const size_t output_size = TensorShape(output_shape).Size() / components / output_number;

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
      static_cast<uint32_t>(output_shape[0]),
      static_cast<uint32_t>(output_shape[1]),
      static_cast<uint32_t>(output_shape[2]),
      static_cast<uint32_t>(output_shape[3] / components)};

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

  output_tensor = context.Output(0, TensorShape(output_shape));
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
    const std::vector<const TensorShape>& input_shapes,
    const InternalActivationAttributes& activationAttributes,
    const TensorShapeVector& outputShape,
    const TensorShapeVector& reshapedOutputShape,
    bool isChannelsLast) {
  // Extract dimensions
  const auto& a_shape = input_shapes[0].GetDims();
  const auto& b_shape = input_shapes[1].GetDims();
  const bool has_bias = input_shapes.size() > 2;

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
  program->AddInputs({{context.Input(0), ProgramTensorMetadataDependency::TypeAndRank, input_shapes[0].AsShapeVector(), static_cast<int>(a_components)}});
  program->AddInputs({{context.Input(1), ProgramTensorMetadataDependency::TypeAndRank, input_shapes[1].AsShapeVector(), static_cast<int>(components)}});
  if (has_bias) {
    const auto bias_components = isChannelsLast ? components : 1;
    program->AddInputs({{context.Input(2),
                         ProgramTensorMetadataDependency::TypeAndRank,
                         input_shapes[2].AsShapeVector(), static_cast<int>(bias_components)}});
  }
  // allocate a Tensor object for outerDims
  std::vector<int64_t> outerDimsInt64(outerDims.begin(), outerDims.end());
  auto outerDimsTensor = context.CreateGPUTensor(context.Input(0)->DataType(), TensorShape(outerDimsInt64));
  program->AddInputs({{&outerDimsTensor, ProgramTensorMetadataDependency::TypeAndRank}});

  // Configure program outputs
  auto* output_tensor = context.Output(0, TensorShape(outputShape));
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
    const std::vector<const TensorShape>& input_shapes,
    const InternalActivationAttributes& activation_attributes,
    const TensorShapeVector& output_shape, const TensorShapeVector* reshapedOutputShape, bool is_channels_last) {
  // Calculate shapes and dimensions
  const Tensor* A = context.Input(0);
  const Tensor* B = context.Input(1);
  const Tensor* bias = nullptr;
  if (context.InputCount() > 2) {
    bias = context.Input(2);
  }
  const auto& a_shape = A->Shape();
  const auto& b_shape = B->Shape();
  const int64_t a_rank = a_shape.NumDimensions();
  const int64_t b_rank = b_shape.NumDimensions();

  const int64_t dim_a_outer = a_shape[a_rank - 2];
  const int64_t dim_inner = a_shape[a_rank - 1];
  const int64_t dim_b_outer = b_shape[b_rank - 1];

  // Extract outer dimensions and calculate batch size
  std::vector<int64_t> outer_dims_a(a_shape.GetDims().begin(), a_shape.GetDims().end() - 2);
  std::vector<int64_t> outer_dims_b(b_shape.GetDims().begin(), b_shape.GetDims().end() - 2);
  std::vector<int64_t> outer_dims = !reshapedOutputShape ? std::vector<int64_t>(output_shape.begin(), output_shape.end() - 2) : std::vector<int64_t>(reshapedOutputShape->begin(), reshapedOutputShape->end() - 2);
  const int64_t batch_size = std::accumulate(outer_dims.begin(), outer_dims.end(), 1LL, std::multiplies<int64_t>());

  // Check if we can use vec4 optimization
  const bool is_vec4 = (dim_inner % 4 == 0) && (dim_b_outer % 4 == 0);
  const uint32_t components = is_vec4 ? 4 : 1;

  // Create temporary shapes for computations
  std::vector<int64_t> a_shape_tmp;
  a_shape_tmp.insert(a_shape_tmp.end(), outer_dims_a.begin(), outer_dims_a.end());
  a_shape_tmp.push_back(dim_a_outer);
  a_shape_tmp.push_back(dim_inner / components);

  std::vector<int64_t> b_shape_tmp;
  b_shape_tmp.insert(b_shape_tmp.end(), outer_dims_b.begin(), outer_dims_b.end());
  b_shape_tmp.push_back(dim_inner);
  b_shape_tmp.push_back(dim_b_outer / components);

  std::vector<int64_t> output_shape_tmp{batch_size, dim_a_outer, dim_b_outer / components};

  // Create TensorShape objects
  TensorShape a_shape_tensor(a_shape_tmp);
  TensorShape b_shape_tensor(b_shape_tmp);
  TensorShape output_shape_tensor(output_shape_tmp);

  // Setup MatmulProgram attributes
  auto program = std::make_unique<MatmulProgram>();
  program->attributes_.components = components;
  program->attributes_.batch_size = static_cast<uint32_t>(batch_size);
  program->attributes_.dim_a_outer = static_cast<uint32_t>(dim_a_outer);
  program->attributes_.dim_inner = static_cast<uint32_t>(dim_inner);
  program->attributes_.dim_b_outer = static_cast<uint32_t>(dim_b_outer);
  program->attributes_.has_bias = (bias != nullptr);
  program->attributes_.is_channels_last = is_channels_last;
  program->attributes_.activationAttributes = activation_attributes;
  program->attributes_.outer_dims = std::vector<uint32_t>(outer_dims.begin(), outer_dims.end());

  // Convert shapes to uint32
  auto to_uint32_vector = [](const TensorShape& shape) {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < shape.NumDimensions(); ++i) {
      result.push_back(static_cast<uint32_t>(shape[i]));
    }
    return result;
  };
  program->attributes_.a_shape = to_uint32_vector(a_shape);
  program->attributes_.b_shape = to_uint32_vector(b_shape);

  // Configure program inputs/outputs
  program->AddInputs({{A, ProgramTensorMetadataDependency::TypeAndShape, a_shape_tensor, static_cast<int>(components)}});
  program->AddInputs({{B, ProgramTensorMetadataDependency::TypeAndShape, b_shape_tensor, static_cast<int>(components)}});
  if (bias) {
    program->AddInputs({{bias, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  // allocate a Tensor object for outerDims
  std::vector<int64_t> outer_dims_int64(outer_dims.begin(), outer_dims.end());
  auto outer_dims_tensor = context.CreateGPUTensor(A->DataType(), TensorShape(outer_dims_int64));
  program->AddInputs({{&outer_dims_tensor, ProgramTensorMetadataDependency::TypeAndRank, 1}});

  program->AddOutput({context.Output(0, output_shape_tensor), ProgramTensorMetadataDependency::None, output_shape_tensor, static_cast<int>(components)});

  // Add uniform variables
  program->AddUniformVariables({{static_cast<int32_t>(dim_a_outer)},
                                {static_cast<int32_t>(dim_b_outer)},
                                {static_cast<int32_t>(dim_inner)},
                                {activation_attributes.clipMax.has_value() ? activation_attributes.clipMax.value() : 3.4028234663852886e38f},
                                {activation_attributes.clipMin.has_value() ? activation_attributes.clipMin.value() : -3.4028234663852886e38f},
                                {activation_attributes.alpha.has_value() ? activation_attributes.alpha.value() : 0.2f},
                                {activation_attributes.beta.has_value() ? activation_attributes.beta.value() : 0.5f}});

  // Setup dispatch parameters
  const std::array<uint32_t, 3> workgroup_size = {8, 8, 1};
  const std::array<uint32_t, 3> elements_per_thread = {
      dim_a_outer <= 8 ? 4u : 4u,
      dim_a_outer <= 8 ? 1u : 4u,
      1u};

  program->attributes_.elements_per_thread = elements_per_thread;  // Add this
  program->attributes_.workgroup_size = workgroup_size;            // Add this

  // Setup dispatch parameters

  std::array<uint32_t, 3> dispatch_size = {
      static_cast<uint32_t>(ceil(static_cast<float>(dim_b_outer) /
                                 (workgroup_size[0] * elements_per_thread[0]))),
      static_cast<uint32_t>(ceil(static_cast<float>(dim_a_outer) /
                                 (workgroup_size[1] * elements_per_thread[1]))),
      static_cast<uint32_t>(ceil(static_cast<float>(batch_size) /
                                 (workgroup_size[2] * elements_per_thread[2])))};

  program->SetDispatchGroupSize(dispatch_size[0], dispatch_size[1], dispatch_size[2]);
  program->SetWorkgroupSize(workgroup_size[0], workgroup_size[1], workgroup_size[2]);

  // Execute the program
  return context.RunProgram(*program);
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

void Conv::CalculateOutputShape(const ComputeContext& context, const ConvAttributes& attributes, TensorShapeVector& output_shape) const {
  // Get input tensor
  const Tensor* input_tensor = context.Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape().GetDims();

  // Extract attributes
  const auto& kernel_shape = attributes.kernelShape;
  const auto& dilations = attributes.dilations;
  const auto& adjust_pads = attributes.pads;
  const auto& strides = attributes.strides;
  const bool is_channel_last = !attributes.nchw;

  // Extract batch size
  const int64_t batch_size = input_shape[0];

  // Extract input spatial shape (height, width)
  const size_t spatial_rank = input_shape.size() - 2;  // Rank of spatial dimensions (2 for 2D convolution)
  TensorShapeVector input_spatial_shape;
  if (is_channel_last) {
    input_spatial_shape = TensorShapeVector(input_shape.begin() + 1, input_shape.end() - 1);
  } else {
    input_spatial_shape = TensorShapeVector(input_shape.begin() + 2, input_shape.end());
  }

  // Extract output channels
  const int64_t out_channels = kernel_shape[0];

  // Extract kernel spatial shape (height, width)
  TensorShapeVector kernel_spatial_shape(kernel_shape.begin() + 2, kernel_shape.end());

  // Calculate dilated kernel shape
  TensorShapeVector dilated_kernel_shape;
  for (size_t i = 0; i < kernel_spatial_shape.size(); ++i) {
    dilated_kernel_shape.push_back(kernel_spatial_shape[i] + (kernel_spatial_shape[i] - 1) * (dilations[i] - 1));
  }

  // Calculate input spatial shape with padding
  TensorShapeVector input_spatial_shape_with_pad;
  for (size_t i = 0; i < input_spatial_shape.size(); ++i) {
    input_spatial_shape_with_pad.push_back(input_spatial_shape[i] + adjust_pads[i] + adjust_pads[i + spatial_rank]);
  }

  // Calculate output spatial shape
  TensorShapeVector output_spatial_shape;
  for (size_t i = 0; i < input_spatial_shape_with_pad.size(); ++i) {
    output_spatial_shape.push_back(
        (input_spatial_shape_with_pad[i] - dilated_kernel_shape[i] + strides[i]) / strides[i]);
  }

  // Construct final output shape
  output_shape.clear();
  output_shape.push_back(batch_size);  // Batch size
  if (is_channel_last) {
    output_shape.insert(output_shape.end(), output_spatial_shape.begin(), output_spatial_shape.end());  // Spatial dimensions
    output_shape.push_back(out_channels);                                                               // Output channels
  } else {
    output_shape.push_back(out_channels);
    output_shape.insert(output_shape.end(), output_spatial_shape.begin(), output_spatial_shape.end());  // Spatial dimensions
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
  const bool is_channels_last = attributes.nchw;  // Assuming 'nchw' corresponds to 'NHWC' format

  // Calculate output shape
  TensorShapeVector output_shape;
  CalculateOutputShape(context, attributes, output_shape);

  // Get input tensors
  const Tensor* input_tensor = context.Input<Tensor>(0);                                      // Input tensor
  const Tensor* weight_tensor = context.Input<Tensor>(1);                                     // Weight tensor
  const Tensor* bias_tensor = context.InputCount() > 2 ? context.Input<Tensor>(2) : nullptr;  // Bias tensor (optional)

  if (attributes.group != 1) {
    // Handle grouped convolution
    std::vector<const Tensor*> conv_inputs = {input_tensor};
    if (is_channels_last) {
      if (!attributes.wT) {
        // Create and run the transpose program
        Tensor* output_tensor;
        std::vector<size_t> perm = {1, 0};                                       // Define permutation
        ORT_RETURN_IF_ERROR(RunTransposeProgram(context, perm, output_tensor));  // Use the new interface

        // Store the transposed weight in attributes.wT
        attributes.wT = output_tensor;
      }
      conv_inputs.push_back(*attributes.wT);
    } else {
      conv_inputs.push_back(weight_tensor);
    }
    if (bias_tensor) {
      conv_inputs.push_back(bias_tensor);
    }

    // Check if grouped convolution with vectorization is enabled
    std::string_view adaptor_arch = context.AdapterInfo().architecture;
    bool enable_grouped_conv_vectorize = adaptor_arch != "ampere";
    if (enable_grouped_conv_vectorize && is_channels_last &&
        weight_tensor->Shape()[0] == attributes.group && weight_tensor->Shape()[1] == 1 &&
        attributes.dilations[0] == 1 && attributes.dilations[1] == 1) {
      Tensor* output_tensor;
      ORT_RETURN_IF_ERROR(RunGroupedConvVectorizeProgram(
          context, conv_inputs, attributes, output_shape, output_tensor));
      return Status::OK();
    } else {
      Tensor* output_tensor;
      ORT_RETURN_IF_ERROR(RunGroupedConvProgram(
          context, conv_inputs, attributes, output_shape, output_tensor));
      return Status::OK();
    }
  }

  // Handle non-grouped convolution
  const bool has_bias = bias_tensor != nullptr;
  const int64_t input_height = input_tensor->Shape()[is_channels_last ? 1 : 2];
  const int64_t input_width = input_tensor->Shape()[is_channels_last ? 2 : 3];
  const int64_t input_channels = input_tensor->Shape()[is_channels_last ? 3 : 1];
  const int64_t weight_height = weight_tensor->Shape()[2];
  const int64_t weight_width = weight_tensor->Shape()[3];

  const int64_t out_height = output_shape[is_channels_last ? 1 : 2];
  const int64_t out_width = output_shape[is_channels_last ? 2 : 3];
  const int64_t out_channels = output_shape[is_channels_last ? 3 : 1];

  // Check if convolution can be optimized as matrix multiplication
  const bool same_size = is_channels_last &&
                         weight_height == input_height &&
                         weight_width == input_width &&
                         attributes.pads[0] == 0 &&
                         attributes.pads[1] == 0;
  const bool is1x1Conv = weight_height == 1 && weight_width == 1 &&
                         attributes.dilations[0] == 1 && attributes.dilations[1] == 1 &&
                         attributes.strides[0] == 1 && attributes.strides[1] == 1 &&
                         attributes.pads[0] == 0 && attributes.pads[1] == 0;

  if (same_size || is1x1Conv) {
    // Perform convolution using matrix multiplication
    const int64_t batch = output_shape[0];
    TensorShapeVector x_reshaped;
    TensorShapeVector w_reshaped;
    TensorShapeVector mat_mul_output_shape;
    std::vector<const TensorShape> mat_mul_inputs;

    if (is_channels_last) {
      if (!attributes.wT) {
        // Create and run the transpose program
        std::vector<size_t> perm = {1, 0};  // Define permutation
        Tensor* output_tensor;

        ORT_THROW_IF_ERROR(RunTransposeProgram(context, perm, output_tensor));  // Assuming permAttr = [1, 0]

        // Store the transposed weight in attributes.wT
        attributes.wT = output_tensor;
      }

      if (same_size) {
        const int64_t sharedDim = input_height * input_width * input_channels;
        x_reshaped = {1, batch, sharedDim};
        w_reshaped = {1, sharedDim, out_channels};
        mat_mul_output_shape = {1, batch, out_channels};
      } else {
        x_reshaped = {batch, input_height * input_width, input_channels};
        w_reshaped = {1, input_channels, out_channels};
        mat_mul_output_shape = {batch, out_height * out_width, out_channels};
      }
      mat_mul_inputs.push_back(x_reshaped);
      mat_mul_inputs.push_back(w_reshaped);
    } else {
      x_reshaped = {batch, input_channels, input_height * input_width};
      w_reshaped = {1, out_channels, input_channels};
      mat_mul_output_shape = {batch, out_channels, out_height * out_width};
      mat_mul_inputs.push_back(w_reshaped);
      mat_mul_inputs.push_back(x_reshaped);
    }

    if (has_bias) {
      mat_mul_inputs.push_back(bias_tensor->Shape().AsShapeVector());
    }

    const int64_t N = mat_mul_output_shape[2];
    const int64_t K = mat_mul_inputs[0].GetDims().back();
    if (N < 8 && K < 8) {
      ORT_RETURN_IF_ERROR(RunNaiveMatmulProgram(
          context, mat_mul_inputs, attributes, output_shape, mat_mul_output_shape, is_channels_last));
      return Status::OK();
    } else {
      ORT_RETURN_IF_ERROR(RunMatmulProgram(context, mat_mul_inputs, attributes, output_shape, &mat_mul_output_shape, is_channels_last));
      return Status::OK();
    }
  }

  // Default convolution using matrix multiplication
  const bool sequential_access_by_threads = false;  // Assuming Not intel backend

  // STEP 1: Transpose weight
  if (!attributes.wT) {
    Tensor* output_tensor;
    ORT_THROW_IF_ERROR(RunTransposeProgram(context, std::vector<std::size_t>({1, 0}), output_tensor));  // Assuming permAttr = [1, 0]

    // Store the transposed weight in attributes.wT
    attributes.wT = output_tensor;
  }

  // STEP 2: Prepare reshaped inputs
  std::vector<const Tensor*> conv_inputs = {input_tensor, *attributes.wT};
  if (has_bias) {
    conv_inputs.push_back(bias_tensor);
  }

  // STEP 3: Compute matmul
  const int64_t dim_a_outer = is_channels_last ? out_height * out_width : out_channels;
  const int64_t dim_b_outer = is_channels_last ? out_channels : out_height * out_width;
  const int64_t dim_inner = weight_height * weight_width * input_channels;

  Tensor* output_tensor;
  ORT_RETURN_IF_ERROR(RunConv2DMatMulProgram(
      context, conv_inputs, attributes, output_shape, dim_a_outer, dim_b_outer, dim_inner, has_bias, sequential_access_by_threads, output_tensor));
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

Status MatmulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Setup input/output variables
  const auto& a = shader.AddInput("a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  const auto& b = shader.AddInput("b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseUniform | ShaderUsage::UseShapeAndStride);

  if (attributes_.has_bias) {
    shader.AddInput("bias", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  }
  const auto& batch_dims = shader.AddInput("batchDims", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseIndicesTypeAlias |
                                                            ShaderUsage::UseOffsetToIndices);

  std::vector<const ShaderVariableHelper*> inputs = {&batch_dims, &a, &b, &output};
  const bool isVec4 = attributes_.components == 4;
  std::string workgroup_decl;
  std::string main_code;
  constexpr const bool kTransposeA = false;
  constexpr const bool kSplitK = false;
  constexpr const uint32_t kSplitedDimInner = 32;
  constexpr const uint32_t kTileInner = 32;
  shader.AdditionalImplementation() << GenerateMatMulReadWriteFnSource(inputs);
  if (isVec4) {
    // For vec4 implementation
    const uint32_t tile_a_outer = attributes_.workgroup_size[1] * attributes_.elements_per_thread[1];
    const uint32_t tile_b_outer = attributes_.workgroup_size[0] * attributes_.elements_per_thread[0];
    const uint32_t tile_a_width = kTransposeA ? tile_a_outer : kTileInner;
    const uint32_t tile_a_height = kTransposeA ? kTileInner : tile_a_outer;
    const uint32_t inner_element_size = tile_a_width / attributes_.workgroup_size[0];
    const uint32_t row_per_thread_b = kTileInner / attributes_.workgroup_size[1];

    // Generate workgroup shared memory declarations
    shader.AdditionalImplementation()
        << "var<workgroup> mm_Asub: array<array<vec" << inner_element_size
        << "<" << "input_value_t" << ">, " << tile_a_width / inner_element_size << ">, " << tile_a_height << ">;\n"
        << "var<workgroup> mm_Bsub: array<array<vec4<f32>, "
        << tile_b_outer / attributes_.elements_per_thread[0] << ">, " << 32 << ">;\n\n"
        << "const rowPerThread = " << attributes_.elements_per_thread[1] << ";\n"
        << "const colPerThread = " << attributes_.elements_per_thread[0] << ";\n"
        << "const innerElementSize = " << inner_element_size << ";\n"
        << "const tileInner = " << 32 << ";\n\n";

    // Generate main computation
    OStringStream& code = shader.MainFunctionBody();
    code << "let localRow = i32(localId.y);\n"
         << "let tileRow = localRow * rowPerThread;\n"
         << "let tileCol = i32(localId.x);\n\n"
         << "let globalRow = i32(globalId.y) * rowPerThread;\n"
         << "let globalCol = i32(globalId.x);\n";
    if (kSplitK) {
      code << "let batch = 0;\n";
    } else {
      code << "let batch = i32(globalId.z);\n";
    }

    if (batch_dims.Rank() > 0) {
      code << "let batchIndices = " << batch_dims.OffsetToIndices("u32(batch)") << ";\n";
    }

    code << "let globalRowStart = i32(workgroupId.y) * " << tile_a_outer << ";\n\n";

    if (kSplitK) {
      code << "let num_tiles = " << (kSplitedDimInner + kTileInner - 1) / kTileInner << ";\n"
           << "var kStart = i32(globalId.z) * " << kSplitedDimInner << ";\n\n";
    } else {
      code << "let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
           << "var kStart = 0;\n\n";
    }
    code << "var acc: array<vec4<" << "input_value_t" << ">, " << "rowPerThreadB" << ">;\n\n"
         << "let tileRowB = localRow * " << row_per_thread_b << ";\n";

    // Generate tile loading and computation loops
    code << "for (var t = 0; t < num_tiles; t = t + 1) {\n"
         << "    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
         << "        let inputRow = tileRow + innerRow;\n"
         << "        let inputCol = tileCol;\n"
         << WriteDataToSubAVec4Snippet(kTransposeA, batch_dims) << "\n"
         << "    }\n\n"
         << "    for (var innerRow = 0; innerRow < " << row_per_thread_b << "; innerRow = innerRow + 1) {\n"
         << "        let inputRow = tileRowB + innerRow;\n"
         << "        let inputCol = tileCol;\n"
         << "        mm_Bsub[inputRow][inputCol] = "
         << GenerateReadBCode("batch", "kStart + inputRow", "globalCol", batch_dims) << ";\n"
         << "    }\n"
         << "    kStart = kStart + tileInner;\n"
         << "    workgroupBarrier();\n\n"
         << GenerateComputationLoop(kTransposeA, inner_element_size)
         << "    workgroupBarrier();\n"
         << "}\n\n"
         << GenerateOutputCode();

  } else {
    // For standard implementation
    const uint32_t tile_a_outer = attributes_.elements_per_thread[1] * attributes_.workgroup_size[1];
    const uint32_t tile_b_outer = attributes_.elements_per_thread[0] * attributes_.workgroup_size[0];
    const uint32_t tile_a_width = kTransposeA ? tile_a_outer : kTileInner;
    const uint32_t tile_a_height = kTransposeA ? kTileInner : tile_a_outer;
    const uint32_t row_per_thread_a = tile_a_height / attributes_.workgroup_size[1];
    const uint32_t col_per_thread_a = tile_a_width / attributes_.workgroup_size[0];
    const uint32_t row_per_thread_b = kTileInner / attributes_.workgroup_size[1];

    // Generate workgroup shared memory declarations
    shader.AdditionalImplementation()
        << "var<workgroup> mm_Asub: array<array<" << "input_value_t" << ", " << tile_a_width << ">, " << tile_a_height << ">;\n"
        << "var<workgroup> mm_Bsub: array<array<" << "input_value_t" << ", " << tile_b_outer << ">, " << kTileInner << ">;\n"
        << "const rowPerThread = " << attributes_.elements_per_thread[1] << ";\n"
        << "const colPerThread = " << attributes_.elements_per_thread[0] << ";\n"
        << "const tileInner = " << kTileInner << ";\n\n";

    // Generate standard matmul implementation
    OStringStream& code = shader.MainFunctionBody();
    code << "let tileRow = i32(localId.y) * rowPerThread;\n"
         << "let tileCol = i32(localId.x) * colPerThread;\n\n"
         << "let globalRow = i32(globalId.y) * rowPerThread;\n"
         << "let globalCol = i32(globalId.x) * colPerThread;\n"
         << "let globalRowStart = i32(workgroupId.y) * " << tile_a_outer << ";\n\n"
         << "let tileRowA = i32(localId.y) * " << row_per_thread_a << ";\n"
         << "let tileColA = i32(localId.x) * " << col_per_thread_a << ";\n"
         << "let tileRowB = i32(localId.y) * " << row_per_thread_b << ";\n\n"
         << "let batch = i32(globalId.z);\n";

    if (batch_dims.Rank() > 0) {
      code << "let batchIndices = " << batch_dims.OffsetToIndices("u32(batch)") << ";\n";
    }

    code << "let num_tiles = (uniforms.dim_inner - 1) / tileInner + 1;\n"
         << "var kStart = 0;\n\n"
         << "var acc: array<array<" << "input_value_t" << ", " << attributes_.elements_per_thread[0] << ">, "
         << attributes_.elements_per_thread[1] << ">;\n\n"
         << GenerateStandardMatmulLoop(kTransposeA, row_per_thread_a, col_per_thread_a, row_per_thread_b, batch_dims);
  }

  return Status::OK();
}

std::string MatmulProgram::GenerateReadBCode(const std::string& batch, const std::string& row,
                                             const std::string& col, const ShaderIndicesHelper& batchDims) const {
  std::stringstream code;
  code << "mm_readB(" << batch << ", " << row << ", " << col
       << (batchDims.Rank() > 0 ? ", batchIndices" : "") << ")";
  return code.str();
}

std::string MatmulProgram::GenerateComputationLoop(bool transposeA, uint32_t inner_element_size) const {
  std::stringstream code;
  code << "    for (var k = 0; k < tileInner; k = k + 1) {\n"
       << "        let BCached0 = mm_Bsub[k][globalCol / 4];\n\n"
       << "        let BCached1 = mm_Bsub[k * innerElementSize + 1][tileCol];\n\n"
       << "        let BCached2 = mm_Bsub[k * innerElementSize + 2][tileCol];";
  if (inner_element_size == 3) {
    code << "        let BCached3 = mm_Bsub[k * innerElementSize + 3][tileCol];\n";
  }

  if (transposeA) {
    code << "        let ACached0 = mm_Asub[k * innerElementSize][localRow];\n"
         << "        let ACached1 = mm_Asub[k * innerElementSize + 1][localRow];\n"
         << "        let ACached2 = mm_Asub[k * innerElementSize + 2][localRow];\n";
    if (inner_element_size == 3) {
      code << "        let ACached3 = mm_Asub[k * innerElementSize + 3][localRow];\n";
    }

    code << "        for (var i = 0; i < rowPerThread; i = i + 1) {\n"
         << "          acc[i] = fma(BCached0, ACached0[i], acc[i]);\n"
         << "          acc[i] = fma(BCached0, ACached1[i], acc[i]);\n"
         << "          acc[i] = fma(BCached0, ACached2[i], acc[i]);\n";
    if (inner_element_size == 3) {
      code << "          acc[i] = fma(BCached0, ACached3[i], acc[i]);\n";
    }
    code << "        }\n";
  } else {
    code << "        for (var i = 0; i < rowPerThread; i = i + 1) {\n"
         << "          let ACached = mm_Asub[tileRow + i][k];\n"
         << "          acc[i] = fma(BCached0, ACached.x, acc[i]);\n"
         << "          acc[i] = fma(BCached0, ACached.y, acc[i]);\n"
         << "          acc[i] = fma(BCached0, ACached.z, acc[i]);\n";
    if (inner_element_size == 3) {
      code << "          acc[i] = fma(BCached0, ACached.w, acc[i]);\n";
    }
    code << "        }\n";
  }

  code << "    }\n";
  return code.str();
}

std::string MatmulProgram::GenerateOutputCode() const {
  return "for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
         "    mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);\n"
         "}\n";
}

std::string MatmulProgram::GenerateStandardMatmulLoop(bool transpose_a, uint32_t row_per_thread_a,
                                                      uint32_t col_per_thread_a, uint32_t row_per_thread_b, const ShaderIndicesHelper& batch_dims) const {
  std::stringstream code;
  code << "for (var t = 0; t < num_tiles; t = t + 1) {\n"
       // Load tile of A
       << "  for (var innerRow = 0; innerRow < " << row_per_thread_a << "; innerRow = innerRow + 1) {\n"
       << "    for (var innerCol = 0; innerCol < " << col_per_thread_a << "; innerCol = innerCol + 1) {\n"
       << "      let inputRow = tileRowA + innerRow;\n"
       << "      let inputCol = tileColA + innerCol;\n"
       << "      " << WriteDataToSubASnippet(transpose_a, batch_dims) << "\n"
       << "    }\n"
       << "  }\n\n"
       // Load tile of B
       << "  for (var innerRow = 0; innerRow < " << row_per_thread_b << "; innerRow = innerRow + 1) {\n"
       << "    for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
       << "      let inputRow = tileRowB + innerRow;\n"
       << "      let inputCol = tileCol + innerCol;\n"
       << "      mm_Bsub[inputRow][inputCol] = " << GenerateReadBCode("batch", "kStart + inputRow", "globalCol + innerCol", batch_dims) << ";\n"
       << "    }\n"
       << "  }\n"
       << "  kStart = kStart + tileInner;\n"
       << "  workgroupBarrier();\n\n"
       // Compute accumulation
       << "  var BCached: array<" << "input_value_t" << ", colPerThread>;\n"
       << "  for (var k = 0; k < tileInner; k = k + 1) {\n"
       << "    for (var inner = 0; inner < colPerThread; inner = inner + 1) {\n"
       << "      BCached[inner] = mm_Bsub[k][tileCol + inner];\n"
       << "    }\n\n"
       << "    for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
       << "      let ACached = mm_Asub[" << (transpose_a ? "k" : "tileRow + innerRow") << "]["
       << (transpose_a ? "tileRow + innerRow" : "k") << "];\n"
       << "      for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
       << "        acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];\n"
       << "      }\n"
       << "    }\n"
       << "  }\n\n"
       << "  workgroupBarrier();\n"
       << "}\n\n"
       // Write output
       << "for (var innerRow = 0; innerRow < rowPerThread; innerRow = innerRow + 1) {\n"
       << "  for (var innerCol = 0; innerCol < colPerThread; innerCol = innerCol + 1) {\n"
       << "    mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);\n"
       << "  }\n"
       << "}\n";

  return code.str();
}

std::string MatmulProgram::WriteDataToSubAVec4Snippet(bool transposeA, const ShaderIndicesHelper& batchDims) const {
  std::stringstream code;
  if (transposeA) {
    code << "mm_Asub[inputRow][inputCol] = mm_readA(batch, kStart + inputRow, "
         << "globalRowStart / innerElementSize + inputCol" << (batchDims.Rank() > 0 ? ", batchIndices" : "") << ");";
  } else {
    code << "mm_Asub[inputRow][inputCol] = mm_readA(batch, globalRowStart + inputRow, "
         << "kStart / innerElementSize + inputCol" << (batchDims.Rank() > 0 ? ", batchIndices" : "") << ");";
  }
  return code.str();
}

std::string MatmulProgram::WriteDataToSubASnippet(bool transposeA, const ShaderIndicesHelper& batchDims) const {
  std::stringstream code;
  if (transposeA) {
    code << "mm_Asub[inputRow][inputCol] = mm_readA(batch, "
         << "kStart + inputRow, "
         << "globalRowStart + inputCol"
         << (batchDims.Rank() > 0 ? ", batchIndices" : "")
         << ");";
  } else {
    code << "mm_Asub[inputRow][inputCol] = mm_readA(batch, "
         << "globalRowStart + inputRow, "
         << "kStart + inputCol"
         << (batchDims.Rank() > 0 ? ", batchIndices" : "")
         << ");";
  }
  return code.str();
}

std::string MatmulProgram::GenerateMatMulReadWriteFnSource(const std::vector<const ShaderVariableHelper*>& variables) const {
  // Get variables from vector instead of lookup
  const auto& batch_dims = *variables[0];  // Batch dimensions
  const auto& a = *variables[1];           // Matrix A
  const auto& b = *variables[2];           // Matrix B
  const auto& output = *variables[3];      // Output

  const std::string type_snippet = (attributes_.components > 1)
                                       ? absl::StrCat("vec", attributes_.components, "<", "output_value_t", ">")
                                       : "output_value_t";
  const std::string zero_value = absl::StrCat(type_snippet, "(0.0)");

  // Generate read/write function code
  std::ostringstream code;

  // Generate mm_readA function
  code << "fn mm_readA(batch: i32, row: i32, colIn: i32"
       << (batch_dims.Rank() > 0 ? ", batchIndices: batchDims_indices_t" : "")
       << ") -> " << type_snippet << " {\n"
       << "  var value = " << zero_value << ";\n"
       << "  let col = colIn * " << attributes_.components << ";\n"
       << "  if(row < uniforms.dim_a_outer && col < uniforms.dim_inner) {\n"
       << "    var aIndices: " << "a_indices_t" << ";\n"
       << ConvertOutputBatchIndicesToInputBatchIndices("aIndices", a, a.Rank() - 2, batch_dims.Rank(), "batchIndices")
       << "    " << a.IndicesSet("aIndices", a.Rank() - 2, "u32(row)") << "\n"
       << "    " << a.IndicesSet("aIndices", a.Rank() - 1, "u32(colIn)") << "\n"
       << "    value = " << a.GetByIndices("aIndices") << ";\n"
       << "  }\n"
       << "  return value;\n"
       << "}\n\n";

  // Generate mm_readB function
  code << "fn mm_readB(batch: i32, row: i32, colIn: i32"
       << (batch_dims.Rank() > 0 ? ", batchIndices: batchDims_indices_t" : "")
       << ") -> " << type_snippet << " {\n"
       << "  var value = " << zero_value << ";\n"
       << "  let col = colIn * " << attributes_.components << ";\n"
       << "  if(row < uniforms.dim_inner && col < uniforms.dim_b_outer) {\n"
       << "    var bIndices: " << "b_indices_t" << ";\n"
       << ConvertOutputBatchIndicesToInputBatchIndices("bIndices", b, b.Rank() - 2, batch_dims.Rank(), "batchIndices")
       << "    " << b.IndicesSet("bIndices", b.Rank() - 2, "u32(row)") << "\n"
       << "    " << b.IndicesSet("bIndices", b.Rank() - 1, "u32(colIn)") << "\n"
       << "    value = " << b.GetByIndices("bIndices") << ";\n"
       << "  }\n"
       << "  return value;\n"
       << "}\n\n";

  // Generate mm_write function
  code << "fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: " << type_snippet << ") {\n"
       << "  let col = colIn * " << attributes_.components << ";\n"
       << "  if (row < uniforms.dim_a_outer && col < uniforms.dim_b_outer) {\n"
       << "    var value = valueIn;\n"
       << "    let coords = vec3<i32>(batch, row, colIn);\n";

  // Add bias if needed
  if (attributes_.has_bias) {
    if (attributes_.is_channels_last) {
      code << "    value = value + bias[colIn];\n";
    } else {
      code << "    value = value + " << type_snippet << "(bias[row]);\n";
    }
  }

  // Add activation
  const std::string apply_activation = GetActivationSnippet(attributes_.activationAttributes, type_snippet, "output_value_t");
  code << "    " << apply_activation << "\n"
       << "    " << output.SetByIndices("vec3<u32>(coords)", "value") << "\n"
       << "  }\n"
       << "}\n";

  return code.str();
}
}  // namespace webgpu
}  // namespace onnxruntime
