#include "main_functions.h"
#include "pico/stdlib.h"

#include <stdio.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/benchmarks/micro_benchmark.h"
#include <climits>

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 134992;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {

  stdio_init_all();
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void infer() {
    // Calculate an x value to feed into the model. We compare the current
    // inference_count to the number of inferences per cycle to determine
    // our position within the range of possible x values the model was
    // trained on, and use this to calculate a value.
    std::srand(167);
    TfLiteTensor* input_tensor = input;

    // Pre-populate input tensor with random values.
    int input_length = input->bytes / sizeof(int8_t);
    int8_t* input_values = tflite::GetTensorData<int8_t>(input_tensor);
    for (int i = 0; i < input_length; i++) {
        // Pre-populate input tensor with a random value based on a constant seed.
        input_values[i] = static_cast<int8_t>(
            std::rand() % (std::numeric_limits<int8_t>::max() -
                            std::numeric_limits<int8_t>::min() + 1));
    }
    int32_t start_ticks;                         
    int32_t duration_ticks;                      
    int32_t duration_ms;
    start_ticks = tflite::GetCurrentTimeTicks(); 
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x:\n");
      return;
    }
    duration_ticks = tflite::GetCurrentTimeTicks() - start_ticks;
    if (duration_ticks > INT_MAX / 1000) {                                
      duration_ms = duration_ticks / (tflite::ticks_per_second() / 1000);
    } else {                                                             
      duration_ms = (duration_ticks * 1000) / tflite::ticks_per_second();
    }
    TF_LITE_REPORT_ERROR(error_reporter, "Duration took %d ticks for %dms\n", duration_ticks, duration_ms);
    printf("Quantized output: \n");
    for (int i = 0; i < output->dims->size; i++) {    
      // Obtain the quantized output from model's output tensor
      int8_t y_quantized = output->data.int8[i];
      TF_LITE_REPORT_ERROR(error_reporter,"%d\n", y_quantized);
    }
    printf("Dequantized output: \n");
    for (int i = 0; i < output->dims->size; i++) {
      // Dequantize the output from integer to floating-point
      int8_t y_quantized = output->data.int8[i];
      float y = (y_quantized - output->params.zero_point) * output->params.scale;
      TF_LITE_REPORT_ERROR(error_reporter, "%6f\n", y);
    }
    // Output the results. A custom HandleOutput function can be implemented
    // for each supported hardware target.
    // HandleOutput(error_reporter, x, y);

    // Increment the inference_counter, and reset it if we have reached
    // the total number per cycle
    inference_count += 1;
    if (inference_count >= kInferencesPerCycle) inference_count = 0;
}