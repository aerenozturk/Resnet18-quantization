#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/classification.hpp>
#include <vitis/ai/demo.hpp>

#include "./process_result.hpp"

int main(int argc, char* argv[]) {
  std::string model = argv[1];
  std::string input_image_path = argv[2];
  
  // Extract the image filename without path
  std::string filename = input_image_path.substr(input_image_path.find_last_of("/") + 1);
  std::string output_image_path = "/home/root/Vitis-AI/examples/vai_library/samples/classification/test_results/" + filename + "_result.jpg";
  std::string log_file_path = "/home/root/Vitis-AI/examples/vai_library/samples/classification/test_results/" + filename + "_log.txt";

  auto classification = vitis::ai::Classification::create(model);

  // Load the input image
  cv::Mat image = cv::imread(input_image_path);
  if (image.empty()) {
    std::cerr << "Error: Could not load image." << std::endl;
    return -1;
  }

  // Run inference
  auto results = classification->run(image);

  // Call the process_result function with dynamic output filename and log results
  process_result(image, results, true, output_image_path, log_file_path);

  std::cout << "Result image saved as " << output_image_path << std::endl;
  return 0;
}

