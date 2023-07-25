#include "ImageReader.hpp"
#include "VariationalAutoEncoder.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <memory>

std::unique_ptr<float[]> prepare_data(ImageReader &imgr, int input_size,
                                      std::string filename) {
  imgr.readPNG(filename);
  std::vector<unsigned char> &data = imgr.getData();

  int channels = data.size() / input_size;
  std::unique_ptr<float[]> normalized_data(new float[data.size() / channels]);

  // Normalize the data and store it in the unique pointer
  for (size_t i = 0; i < data.size() / channels; i++) {
    float u = static_cast<float>(data[i * channels]) / 255.0f;
    normalized_data[i] = u;
  }

  return normalized_data;
}

void export_new_data(ImageReader &imgr, float *out_data, size_t size,
                     std::string filename) {
  std::vector<unsigned char> &data = imgr.getData();

  // Normalize the data and store it in the unique pointer
  for (size_t i = 0; i < size; i++) {
    data[i] = static_cast<unsigned char>(out_data[i] * 255);
  }

  imgr.writePNG(filename);
}

int main(int argc, char *argv[]) {

  std::string weights_file = "example.json";
  std::vector<std::string>
      input_images; // Create a vector to store input images

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--weights" || arg == "-w") {
      if (i + 1 < argc) {
        weights_file = argv[i + 1];
        i++;
      }
    } else if (arg == "--input" || arg == "-i") {
      // Check if there are more arguments after "--input" flag
      if (i + 1 < argc) {
        // Loop through the remaining arguments until a new flag is encountered
        // or end of command line arguments
        for (int j = i + 1; j < argc; j++) {
          std::string input_arg = argv[j];
          // Check if the argument is a flag
          if (input_arg.substr(0, 2) == "--" || input_arg.substr(0, 1) == "-") {
            break; // Break the loop if a new flag is encountered
          } else {
            input_images.push_back(
                input_arg); // Add the input argument to the vector
            i++;            // Increment i to skip the added input argument
          }
        }
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "usage: " << argv[0]
                << " --weights weights.json --input input.png" << std::endl;
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }
  VariationalAutoEncoder vae;
  ImageReader imgr;
  vae.load_weights(weights_file);
  int input_size = vae.get_input_dim();

  if (input_images.size() == 0) {
    std::cerr << "error: no input files given with --input" << std::endl;
    return 1;
  }
  // Process the input images
  for (const std::string &input_image : input_images) {
    std::unique_ptr<float[]> data_in =
        prepare_data(imgr, input_size, input_image);
    float data_out[vae.get_input_dim()];

    vae.forward(data_in, data_out);
    export_new_data(imgr, data_out, input_size, "generated_" + input_image);
  }

  return 0;
}
