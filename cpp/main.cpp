#include "ImageReader.hpp"
#include "VariationalAutoEncoder.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <memory>

std::unique_ptr<float[]> prepare_data(ImageReader &imgr, std::string &filename) {
  imgr.readPNG(filename);
  std::vector<unsigned char> &data = imgr.getData();

  std::unique_ptr<float[]> normalized_data(new float[data.size()]);

  // Normalize the data and store it in the unique pointer
  for (size_t i = 0; i < data.size(); i++) {
    normalized_data[i] = static_cast<float>(data[i]) / 255.0f;
  }

  return normalized_data;
}

void export_new_data(ImageReader &imgr, float *out_data, std::string filename) {
  std::vector<unsigned char> &data = imgr.getData();

  // Normalize the data and store it in the unique pointer
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = static_cast<unsigned char>(out_data[i]*255);
  }

  imgr.writePNG(filename);
}

int main(int argc, char *argv[]) {

  std::string weights_file = "example.json";
  std::string input_image = "example.png";

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--weights" || arg == "-w") {
      if (i + 1 < argc) {
        weights_file = argv[i + 1];
        i++;
      }
    } else if (arg == "--input" || arg == "-i") {
      if (i + 1 < argc) {
        input_image = argv[i + 1];
        i++;
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

  std::unique_ptr<float[]> data_in = prepare_data(imgr, input_image);
  float data_out[imgr.getData().size()];
  printf("forward\n");
  vae.forward(data_in, data_out);
  printf("OK!\n");
  export_new_data(imgr, data_out, "test.png");


  return 0;
}
