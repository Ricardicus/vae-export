/* Example CPP program that can open a JSON file */
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include "VariationalAutoEncoder.hpp"

int main(int argc, char *argv[]) {

  std::string file = "example.json";
  if (argc > 1) {
    file = std::string(argv[1]);
  } else {
    std::cerr << "usage: " << argv[0] << ": weights-file.json" << std::endl;
    return 1;
  }
  VariationalAutoEncoder vae;
  
  vae.load_weights(file);

  return 0;
}
