#include "VariationalAutoEncoder.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <string>

VariationalAutoEncoder::VariationalAutoEncoder(int input_dim, int h_dim,
                                               int z_dim) {
  this->load_new(input_dim, h_dim, z_dim);
}

void VariationalAutoEncoder::relu(std::unique_ptr<float[]> &x, int size) {
  for (int i = 0; i < size; i++) {
    if (x[i] < 0) {
      x[i] = 0;
    }
  }
}

void VariationalAutoEncoder::sigmoid(std::unique_ptr<float[]> &x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = 1.0 / (1.0 + exp(-x[i]));
  }
}

void VariationalAutoEncoder::encode(std::unique_ptr<float[]> &x) {
  // Compute hidden layer
  for (int o = 0; o < this->h_dim; o++) {
    this->hid_1[o] = 0;
    for (int i = 0; i < this->input_dim; i++) {
      this->hid_1[o] += this->img_2hid[o * input_dim + i] * x[i];
    }
  }
  this->relu(this->hid_1, this->h_dim);
  // Compute mu, sigma
  for (int o = 0; o < this->z_dim; o++) {
    this->mu[o] = 0;
    this->sigma[o] = 0;
    for (int i = 0; i < this->h_dim; i++) {
      this->mu[o] += this->hid_2mu[o * this->h_dim + i] * this->hid_1[i];
      this->sigma[o] += this->hid_2sigma[o * this->h_dim + i] * this->hid_1[i];
    }
  }
}

void VariationalAutoEncoder::get_encoded(std::unique_ptr<float[]> &mu,
                                         std::unique_ptr<float[]> &sigma) {
  for (int i = 0; i < this->z_dim; i++) {
    mu[i] = this->mu[i];
    sigma[i] = this->sigma[i];
  }
}
void VariationalAutoEncoder::get_decoded(std::unique_ptr<float[]> &out) {
  for (int i = 0; i < this->input_dim; i++) {
    out[i] = this->out[i];
  }
}
void VariationalAutoEncoder::decode(std::unique_ptr<float[]> &z) {
  // Compute hidden layer
  for (int o = 0; o < this->h_dim; o++) {
    this->hid_2[o] = 0;
    for (int i = 0; i < this->z_dim; i++) {
      this->hid_2[o] += this->z_2hid[o * z_dim + i] * z[i];
    }
  }
  this->relu(this->hid_2, this->h_dim);

  // Compute output
  for (int o = 0; o < this->input_dim; o++) {
    this->out[o] = 0;
    for (int i = 0; i < this->h_dim; i++) {
      this->out[o] += this->hid_2img[o * this->h_dim + i] * this->hid_2[i];
    }
  }
  this->sigmoid(this->out, this->input_dim);
}

void VariationalAutoEncoder::generate_latent() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(0, 1);

  for (int i = 0; i < this->z_dim; i++) {
    this->epsilon[i] = d(gen);
  }
}

void VariationalAutoEncoder::forward(std::unique_ptr<float[]> &x,
                                     float *out) {
  // encode
  this->encode(x);

  // generate latent space point
  this->generate_latent();

  // compute z
  for (int i = 0; i < this->z_dim; i++) {
    this->z[i] = this->mu[i] + this->sigma[i] * this->epsilon[i];
  }

  // compute output
  this->decode(this->z);

  // forward the output
  for (int i = 0; i < this->input_dim; i++) {
    out[i] = this->out[i];
  }
}

void VariationalAutoEncoder::load_element(std::unique_ptr<float[]> &load,
                                          nlohmann::json &json,
                                          std::string field, int size1,
                                          int size2) {
  std::vector<float> values;
  int dim1 = 0;
  for (auto &element : json[field]) {
    dim1 += 1;
    int dim2 = 0;
    for (auto &number : element) {
      dim2 += 1;
      values.push_back(number);
    }
    assert(dim2 == size2);
  }
  assert(dim1 == size1);
  assert(static_cast<size_t>(size1 * size2) == values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    load[i] = values[i];
  }
}

void VariationalAutoEncoder::get_element_size(nlohmann::json &json,
                                              std::string field, int &size1,
                                              int &size2) {
  int dim1 = 0;
  int dim2 = 0;
  for (auto &element : json[field]) {
    dim1 += 1;
    dim2 = 0;
    for (auto &_ : element) {
      (void)_;
      dim2 += 1;
    }
  }

  size1 = dim1;
  size2 = dim2;
}

void VariationalAutoEncoder::load_new(int input_dim, int h_dim, int z_dim) {
  // Dimensions
  this->input_dim = input_dim;
  this->h_dim = h_dim;
  this->z_dim = z_dim;

  // Encoder
  this->img_2hid = std::make_unique<float[]>(input_dim * h_dim);
  this->hid_2mu = std::make_unique<float[]>(h_dim * z_dim);
  this->hid_2sigma = std::make_unique<float[]>(h_dim * z_dim);

  // Decoder
  this->z_2hid = std::make_unique<float[]>(z_dim * h_dim);
  this->hid_2img = std::make_unique<float[]>(h_dim * input_dim);

  // Internal outputs
  this->sigma = std::make_unique<float[]>(z_dim);
  this->mu = std::make_unique<float[]>(z_dim);
  this->hid_1 = std::make_unique<float[]>(h_dim);
  this->hid_2 = std::make_unique<float[]>(h_dim);

  // Output
  this->out = std::make_unique<float[]>(input_dim);

  // Latent space
  this->epsilon = std::make_unique<float[]>(z_dim);
  this->z = std::make_unique<float[]>(z_dim);
}

void VariationalAutoEncoder::load_weights(std::string file_path) {
  // Open the JSON file
  std::ifstream jsonFile(file_path);

  // Check if the file is open
  if (!jsonFile.is_open()) {
    std::cerr << "Error opening the file: " << file_path << std::endl;
    return;
  }

  // Parse the JSON file
  nlohmann::json j;
  jsonFile >> j;

  // Close the file
  jsonFile.close();

  // Investigate dimensions
  int input_dim, z_dim, h_dim;
  this->get_element_size(j, "img_2hid", h_dim, input_dim);
  this->get_element_size(j, "z_2hid", h_dim, z_dim);

  this->load_new(input_dim, h_dim, z_dim);
  // Load encoder
  this->load_element(this->img_2hid, j, "img_2hid", this->h_dim,
                     this->input_dim);
  this->load_element(this->hid_2mu, j, "hid_2mu", this->z_dim, this->h_dim);
  this->load_element(this->hid_2sigma, j, "hid_2sigma", this->z_dim,
                     this->h_dim);

  // Load decoder
  this->load_element(this->z_2hid, j, "z_2hid", this->h_dim, this->z_dim);
  this->load_element(this->hid_2img, j, "hid_2img", this->input_dim,
                     this->h_dim);
}
