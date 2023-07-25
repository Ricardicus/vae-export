#include "nlohmann/json.hpp"
#include <memory>
#include <string>

class VariationalAutoEncoder {
public:
  VariationalAutoEncoder(int input_dim, int h_dim = 200, int z_dim = 20);
  VariationalAutoEncoder(){};
  void encode(std::unique_ptr<float[]> &x);
  void decode(std::unique_ptr<float[]> &z);
  void forward(std::unique_ptr<float[]> &x, float *out);

  void get_encoded(std::unique_ptr<float[]> &mu,
                   std::unique_ptr<float[]> &sigma);
  void get_decoded(std::unique_ptr<float[]> &z);

  int get_input_dim() const { return this->input_dim; };

  void load_weights(std::string file_path = "model_weights.pth");

private:
  void relu(std::unique_ptr<float[]> &x, int size);
  void sigmoid(std::unique_ptr<float[]> &x, int size);
  void get_element_size(nlohmann::json &json, std::string field, int &size1,
                        int &size2);
  void load_new(int input_dim, int h_dim, int z_dim);
  void load_element(std::unique_ptr<float[]> &load, nlohmann::json &json,
                    std::string field, int size1, int size2);
  void generate_latent();

  int input_dim, h_dim, z_dim;
  std::unique_ptr<float[]> img_2hid;
  std::unique_ptr<float[]> hid_2mu;
  std::unique_ptr<float[]> hid_2sigma;
  std::unique_ptr<float[]> z_2hid;
  std::unique_ptr<float[]> hid_2img;
  std::unique_ptr<float[]> hid_1;
  std::unique_ptr<float[]> hid_2;
  std::unique_ptr<float[]> mu;
  std::unique_ptr<float[]> sigma;
  std::unique_ptr<float[]> out;
  std::unique_ptr<float[]> epsilon;
  std::unique_ptr<float[]> z;
};
