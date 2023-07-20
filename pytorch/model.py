import torch
from torch import nn

# Input img -> hidden dim -> mean, std -> parametrization trick -> decode -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma

    def save_weights(self, file_path='model_weights.pth'):
        torch.save(self.state_dict(), file_path)

    def load_weights(self, file_path='model_weights.pth'):
        self.load_state_dict(torch.load(file_path))

    def print_weights(self):
        print("Encoder:")
        print(f"self.img_2hid: {self.img_2hid.weight.shape}")
        print(self.img_2hid.weight)
        print("self.hid_2mu:")
        print(f"{self.hid_2mu.weight.shape}")
        print(self.hid_2mu.weight)
        print("self.hid_2sigma")
        print(f"{self.hid_2sigma.weight.shape}")
        print(self.hid_2sigma.weight)
        print("Decoder:")
        print("z_2hid:")
        print(f"{self.z_2hid.weight.shape}")
        print(self.z_2hid.weight)
        print("hid_2img:")
        print(f"{self.hid_2img.weight.shape}")
        print(self.hid_2img.weight)

    def _print_weight_2_dim(self, module, file):
        file.write("[")
        weight = module.weight
        for r in range(weight.shape[0]):
            if r > 0:
                file.write(",")
            file.write("[")
            for c in range(weight.shape[1]):
                if c > 0:
                    file.write(",")
                file.write(f"{weight[r][c].item()}")
            file.write("]")
        file.write("]")

    def convert_to_json(self, file="out_weights.json"):
        
        with open(file, "w") as f:
            f.write('{')
            f.write('"img_2hid":')
            self._print_weight_2_dim(self.img_2hid, f)
            f.write(',')

            # Add missing weights to JSON file
            f.write('"hid_2mu":')
            self._print_weight_2_dim(self.hid_2mu, f)
            f.write(',')

            f.write('"hid_2sigma":')
            self._print_weight_2_dim(self.hid_2sigma, f)
            f.write(',')

            f.write('"z_2hid":')
            self._print_weight_2_dim(self.z_2hid, f)
            f.write(',')

            f.write('"hid_2img":')
            self._print_weight_2_dim(self.hid_2img, f)

            f.write('}')

if __name__ == "__main__":
    input_dim = 28*28
    x = torch.randn(4, input_dim)
    vae = VariationalAutoEncoder(input_dim=input_dim)
    x_reconstructed, mu, sigma = vae(x) 
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)

