import argparse
import torch
from model import VariationalAutoEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file')
    args = parser.parse_args()

    filename = args.weights
    model = VariationalAutoEncoder(784)
    model.load_weights(filename)

    model.print_weights()
    model.convert_to_json(filename.split(".")[0] + ".json")
    """
    sm = torch.jit.script(model)

    outfile = filename.split(".")[0] + ".pt"
    print(f"Storing {outfile}")
    sm.save(outfile)
    """
