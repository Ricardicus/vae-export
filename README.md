# Proof of concept: weight distribution

I want to build a neural network with Pytorch and export its weights.
I do this by building a neural network module, a variational auto encoder, and
then export its weights by inspecting the tensors content into a json file.

I then build a Cpp program that parses this json file and can run inference.

## Building the model

```bash
cd pytorch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
python export.py --weights weights.pth
```

## Loading it into cpp

```bash
cd cpp
make
cp ../pytorch/weights.json .
./main weights.json
```
# Inference 

After running 20 epochs with the standard traning configuration on MNIST:

![Image 0 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_0_ex0.png)
![Image 0 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_0_ex0.png)

![Image 1 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_1_ex0.png)
![Image 1 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_1_ex0.png)

![Image 2 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_2_ex0.png)
![Image 2 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_2_ex0.png)

![Image 3 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_3_ex0.png)
![Image 3 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_3_ex0.png)

![Image 4 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_4_ex0.png)
![Image 4 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_4_ex0.png)

![Image 5 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_5_ex0.png)
![Image 5 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_5_ex0.png)

![Image 6 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_6_ex0.png)
![Image 6 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_6_ex0.png)

![Image 7 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_7_ex0.png)
![Image 7 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_7_ex0.png)

![Image 8 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_8_ex0.png)
![Image 8 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_8_ex0.png)

![Image 9 conditioning](https://github.com/Ricardicus/vae-export/blob/master/media/generated_9_ex0.png)
![Image 9 generated](https://github.com/Ricardicus/vae-export/blob/master/media/incoming_9_ex0.png)

# Inference video from C++ code

![Image 9 generated](https://github.com/Ricardicus/vae-export/blob/master/media/output.gif)


