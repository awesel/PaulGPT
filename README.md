## PaulGPT

This contains a few fine-tuned Gemma models intended to sound like Paul Graham. It took me two tries to get a reasonable result. Data processing and training cost about $5 in OpenAI requests and H100 time.

Round 1: Fail! Overfit, catastrophic forgetting. Fine-tuned from Gemma-7B

Round 2: Success! Pretty reasonable result. Adopts more casual structure and some of Paul's opinions and writing style. Fine-tuned from Gemma-3-4B

You can read my full narrative about the process of making this model on my blog awesel.com

## How to generate tokens with this model
- git clone https://github.com/awesel/PaulGPT
- cd PaulGPT
- huggingface-cli login (enter your huggingface access token here)
- huggingface download https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/tree/main
- Set your paths in inference.py. The LoRA path should go to the Round-2-F16-LoRA.gguf file, and the model file should go to the .gguf file you just downloaded.
- cd inference
- python inference.py
Enjoy! The model is tiny and runs at 10-15 tok/second on my Macbook Pro with 24 gb RAM. If inference doesn't work, try not loading both models into memory at the same time.

Please reach out if generate anything funny using this repo! awesel [at] stanford [dot] edu

