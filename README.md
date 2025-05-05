## PaulGPT

This contains a few fine-tuned Gemma models intended to sound like Paul Graham. It took me two tries to get a reasonable result. Data processing and training cost about $5 in OpenAI requests and H100 time.

Round 1: Fail! Overfit, catastrophic forgetting. Fine-tuned from Gemma-7B

Round 2: Success! Pretty reasonable result. Adopts more casual structure and some of Paul's opinions and writing style. Fine-tuned from Gemma-3-4B

You can read my full narrative about the process of making this model on my blog https://awesel.com/paulgpt

## How to generate tokens with this model
- git clone https://github.com/awesel/PaulGPT
- cd PaulGPT
- huggingface-cli login (enter your huggingface access token here)
- huggingface-cli download google/gemma-3-4b-it-qat-q4_0-gguf gemma-3-4b-it-q4_0.gguf --local-dir .
- cd inference
- python inference.py
- You will be prompted to choose whether to load both the base and fine-tuned model at the same time, or just talk to one. Then, you will be able to ask questions!
  
Enjoy! The model is tiny and runs at 10-15 tok/second on my Macbook Pro with 24 gb RAM.

Please reach out if generate anything funny using this repo! awesel [at] stanford [dot] edu

