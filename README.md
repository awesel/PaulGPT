## PaulGPT

This contains a few fine-tuned Gemma models intended to sound like Paul Graham.

Round 1: Fail! Overfit, catastrophic forgetting. Fine-tuned from Gemma-7B

Round 2: Success! Pretty reasonable result. Adopts more casual structure and some of Paul's opinions and writing style. Fine-tuned from Gemma-3-4B

The .gitignore is set to ignore the full model weights. So, if you want to run inference, you need to download the appropriate quantized file from Hugging Face. For the Round 2 (successful run), download https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/tree/main using "huggingface download".

You will also need to adjust the pathnames in inference.py to your computer. The model is tiny and runs at 10-15 tok/second on my Macbook Pro with 24 gb RAM. If inference doesn't work, try not loading both models into memory at the same time.

You can read my full narrative about this process on my blog on awesel.com

Please reach out if generate anything funny using this repo! awesel [at] stanford [dot] edu
