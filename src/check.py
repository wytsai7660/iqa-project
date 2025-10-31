from transformers import AutoTokenizer
import torch
# from src.owl3.processing_mplugowl3 import mPLUGOwl3BatchFeature, mPLUGOwl3ImageProcessor, mPLUGOwl3Processor
from src.owl3.modeling_mplugowl3 import mPLUGOwl3Model

MODEL_DIR = "./src/owl3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model=mPLUGOwl3Model.from_pretrained(MODEL_DIR)

ids=torch.tensor([[151644,   8948,    198, 151645,    198, 151644,    872,    198,     27,
             91,   1805,     91,     29,    198,   3838,    374,    697,   4271,
          10728,    369,    419,   2168,     30, 151645,    198, 151644,  77091,
            198,    785,   4271,    315,    419,   2168,    374,  12456,     13,
         151645, 151643]])


tokens=model._decode_text(ids,tokenizer=tokenizer)
print(tokens)
