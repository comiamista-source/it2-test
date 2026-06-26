#!/usr/bin/env python3
"""Test IndicTrans2 Hindi->English translation quality."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL="ai4bharat/indictrans2-indic-en-1B"
print("Loading IndicTrans2 1B (indic-en)...", flush=True)
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True)
ip=IndicProcessor(inference=True)

# Clean Hindi Devanagari test sentences (spiritual/philosophical, like the content)
hindi=[
  "शरीर का वजूद नहीं रहता, लेकिन आत्मा हमेशा रहती है।",
  "पाँच तत्व हैं - भूमि, अग्नि, जल, वायु और गगन।",
  "अब युद्ध की बात मत करो, मामा जी के पास पहुँचना है।",
  "जैसे पति को समझा जाए उतना ही काफी है।",
  "धर्म की राह पर चलने से ही सच्चाई मिलती है।",
]

src="hin_Deva"; tgt="eng_Latn"
batch=ip.preprocess_batch(hindi, src_lang=src, tgt_lang=tgt)
inputs=tok(batch, truncation=True, padding="longest", return_tensors="pt")
with torch.no_grad():
    gen=model.generate(**inputs, num_beams=5, max_length=256)
out=tok.batch_decode(gen, skip_special_tokens=True)
out=ip.postprocess_batch(out, lang=tgt)

print("\n===== IndicTrans2 RESULTS =====", flush=True)
for h,e in zip(hindi,out):
    print(f"HI: {h}")
    print(f"EN: {e}\n")