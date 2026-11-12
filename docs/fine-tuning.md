# Making the small model better: what is used, and how to train it

A 2B model on a CPU is fast and private but makes more mistakes than a large one. These are the techniques that pay
off here, what Plane AI does with each, and what is left out on purpose.

| Technique | In Plane AI |
| --- | --- |
| **RAG** (retrieval-augmented generation) | Yes. Work items and Hub docs are chunked and searched (BM25, plus embeddings when `PLANE_AI_EMBED_MODEL` is set); chat and Ask answer from what is retrieved, limited to projects the person can open. |
| **Mixture of experts at the system level** | Yes. Ask routes each question to at most three data "experts" (hours, Day Report notes, evidence, risks, workload, documents), each with its own instructions. A lexical router always runs; at Medium/High the model also picks. A real MoE inside a dense 2B model is not practical. |
| **Reasoning optimisation** (plan, verify) | Yes, cheaply. Summaries at High effort: a reviewer agent checks the draft and the writer fixes it (writer/evaluator loop). Ask: every number in an answer must appear in the data; at Medium/High a wrong one is rewritten once and the page shows whether the answer passed. Daily review: the person's verdict is computed from the row verdicts, not guessed. Thinking tokens are turned off: on a 2B model they cost minutes for little gain. |
| **Quantization** | Yes. The model is Q4_K_M (4-bit); the attention (KV) cache is 8-bit (`PLANE_AI_KV_CACHE_TYPE=q8_0`) with flash attention. |
| **KV cache** | Yes. Ollama keeps the model loaded (`keep_alive`), llama.cpp reuses the cached prompt prefix (`cache_prompt`), and context per effort keeps the cache small at Low. |
| **Speculative decoding** | Optional, llama.cpp only: `runtimes/llama-cpp/speculative.yaml` with a small draft model of the same family. |
| **Knowledge distillation + synthetic data** | Prepared. Every model call a job makes is stored with its prompt. Admins rate answers in Plane (thumbs, with the right answer when wrong). "Download training data" exports them as chat-format JSONL. A stronger "teacher" model can also produce answers for the same prompts (synthetic data) before a human checks them. |
| **LoRA / QLoRA** | Prepared. Train an adapter from that JSONL on a machine with a GPU, convert it to GGUF and load it with `PLANE_AI_LORA_ADAPTER` (below). The server only runs it. |
| **Pruning** | No. Removing weights from a 2B model costs accuracy and rarely speeds up CPU inference; a smaller quantization or model does that better. |

## From ratings to a LoRA adapter

1. **Collect.** Use the Daily review, Ask and the other features; rate answers in Plane. A wrong answer with a
   correction is the most valuable example. A few hundred good examples already help.
2. **Export.** Dashboard › AI › "Download training data" gives `plane-ai-training-<workspace>.jsonl`; each line is
   `{"messages": [system, user, assistant]}`. It contains your team's data: keep it as private as the database.
3. **Optional teacher pass (distillation).** For prompts without a correction, let a stronger model answer the same
   `messages` and keep only the answers a person approves. Never send the file to a service your company has not
   approved for this data.
4. **Train** a LoRA on the same base model the server runs (the original, non-GGUF Qwen3.5-2B weights), for example with Unsloth or
   PEFT (QLoRA on one GPU: rank 16, 2-3 epochs, learning rate around 2e-4). Hold back 10% of the lines to check that
   answers got better, not just different.
5. **Convert** the adapter to GGUF with llama.cpp's `convert_lora_to_gguf.py`.
6. **Load** it: copy the `.gguf` into the model volume (next to the main model) and set in `.env`

   ```dotenv
   PLANE_AI_LORA_ADAPTER=plane-lora.gguf
   ```

   Ollama re-imports the model with the adapter at its next start (`docker compose up -d`). For llama.cpp add
   `-f runtimes/llama-cpp/lora.yaml`. Remove the line to go back to the plain model.
