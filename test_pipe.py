import os
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    LlavaForConditionalGeneration, AutoProcessor
)
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
TEST_GALLERY = [
    r"img\apple1.jpg",
    r"img\apple2.jpg",
    r"img\apple3.jpg",
    r"img\apple4.jpg"
]
QUERY_IMAGE = r"img\query_apple.jpg"
ALPHA = 0.4  # Weight: 40% Visual, 60% Semantic

def main():
    print("--- [INIT] Initializing Research Pipeline ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- [INFO] Using device: {device}")

    # ------------------------------------------------------------------ #
    # STAGE 1: CLIP — visual embedding + FAISS retrieval (unchanged)
    # ------------------------------------------------------------------ #
    print("--- [INIT] Loading CLIP (Stage 1)...")
    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)

    # ------------------------------------------------------------------ #
    # STAGE 2: LLaVA-1.5-7B — replaces Moondream
    #   - Actively maintained under huggingface/transformers directly
    #   - No trust_remote_code, no revision pinning, no cache patching
    #   - Same interface: image -> caption string
    #   - Uses 4-bit quantization on GPU to keep VRAM under 8GB,
    #     falls back to float32 on CPU (slow but works)
    # ------------------------------------------------------------------ #
    print("--- [INIT] Loading LLaVA-1.5 VLM (Stage 2)...")
    vlm_id = "llava-hf/llava-1.5-7b-hf"

    if device == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        vlm_model = LlavaForConditionalGeneration.from_pretrained(
            vlm_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # CPU: no quantization, load in float32
        vlm_model = LlavaForConditionalGeneration.from_pretrained(
            vlm_id,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    vlm_processor = AutoProcessor.from_pretrained(vlm_id)
    vlm_model.eval()

    # ------------------------------------------------------------------ #
    # STAGE 3: SBERT — semantic scoring (unchanged)
    # ------------------------------------------------------------------ #
    print("--- [INIT] Loading SBERT (Stage 3)...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2')

    # ------------------------------------------------------------------ #
    # FAISS index (unchanged)
    # ------------------------------------------------------------------ #
    index = faiss.IndexFlatL2(512)
    image_paths = []

    def get_clip_features(path):
        img = Image.open(path).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            tensor = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        return tensor.cpu().numpy()

    def get_vlm_caption(img_path):
        """
        LLaVA-1.5 captioning. Uses the chat-style prompt format it was trained on.
        Returns a plain string caption.
        """
        image = Image.open(img_path).convert("RGB")
        prompt = "USER: <image>\nDescribe this object and its setting briefly.\nASSISTANT:"
        inputs = vlm_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(vlm_model.device)

        with torch.no_grad():
            output_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )

        # Decode only the newly generated tokens (skip the prompt)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        caption = vlm_processor.decode(generated, skip_special_tokens=True).strip()
        return caption

    # ------------------------------------------------------------------ #
    # INDEXING
    # ------------------------------------------------------------------ #
    print(f"\n--- [INDEX] Processing {len(TEST_GALLERY)} items ---")
    for path in TEST_GALLERY:
        if not os.path.exists(path):
            print(f"  Skipping: {path} (Not Found)")
            continue
        emb = get_clip_features(path)
        faiss.normalize_L2(emb)
        index.add(emb)
        image_paths.append(path)
        print(f"  Indexed: {os.path.basename(path)}")

    if len(image_paths) == 0:
        print("FATAL: No images indexed. Check your img\\ paths.")
        return

    # ------------------------------------------------------------------ #
    # STAGE 1 — VISUAL SEARCH
    # ------------------------------------------------------------------ #
    print(f"\n--- [STAGE 1] Searching for: {os.path.basename(QUERY_IMAGE)} ---")
    if not os.path.exists(QUERY_IMAGE):
        print(f"FATAL: Query image not found: {QUERY_IMAGE}")
        return

    q_emb = get_clip_features(QUERY_IMAGE)
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, len(image_paths))

    initial_matches = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        initial_matches.append({
            'path': image_paths[idx],
            'visual_score': 1 / (1 + dist),
            'rank': rank + 1
        })

    # ------------------------------------------------------------------ #
    # STAGE 2 — SEMANTIC REFINEMENT
    # ------------------------------------------------------------------ #
    print("--- [STAGE 2] Performing Semantic Refinement...")

    q_ctx_text = get_vlm_caption(QUERY_IMAGE)
    print(f"  [Query Context]: {q_ctx_text}")

    # Encode query caption once
    q_text_emb = text_model.encode(q_ctx_text, convert_to_tensor=True)

    final_results = []
    for match in initial_matches:
        m_ctx_text = get_vlm_caption(match['path'])
        m_text_emb = text_model.encode(m_ctx_text, convert_to_tensor=True)
        semantic_score = util.cos_sim(q_text_emb, m_text_emb).item()

        final_score = (ALPHA * match['visual_score']) + ((1 - ALPHA) * semantic_score)

        final_results.append({
            "path": match['path'],
            "old_rank": match['rank'],
            "final_score": final_score,
            "caption": m_ctx_text
        })

    final_results = sorted(final_results, key=lambda x: x['final_score'], reverse=True)

    # ------------------------------------------------------------------ #
    # RESULTS
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 95)
    print(f"{'FILENAME':<20} | {'OLD RANK':<10} | {'NEW RANK':<10} | {'SCORE':<10} | {'CONTEXTUAL IDENTITY'}")
    print("-" * 95)
    for i, res in enumerate(final_results):
        fname = os.path.basename(res['path'])
        print(f"{fname:<20} | {res['old_rank']:<10} | {i+1:<10} | {res['final_score']:.4f}     | {res['caption']}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()