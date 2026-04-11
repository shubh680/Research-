import os
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
# Use raw strings (r"") to prevent Windows path errors
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
    
    # 1. Initialize Stage 1: CLIP
    print("--- [INIT] Loading CLIP (Stage 1)...")
    clip_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    
    # 2. Initialize Stage 2: Moondream (WITH PHICONFIG FIX)
    print("--- [INIT] Loading Moondream VLM (Stage 2)...")
    vlm_id = "vikhyatk/moondream2"
    vlm_revision = "2024-08-26" # Using a stable recent revision
    
    # FIX: Load tokenizer and config separately to patch the missing pad_token_id
    vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_id, revision=vlm_revision)
    vlm_config = AutoConfig.from_pretrained(vlm_id, trust_remote_code=True, revision=vlm_revision)
    
    # Explicitly set the pad_token_id to avoid the AttributeError in Phi modeling
    if not hasattr(vlm_config, "pad_token_id") or vlm_config.pad_token_id is None:
        vlm_config.pad_token_id = vlm_tokenizer.eos_token_id
    
    vlm_model = AutoModelForCausalLM.from_pretrained(
        vlm_id, 
        config=vlm_config,
        trust_remote_code=True, 
        revision=vlm_revision,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16
    ).to(device)
    vlm_model.eval()
    
    # 3. Initialize Stage 3: SBERT
    print("--- [INIT] Loading SBERT (Stage 3)...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Infrastructure (FAISS)
    index = faiss.IndexFlatL2(512)
    image_paths = []

    def get_clip_features(path):
        """Extracts features and handles potential wrapper object errors."""
        img = Image.open(path)
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model.get_image_features(**inputs)
            # Handle BaseOutput wrappers
            tensor = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        return tensor.cpu().numpy()

    # --- EXECUTION: INDEXING ---
    print(f"\n--- [INDEX] Processing {len(TEST_GALLERY)} items ---")
    for path in TEST_GALLERY:
        if not os.path.exists(path):
            print(f" Skipping: {path} (Not Found)")
            continue
        emb = get_clip_features(path)
        faiss.normalize_L2(emb)
        index.add(emb)
        image_paths.append(path)
        print(f"Indexed: {os.path.basename(path)}")

    # --- EXECUTION: STAGE 1 (VISUAL) ---
    print(f"\n--- [STAGE 1] Searching for: {os.path.basename(QUERY_IMAGE)} ---")
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

    # --- EXECUTION: STAGE 2 (SEMANTIC REFINEMENT) ---
    print("--- [STAGE 2] Performing Semantic Refinement...")
    
    def get_vlm_context(img_path):
        image = Image.open(img_path)
        # Moondream internal method for captioning
        enc_image = vlm_model.encode_image(image)
        caption = vlm_model.answer_question(enc_image, "Describe this object and its setting briefly.", vlm_tokenizer)
        return caption

    q_ctx_text = get_vlm_context(QUERY_IMAGE)
    print(f"   [Query Context]: {q_ctx_text}")

    final_results = []
    for match in initial_matches:
        m_ctx_text = get_vlm_context(match['path'])
        
        # Compute Semantic Consistency via SBERT
        q_text_emb = text_model.encode(q_ctx_text, convert_to_tensor=True)
        m_text_emb = text_model.encode(m_ctx_text, convert_to_tensor=True)
        semantic_score = util.cos_sim(q_text_emb, m_text_emb).item()

        # Weighted Score Fusion
        final_score = (ALPHA * match['visual_score']) + ((1 - ALPHA) * semantic_score)

        final_results.append({
            "path": match['path'],
            "old_rank": match['rank'],
            "final_score": final_score,
            "caption": m_ctx_text
        })

    # Re-rank based on the fused score
    final_results = sorted(final_results, key=lambda x: x['final_score'], reverse=True)

    # --- TERMINAL LOGGING ---
    print("\n" + "="*95)
    print(f"{'FILENAME':<20} | {'OLD RANK':<10} | {'NEW RANK':<10} | {'SCORE':<10} | {'CONTEXTUAL IDENTITY'}")
    print("-" * 95)
    for i, res in enumerate(final_results):
        fname = os.path.basename(res['path'])
        print(f"{fname:<20} | {res['old_rank']:<10} | {i+1:<10} | {res['final_score']:.4f} | {res['caption']}")
    print("="*95 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()