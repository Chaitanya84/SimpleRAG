# Improved PDF -> sentence chunks -> embeddings pipeline
# Preserves original CSV outputs and adds binary persistence of raw embeddings.
import fitz
import random
import pandas as pd
import re
from spacy.lang.en import English
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import torch

PDF_PATH = "Mahabharata.pdf"
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30  # used to filter very short chunks
PAGES_CSV = "human_nutritiMahaBharataon_text_pages.csv"
CHUNKS_EMBEDDINGS_CSV = "MahaBharata_df.csv"
EMBEDDINGS_TENSOR_FILE = "MahaBharata.pt"  # new binary file with raw embeddings
DEVICE = "cuda"  # change to "cpu" if no GPU is available

def text_formatter(text: str) -> str:
    """Minor, document-specific text normalization."""
    return text.replace("\n", " ").strip()

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Open PDF and return a list of dicts with page metadata and extracted text.
    Page number offset remains exactly as in the original script (page_number - 41).
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc), desc="Reading pages"):
        text = text_formatter(page.get_text())
        pages_and_texts.append({
            "page_number": page_number - 20,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text
        })
    return pages_and_texts

def add_sentences_with_spacy(pages_and_texts: list[dict], nlp) -> None:
    """
    Use nlp.pipe to process pages in batches and add 'sentences' and
    'page_sentence_count_spacy' to each page dict in place.
    """
    texts = [item["text"] for item in pages_and_texts]
    for doc, item in tqdm(zip(nlp.pipe(texts, batch_size=32), pages_and_texts),
                         desc="Sentence segmentation", total=len(texts)):
        sents = [str(s) for s in doc.sents]
        item["sentences"] = sents
        item["page_sentence_count_spacy"] = len(sents)

def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """Split a list into sublists of size slice_size (last one may be smaller)."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def build_sentence_chunks(pages_and_texts: list[dict], chunk_size: int) -> list[dict]:
    """
    Build sentence chunks from pages_and_texts and return list of chunk dicts.
    Each chunk dict contains page_number, sentence_chunk, chunk_char_count,
    chunk_word_count, chunk_token_count.
    """
    pages_and_chunks = []
    for item in tqdm(pages_and_texts, desc="Chunking sentences"):
        item["sentence_chunks"] = split_list(item["sentences"], slice_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
        for sentence_chunk in item["sentence_chunks"]:
            joined = "".join(sentence_chunk).replace("  ", " ").strip()
            joined = re.sub(r'\.([A-Z])', r'. \1', joined)  # space after period before capital letter
            chunk_dict = {
                "page_number": item["page_number"],
                "sentence_chunk": joined,
                "chunk_char_count": len(joined),
                "chunk_word_count": len([w for w in joined.split(" ")]),
                "chunk_token_count": len(joined) / 4
            }
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

def filter_and_embed_chunks(pages_and_chunks: list[dict], embedding_model) -> list[dict]:
    """
    Filter chunks by MIN_TOKEN_LENGTH and compute embeddings.
    Returns list of chunk dicts (only those above the token length threshold)
    with an added 'embedding' key. Also persists raw embeddings to a binary file.
    """
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min = df[df["chunk_token_count"] > MIN_TOKEN_LENGTH].to_dict(orient="records")

    # Prepare text list and compute embeddings in batches
    all_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min]
    all_chunk_embeddings = embedding_model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Persist raw tensor embeddings to a binary file for fast repeated loads.
    # This file is additional and does not change any CSV outputs.
    torch.save(all_chunk_embeddings.cpu(), EMBEDDINGS_TENSOR_FILE)

    # Attach embeddings back to dicts
    for i, item in enumerate(pages_and_chunks_over_min):
        item["embedding"] = all_chunk_embeddings[i]
    return pages_and_chunks_over_min

def main():
    # 1) Prepare spaCy sentencizer pipeline
    nlp = English()
    nlp.add_pipe("sentencizer")

    # 2) Load PDF and extract page texts + metadata
    pages_and_texts = open_and_read_pdf(PDF_PATH)

    # 3) Sentence segmentation (use nlp.pipe for speed/batching)
    add_sentences_with_spacy(pages_and_texts, nlp)

    # 4) Save page-level CSV exactly as before
    df_pages = pd.DataFrame(pages_and_texts)
    df_pages["page_sentence_count_spacy"] = [item["page_sentence_count_spacy"] for item in pages_and_texts]
    df_pages.to_csv(PAGES_CSV, index=False)

    # 5) Build sentence chunks
    pages_and_chunks = build_sentence_chunks(pages_and_texts, chunk_size=NUM_SENTENCE_CHUNK_SIZE)

    # 6) Load sentence-transformers embedding model (on DEVICE) and compute embeddings
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=DEVICE)

    # 7) Filter short chunks, compute embeddings, and persist raw tensor embeddings to binary file
    pages_and_chunks_over_min = filter_and_embed_chunks(pages_and_chunks, embedding_model)

    # 8) Save chunks + embeddings to CSV exactly as before
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min)
    text_chunks_and_embeddings_df.to_csv(CHUNKS_EMBEDDINGS_CSV, index=False)

    # 9) Re-load and print a preview exactly as original script did
    loaded = pd.read_csv(CHUNKS_EMBEDDINGS_CSV)
    print(loaded.head())

if __name__ == "__main__":
    main()