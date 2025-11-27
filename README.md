
import os
import re
import sys
from typing import List, Dict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# Konfiguration
# ----------------------------
INPUT_XLSX  = r"C:\Users\daltmoos\EZB\Misinfo_classifier\Maltego_dataset_v03.xlsx"
OUTPUT_XLSX = r"C:\Users\daltmoos\EZB\Misinfo_classifier\Maltego_dataset_v03_translated.xlsx"
LANG_COL    = "lang_flag"
TEXT_COL    = "Text"

MODEL_NAME  = "facebook/nllb-200-distilled-600M"
TARGET_LANG = "deu_Latn"  # Deutsch

BATCH_SIZE      = 8
MAX_CHARS_CHUNK = 900  # Grobe Grenze pro Chunk (sicher unter NLLB-Tokenlimit)
SENT_SPLIT_RE   = re.compile(r'(?<=[.!?])\s+')

# ----------------------------
# NLLB Sprachcode-Mapping
# ----------------------------
LANG_MAP: Dict[str, str] = {
    "german":      "deu_Latn",
    "swedish":     "swe_Latn",
    "spanish":     "spa_Latn",
    "slovensk":    "slk_Latn",  # Slowakisch
    "russian":     "rus_Cyrl",
    "romanian":    "ron_Latn",
    "portuguese":  "por_Latn",
    "polish":      "pol_Latn",
    "norwegian":   "nob_Latn",  # Bokmål (typisch)
    "lithuanian":  "lit_Latn",
    "latvian":     "lav_Latn",
    "italian":     "ita_Latn",
    "hungarian":   "hun_Latn",
    "french":      "fra_Latn",
    "english":     "eng_Latn",
    "dutch":       "nld_Latn",
    "danish":      "dan_Latn",
    "czech":       "ces_Latn",
}

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def normalize_lang(s: str) -> str:
    return (s or "").strip().lower()

def chunk_text(text: str, max_chars: int = MAX_CHARS_CHUNK) -> List[str]:
    """Teilt lange Texte zunächst satzweise, dann hart nach Zeichenlimit."""
    t = (text or "").strip()
    if not t:
        return [t]
    if len(t) <= max_chars:
        return [t]

    sentences = SENT_SPLIT_RE.split(t)
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    # Notfall: Falls einzelne Sätze extrem lang sind
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars])
    return final

def translate_batch(texts: List[str], tokenizer, model, device) -> List[str]:
    """NLLB Batch-Übersetzung für bereits gechunkte Texte (gleiche src_lang!).
       Wichtig: tokenizer.src_lang muss VORHER gesetzt sein.
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    # Zielsprach-ID via Tokenizer (Fast-Tokenizer hat kein lang_code_to_id):
    bos_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    with torch.no_grad():
        gen = model.generate(**enc, forced_bos_token_id=bos_id)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)

def translate_text_full(text: str, tokenizer, model, device) -> str:
    """Übersetzt einen Text robust via Chunking + Batch."""
    if text is None:
        return text
    chunks = chunk_text(text)
    out_parts = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        out_parts.extend(translate_batch(batch, tokenizer, model, device))
    return " ".join(out_parts).strip()

# ----------------------------
# Hauptlogik
# ----------------------------
def main():
    # 1) Modell & Tokenizer laden (online). Bei 401 bitte vorher: huggingface-cli login
    print("[INFO] Lade NLLB Modell...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Gerät: {device}")
    model = model.to(device)
    model.eval()

    # 2) Daten laden
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
    if TEXT_COL not in df.columns or LANG_COL not in df.columns:
        raise ValueError(f"Spalten '{TEXT_COL}' und/oder '{LANG_COL}' fehlen.")

    # 3) Normalisieren & Gruppen bilden (pro Sprache übersetzen)
    df["_lang_norm"] = df[LANG_COL].apply(normalize_lang)
    # Deutsche Texte überspringen
    mask_translate = df["_lang_norm"] != "german"
    to_translate_df = df[mask_translate]

    if to_translate_df.empty:
        print("[INFO] Nichts zu übersetzen.")
        df.drop(columns=["_lang_norm"], inplace=True)
        df.to_excel(OUTPUT_XLSX, index=False)
        print(f"[DONE] Datei gespeichert: {OUTPUT_XLSX}")
        return

    print(f"[INFO] Zu übersetzen: {len(to_translate_df)} Zeilen.")

    # 4) Pro Sprache verarbeiten (damit tokenizer.src_lang konsistent ist)
    for lang_norm, group in to_translate_df.groupby("_lang_norm", sort=False):
        src_code = LANG_MAP.get(lang_norm, "eng_Latn")  # Fallback auf Englisch, falls unbekannt
        print(f"[INFO] Sprache: {lang_norm} → NLLB-Code: {src_code} | Zeilen: {len(group)}")

        # NLLB verlangt die Quellsprache am Tokenizer:
        tokenizer.src_lang = src_code

        idxs = group.index.tolist()
        texts = group[TEXT_COL].astype(str).tolist()

        # Fortschritt pro Gruppe
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{lang_norm} → deu"):
            batch_rows = texts[i:i+BATCH_SIZE]
            try:
                translated = translate_batch(batch_rows, tokenizer, model, device)
            except Exception as e:
                # Fallback: chunked, falls die Sequenzen zu lang waren
                print(f"[WARN] Batch-Fehler → Chunking-Fallback. Grund: {e}", file=sys.stderr)
                translated = []
                for t in batch_rows:
                    try:
                        translated.append(translate_text_full(t, tokenizer, model, device))
                    except Exception as e2:
                        # Ultimativer Fallback: Original belassen
                        print(f"[ERROR] Einzeltext-Fallback fehlgeschlagen: {e2}", file=sys.stderr)
                        translated.append(t)

            # Ergebnisse zurückschreiben
            write_idxs = idxs[i:i+BATCH_SIZE]
            for di, new_text in zip(write_idxs, translated):
                df.at[di, TEXT_COL] = new_text

    # 5) Aufräumen & speichern
    df.drop(columns=["_lang_norm"], inplace=True)
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"[DONE] Übersetzung abgeschlossen. Datei gespeichert: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
