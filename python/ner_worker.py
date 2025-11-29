# /python/ner_worker.py

import sys
import json
import logging
from gliner import GLiNER

# Configure logging to stderr to keep stdout clean for JSON
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[NERWorker] %(message)s')
logger = logging.getLogger()

DEFAULT_LABELS = [
    "person", "organization", "location", "technology", 
    "process", "event", "metric", "date", "concept", "issue"
]

def load_model():
    model_name = "urchade/gliner_multi"
    logger.info(f"Loading GLiNER: {model_name}...")
    try:
        model = GLiNER.from_pretrained(model_name)
        logger.info("GLiNER loaded.")
        return model
    except Exception as e:
        logger.error(f"Failed to load GLiNER: {e}")
        # Fallback not really possible for NER, exit
        sys.exit(1)

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    model = load_model()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            req = json.loads(line)
            texts = req.get("texts", [])
            labels = req.get("labels", [])

            if not texts:
                sys.stdout.write(json.dumps({"error": "No texts provided"}) + "\n")
                sys.stdout.flush()
                continue

            if not labels:
                labels = DEFAULT_LABELS

            # Prepare results container: List of Lists
            # batch_results[i] corresponds to texts[i]
            batch_results = [[] for _ in texts]

            # 1. Split labels into batches of 10 to preserve model accuracy
            label_batches = list(chunk_list(labels, 10))

            # 2. Process each text
            for i, text in enumerate(texts):
                if not text.strip():
                    continue

                # 3. Process each label batch for this text
                for label_batch in label_batches:
                    # predict_entities returns a list of dicts
                    entities = model.predict_entities(text, label_batch)
                    
                    for e in entities:
                        batch_results[i].append({
                            "text": e["text"],
                            "label": e["label"],
                            "start": e["start"],
                            "end": e["end"],
                            "score": e["score"]
                        })

            sys.stdout.write(json.dumps({"results": batch_results}) + "\n")
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"NER Error: {e}")
            error_resp = {"error": str(e)}
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()