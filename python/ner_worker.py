# /python/ner_worker.py

import sys
import json
import logging
from gliner import GLiNER

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("ner_worker")

DEFAULT_LABELS = [
    "person", "organization", "location", "technology", 
    "process", "event", "metric", "date", "concept", "issue"
]

def load_model():
    model_name = "urchade/gliner_multi" # or "urchade/gliner_small-v2.1" for speed
    logger.info(f"Loading GLiNER: {model_name}...")
    try:
        model = GLiNER.from_pretrained(model_name)
        logger.info("GLiNER loaded.")
        return model
    except Exception as e:
        logger.error(f"Failed to load GLiNER: {e}")
        sys.exit(1)

def main():
    model = load_model()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            req = json.loads(line)
            text = req.get("text", "")
            # Allow custom labels per request, fallback to default
            labels = req.get("labels", DEFAULT_LABELS) 

            if not text:
                raise ValueError("Empty text")

            # Predict entities
            entities = model.predict_entities(text, labels)
            
            # Format for Go: GLiNER returns dicts, we just ensure they are clean
            # Output: [{"text": "Java", "label": "technology", "start": 10, "end": 14}, ...]
            formatted_entities = []
            for e in entities:
                formatted_entities.append({
                    "text": e["text"],
                    "label": e["label"],
                    "start": e["start"],
                    "end": e["end"]
                })

            resp = {"entities": formatted_entities}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"NER Error: {e}")
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()