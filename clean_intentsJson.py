import json
from collections import defaultdict

# Load original dataset
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

merged = defaultdict(lambda: {"tag": "", "patterns": [], "responses": []})

for intent in data["intents"]:
    tag = intent["tag"]

    # Initialize tag if not set
    if not merged[tag]["tag"]:
        merged[tag]["tag"] = tag

    # Merge patterns (avoid duplicates)
    for p in intent.get("patterns", []):
        if p not in merged[tag]["patterns"]:
            merged[tag]["patterns"].append(p)

    # Merge responses (avoid duplicates)
    for r in intent.get("responses", []):
        if r not in merged[tag]["responses"]:
            merged[tag]["responses"].append(r)

# Convert dict → list format
cleaned_intents = {
    "intents": list(merged.values())
}

# Save cleaned dataset
with open("intents_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_intents, f, indent=4, ensure_ascii=False)

print("Cleaning complete! File saved as intents_cleaned.json")