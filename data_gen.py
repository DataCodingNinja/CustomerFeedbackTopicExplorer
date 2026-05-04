# data_gen.py
"""
Generates a synthetic customer feedback CSV suitable for topic modeling.
Writes data/feedback_sample.csv with columns: id,text,rating,source
"""
import csv
import random
import os

OUTPUT_DIR = "data"
OUT_FILE = os.path.join(OUTPUT_DIR, "feedback_sample.csv")
N = 1000  # adjust down if needed

SOURCES = ["web", "mobile", "email", "chat"]
RATINGS = [1,2,3,4,5]

TEMPLATES = {
    "bug": [
        "App crashes when I try to {}.",
        "I keep getting an error on {} screen.",
        "The {} button doesn't work, it just {}."
    ],
    "ux": [
        "The new layout is confusing and I can't {}.",
        "Navigation is hard; I expected {} to be easier.",
        "The font size is too small on {} pages."
    ],
    "pricing": [
        "Pricing is too high for {} feature.",
        "I was charged twice when I {}.",
        "Subscription plans are confusing; I want {} option."
    ],
    "feature": [
        "Please add {} to improve {}.",
        "I'd love a feature that can {}.",
        "Need support for {} integrations."
    ],
    "praise": [
        "Loving the app, {} works great!",
        "Great experience with {} — thank you!",
        "Excellent support when I {}."
    ],
    "other": [
        "No comment, just checking in.",
        "I have a general question about {}.",
        "Not sure how to use the app for {}."
    ]
}

SAMPLE_ITEMS = [
    "logging in", "signup", "uploading photos", "checkout", "search",
    "profile update", "notifications", "sync", "payments", "integration with serviceX",
    "dark mode", "two-factor auth", "sharing", "filters", "sorting"
]

def synth_sentence(category):
    template = random.choice(TEMPLATES[category])
    slot = random.choice(SAMPLE_ITEMS)
    extra = random.choice(["quickly", "without errors", "every time", "sometimes", "recently"])
    return template.format(slot, extra)

def generate_row(i):
    # choose category distribution skewed to realistic mix
    cat = random.choices(
        population=["bug","ux","pricing","feature","praise","other"],
        weights=[0.25,0.2,0.1,0.2,0.15,0.1],
        k=1
    )[0]
    text = synth_sentence(cat)
    # vary length a bit
    if random.random() < 0.3:
        text += " " + random.choice(["Also, " + synth_sentence(random.choice(list(TEMPLATES.keys())))])
    rating = random.choices(RATINGS, weights=[0.2,0.15,0.25,0.25,0.15])[0] if cat!="praise" else random.choices(RATINGS, weights=[0.05,0.05,0.1,0.3,0.5])[0]
    source = random.choice(SOURCES)
    return {"id": i, "text": text, "rating": rating, "source": source, "category": cat}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = [generate_row(i) for i in range(1, N+1)]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","text","rating","source","category"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {OUT_FILE}")

if __name__ == "__main__":
    main()
