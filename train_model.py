"""
train_model.py  — base model (no external dataset needed)
----------------------------------------------------------
Uses 150 built-in labelled samples (50 per class).
TextPreprocessor is imported from preprocessor.py so that
pickle can correctly deserialise the model in main.py.

Run:
    python train_model.py
"""

import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# ── Import from shared module (fixes the pickle AttributeError)
from preprocessor import TextPreprocessor

# ─────────────────────────────────────────────────────────────────────────────
# Dataset — 150 samples (50 per class)
# ─────────────────────────────────────────────────────────────────────────────
TEXTS = [
    # ── POSITIVE (50) ──────────────────────────────────────────────────────
    "I absolutely loved this product, it exceeded all my expectations!",
    "This is the best experience I have ever had, truly outstanding.",
    "Fantastic quality and great value for money. Highly recommended!",
    "The customer service was amazing, they resolved my issue instantly.",
    "I am so happy with this purchase. It works perfectly.",
    "Brilliant! This is exactly what I was looking for. Five stars.",
    "Excellent performance and beautiful design. Very impressed.",
    "This made my day so much better. I cannot recommend it enough.",
    "Super fast delivery and the product is even better than described.",
    "Outstanding quality. I will definitely be buying again.",
    "A wonderful experience from start to finish. Truly delightful.",
    "I was blown away by how good this is. Absolutely incredible.",
    "Great product! Works exactly as described and looks amazing.",
    "Very happy with my purchase. High quality and worth every penny.",
    "Love it! The design is sleek and it performs brilliantly.",
    "The best I have ever tried. Incredibly satisfying experience.",
    "Pleasantly surprised by the quality. Exceeded expectations by far.",
    "Absolutely wonderful! I could not be more pleased with this.",
    "Top-notch quality and the delivery was super quick. Very satisfied.",
    "Remarkable product. Everything about it is just right.",
    "This is a game changer. I recommend it to everyone I know.",
    "Perfect in every way. The packaging was also beautiful.",
    "Such a joy to use. Intuitive, fast and well-built.",
    "I gave this as a gift and the recipient was thrilled.",
    "Genuinely impressed. This exceeded every single expectation.",
    "Works like a charm. No issues whatsoever after months of use.",
    "Five stars without hesitation. Absolutely love this item.",
    "The quality is phenomenal for the price. Great value.",
    "Arrived early and in perfect condition. Very pleased.",
    "Smooth experience from order to delivery. Will order again.",
    "I never write reviews but this deserved one. Simply wonderful.",
    "Better than advertised. A truly premium product.",
    "So glad I bought this. Has made my life noticeably easier.",
    "Flawless build quality and it performs exactly as promised.",
    "Incredibly well made. Feels luxurious to use.",
    "This is my second purchase, it is that good.",
    "Hands down the best product in this category.",
    "I am thoroughly impressed. Could not be happier.",
    "Everything I hoped for and more. Stellar product.",
    "The attention to detail is remarkable. Worth every cent.",
    "Life-changing purchase. Cannot imagine going back now.",
    "Works perfectly straight out of the box. Very pleased.",
    "The description was accurate and the quality is superb.",
    "Fast shipping, excellent product, wonderful seller.",
    "Bought three for friends after using mine. That good.",
    "This has become an essential part of my daily routine.",
    "Absolutely no complaints. Will be a repeat customer.",
    "Impressive craftsmanship and it performs brilliantly.",
    "I was skeptical but now I am a total convert. Love it.",
    "Quality is through the roof. Totally worth the price.",

    # ── NEGATIVE (50) ──────────────────────────────────────────────────────
    "Terrible product, broke after just one day of use.",
    "Worst purchase I have ever made. Complete waste of money.",
    "I am very disappointed. The quality is absolutely awful.",
    "Do not buy this. It stopped working immediately and support is useless.",
    "Horrible experience. The product is nothing like the description.",
    "Very poor quality. I regret buying this so much.",
    "Dreadful customer service. They ignored all my complaints.",
    "This product is a scam. It does not work at all.",
    "Extremely disappointed. Fell apart after one use.",
    "Awful. The worst product I have ever encountered.",
    "Total garbage. Broke the same day I received it.",
    "Terrible quality and even worse customer support. Avoid at all costs.",
    "Disgusting experience. I want a refund immediately.",
    "Completely useless product. Nothing works as advertised.",
    "I hate this product. It has caused me nothing but problems.",
    "Worst thing I have ever bought. Absolute rubbish.",
    "Deeply disappointed with the poor quality and late delivery.",
    "Not worth a single penny. Broke within hours of unboxing.",
    "Shameful product quality. Would give zero stars if I could.",
    "Appalling service and a defective product. Never again.",
    "Purchased this as a gift and it embarrassed me. So bad.",
    "Looks nothing like the photos. Cheap and nasty.",
    "Stopped functioning after one week. Absolutely furious.",
    "The worst customer service I have ever experienced.",
    "Arrived broken and the return process is a nightmare.",
    "Cheap materials that feel like they will snap immediately.",
    "Misleading product description. I feel cheated.",
    "Overpriced junk. Do yourself a favour and avoid this.",
    "It does not do half of what it claims to do.",
    "Within two days it was falling apart. Shocking quality.",
    "I have requested a refund three times with no response.",
    "Complete failure. Does not work in any meaningful way.",
    "Terrible in every possible sense. Avoid like the plague.",
    "Flimsy and poorly constructed. Not fit for purpose.",
    "Delivery was two weeks late and the item was damaged.",
    "False advertising. The product bears no relation to the listing.",
    "I threw it in the bin after one hour. Useless.",
    "Broke on first use. This is unacceptable for the price paid.",
    "Absolutely furious with this purchase. Zero stars.",
    "Faulty from day one and nobody will help me.",
    "Dangerous product. Would not recommend to anyone.",
    "A total waste of time, money, and energy.",
    "The seller ignored every single message I sent.",
    "Poorly packaged, arrived smashed and no apology.",
    "I have never been more frustrated with a product.",
    "Do not be fooled by the photos. It is awful in real life.",
    "Nothing works as described. Completely mis-sold.",
    "Cheap plastic that cracked immediately. Utter rubbish.",
    "I regret every penny spent on this. Deeply unsatisfied.",
    "Faulty wiring, dangerous to use. Avoid completely.",

    # ── NEUTRAL (50) ───────────────────────────────────────────────────────
    "The product is okay, nothing special but gets the job done.",
    "Decent quality for the price, but I have seen better.",
    "It works as expected. Not amazing, not terrible.",
    "Average product. Does what it says on the box, nothing more.",
    "Fairly standard. Some features I liked, some I did not.",
    "It is fine. Not what I expected but not bad either.",
    "Mediocre experience. Could be improved in several areas.",
    "Works most of the time. Has some issues but manageable.",
    "Not bad, not great. Middle-of-the-road product.",
    "Acceptable quality. I am neither happy nor unhappy.",
    "It does the job. Nothing to write home about.",
    "Reasonable for the price. Would not rush to buy again.",
    "Some good points, some bad. On balance I would say average.",
    "The product is functional. Not exciting but usable.",
    "Arrived on time. Looks as described. No strong feelings.",
    "I have mixed feelings about this. Good and bad in equal measure.",
    "It fulfils its purpose but lacks any wow factor.",
    "Standard product. Nothing more, nothing less.",
    "Works fine. Delivery was a bit slow but acceptable.",
    "It is what it is. Functional but unremarkable.",
    "The packaging was nice but the product itself is just okay.",
    "Not particularly impressed or disappointed. Just average.",
    "Does what it advertises. Nothing extra though.",
    "A perfectly adequate product for everyday use.",
    "Neither a great buy nor a terrible one. Fairly neutral.",
    "I have used better, but I have also used worse.",
    "Met my basic requirements. That is about all I can say.",
    "The instructions were confusing but the product is okay.",
    "Quality seems average for the price range.",
    "It works, though I expected a bit more at this price.",
    "Satisfactory. Ticks the basic boxes without going further.",
    "No complaints, but nothing to get excited about either.",
    "So-so experience. Would consider alternatives next time.",
    "It does what you need it to. No frills, no problems.",
    "Arrived in good condition. Performance is middle of the road.",
    "Acceptable for occasional use. Not for heavy or daily use.",
    "The build feels average. Time will tell how long it lasts.",
    "I am indifferent. It served its purpose for now.",
    "The colour is different to photos but the product works fine.",
    "Not the best I have used, but passable for the price.",
    "Works as described. I just expected a little more quality.",
    "Fits my needs for now. Might upgrade in the future.",
    "Fine for basic tasks. Do not expect much more than that.",
    "The quality is consistent with the price point. Average.",
    "It does the job adequately. No strong recommendation either way.",
    "Nothing exceptional here. A plain, functional product.",
    "Usable but not impressive. Three stars feels right.",
    "I have mixed views. Some days it works well, others less so.",
    "Gets the job done without any fuss. That is sufficient.",
    "Reasonable purchase for what it is. Would not rave about it.",
]

LABELS     = [1]*50 + [0]*50 + [2]*50   # 1=Positive, 0=Negative, 2=Neutral
LABEL_NAMES = {0: "Negative", 1: "Positive", 2: "Neutral"}

# ─────────────────────────────────────────────────────────────────────────────
# Build Candidate Pipelines
# ─────────────────────────────────────────────────────────────────────────────
def make_pipeline(clf):
    return Pipeline([
        ('pre',   TextPreprocessor()),
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=25_000,
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            min_df=1,
        )),
        ('clf', clf),
    ])

candidates = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(max_iter=2000, C=5.0, class_weight='balanced',
                           solver='lbfgs', multi_class='multinomial')
    ),
    "LinearSVC (calibrated)": make_pipeline(
        CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, class_weight='balanced'), cv=5
        )
    ),
    "Complement Naive Bayes": make_pipeline(
        ComplementNB(alpha=0.3)
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Compare with 5-Fold CV
# ─────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 54)
print("   MODEL COMPARISON — 5-Fold Stratified CV")
print("=" * 54)

best_name, best_score, best_pipeline = None, 0.0, None

for name, pipe in candidates.items():
    scores = cross_val_score(pipe, TEXTS, LABELS, cv=cv, scoring='accuracy')
    mean, std = scores.mean(), scores.std()
    print(f"  {name:<28}  {mean:.4f} ± {std:.4f}")
    if mean > best_score:
        best_score, best_name, best_pipeline = mean, name, pipe

print("-" * 54)
print(f"  ✅  Best: {best_name}  ({best_score:.4f})")
print("=" * 54)

# ─────────────────────────────────────────────────────────────────────────────
# Train on full dataset & evaluate
# ─────────────────────────────────────────────────────────────────────────────
best_pipeline.fit(TEXTS, LABELS)
y_pred = best_pipeline.predict(TEXTS)

print("\n--- Classification Report ---")
print(classification_report(LABELS, y_pred,
      target_names=["Negative", "Positive", "Neutral"]))

# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────
test_sentences = [
    "This is absolutely amazing, I love it!",
    "Terrible product, complete waste of money.",
    "It is okay, nothing special.",
    "Outstanding! Best purchase of the year.",
    "Broken on arrival. Totally useless.",
    "Average at best. Does the job, barely.",
]

print("--- Sanity Check ---")
for s in test_sentences:
    proba = best_pipeline.predict_proba([s])[0]
    pred  = best_pipeline.predict([s])[0]
    conf  = round(max(proba) * 100, 1)
    label = LABEL_NAMES[pred]
    bar   = "█" * int(conf / 5)
    print(f"  [{label:8s} {conf:5.1f}%] {bar:<20}  {s}")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

print(f"\n✅  model.pkl saved  |  {best_name}  |  CV acc: {best_score:.4f}")
