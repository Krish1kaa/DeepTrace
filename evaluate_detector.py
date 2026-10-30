
from deeptrace.detector import AITextDetector

samples = {
    "H1": ("Human", "when i think about my parents escpecially my mom, i feel a sense of calm and at the same time the urge to protect and keep her happy."),
    "H2": ("Human", "I think i have seen this film before.It was excrutiating to see the main lead in pain"),
    "H3": ("Human", "Seconf chance,do not mess it up."),

    "AI1": ("AI", "When I think about my parents, I feel grateful for everything they have done for me. My mother, especially, has always been a source of comfort. Knowing that she is happy and doing well gives me a sense of peace."),
    "AI2": ("AI", "I feel like I have watched this movie before, but somehow it still manages to affect me. Seeing the main character struggle was difficult, especially because you could understand how much pain they were going through."),
    "AI3": ("AI", "Everyone deserves a second chance, but that doesn't mean they should keep making the same mistakes. Sometimes, giving someone another opportunity is also about trusting them to do better.")
}

detector = AITextDetector()

for sample_id, (actual, text) in samples.items():
    score = detector.score(text)
    predicted = "AI" if score >= 0.5 else "Human"

    print(
        f"{sample_id} | Actual: {actual:5} | "
        f"Predicted: {predicted:5} | "
        f"AI probability: {score * 100:.2f}%"
    )
