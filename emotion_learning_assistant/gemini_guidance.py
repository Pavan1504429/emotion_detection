"""
Phase 4: AI-Powered Guidance using Gemini AI.
Generates a short, empathetic, actionable response tailored to the student's
detected emotion and the problem they described.

Requires env var GEMINI_API_KEY. If missing/unavailable, falls back to a
template-based response so the app never breaks the demo/flow.
"""
import os

TEMPLATES = {
    "Confused": (
        "It's completely okay to feel confused here — that usually means you're right at the "
        "edge of learning something new. Try breaking the problem into the smallest possible "
        "sub-step, work through one concrete example by hand, and re-read only the section that "
        "covers that exact sub-step before moving on."
    ),
    "Frustrated": (
        "That frustration is valid — you've clearly put in real effort. Step away for 5 minutes, "
        "then come back and isolate exactly where it breaks (print/log intermediate values). "
        "Fixing one small piece at a time will rebuild momentum faster than trying to fix everything at once."
    ),
    "Bored": (
        "Sounds like the material isn't challenging you right now. Try applying the same concept "
        "to a harder or more unusual example, teach it to someone else in one sentence, or skip "
        "ahead and see how this topic connects to something more advanced."
    ),
    "Curious": (
        "Great instinct to want the 'why', not just the 'how'. Look up one real-world example of "
        "this concept in action, then try to explain it in your own words without notes — that's "
        "usually the fastest way to satisfy real curiosity and make it stick."
    ),
    "Confident": (
        "You're in a strong spot right now — a great time to test that confidence with a slightly "
        "harder variation of the problem, or to time yourself solving a similar one to build exam speed."
    ),
}


def _fallback_guidance(emotion: str, problem_text: str) -> str:
    return TEMPLATES.get(emotion, "Keep going — describe what specifically is tripping you up and we'll tackle it step by step.")


def generate_guidance(problem_text: str, emotion: str, confidence: float, api_key: str = None) -> str:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback_guidance(emotion, problem_text)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are an empathetic academic support assistant for students.\n"
            f"Student's detected emotion: {emotion} (confidence {confidence:.0%})\n"
            f"Student's message: \"{problem_text}\"\n\n"
            "Write a short response (3-5 sentences) that:\n"
            "1. Acknowledges their emotional state naturally (without naming it clinically)\n"
            "2. Gives 1-2 concrete, actionable next steps for their specific problem\n"
            "3. Ends with a brief, genuine line of encouragement\n"
            "Keep it warm but concise. No markdown headers, no bullet lists, plain prose."
        )
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return text if text else _fallback_guidance(emotion, problem_text)
    except Exception as e:
        return _fallback_guidance(emotion, problem_text) + f"\n\n(Gemini unavailable: {e})"


if __name__ == "__main__":
    print(generate_guidance("I'm so lost on recursion", "Confused", 0.82))
