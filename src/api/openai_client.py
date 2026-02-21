from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_topic_label(documents):
    """
    Generate a concise topic label based on representative documents.
    """

    sample = "\n\n".join(documents[:5])

    prompt = f"""
    You are an expert topic labeling system.

    Based on the following documents, generate a concise topic label
    of 2–5 words maximum.

    Documents:
    {sample}

    Topic label:
    """

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You label topics concisely."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=20,
    )

    return response.choices[0].message.content.strip()
