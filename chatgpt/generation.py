import time

import openai


def generate_completion(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    return response["choices"][0]["message"]["content"]


def clean_word(word):
    return "".join([c for c in word if c.isalpha()])


def clean_response(response, word, suffix, language):
    # Remove all non-alphabetic characters from response
    response_split = [
        clean_word(t) for t in response.strip().split() if clean_word(t) != ""
    ]
    # If there is only one word, return it
    if len(response_split) == 1:
        return response_split[0]
    else:
        # Filter based on suffix and input word
        suffix_split = [
            clean_word(t).lower() for t in suffix.split() if clean_word(t) != ""
        ]
        response_filtered = [
            t for t in response_split if t.lower() != word.lower() and 
            t.lower() not in suffix_split
        ]
        # For German, remove "die" and "zwei"
        if language == "german":
            response_filtered = [
                t for t in response_filtered if t.lower() != "die" and
                t.lower() != "zwei" and
                t.lower() != word.split()[1].lower()  # Repeat without article
            ]
        # For English, remove "answer"
        elif language == "english":
            response_filtered = [
                t.lower() for t in response_filtered if t.lower() != "answer"
            ]
        if len(response_filtered) != 1:
            print(word, response, response_filtered)
        if len(response_filtered) == 0:
            return word
        return response_filtered[-1]


def generate_wug(word, shots, prompt_generator):
    prompt = prompt_generator.generate_prompt(word, shots=shots)
    generated_word = None
    while generated_word is None:
        try:
            generated_word = generate_completion(prompt)
        except:
            print("Waiting for 10 seconds...")
            time.sleep(10)
    return clean_response(
        generated_word, 
        word, 
        prompt_generator.suffix, 
        prompt_generator.language
    )
