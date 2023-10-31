import argparse
import random
import tqdm

import openai
import pandas as pd

import generation
import prompting


def main():

    random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language", 
        default=None, 
        type=str, 
        help="Language to analyze."
    )
    parser.add_argument(
        "--prompt_type", 
        default=None, 
        type=str, 
        help="Short or long prompts."
    )
    args = parser.parse_args()

    # Add API key
    API_KEY = ""
    openai.api_key = API_KEY

    # Load data
    data = pd.read_csv(
        "data/{}.csv".format(args.language)
    )

    # Prepare prompting
    prefix, suffix, shots = prompting.load_prompt(args.language, args.prompt_type)
    prompt_generator = prompting.PromptGenerator(
        prefix, 
        suffix,
        args.language
    )

    # Zero-shot
    print("Conducting zero-shot experiments...")
    generated_words = []
    for _ in range(10):
        generated_words_run = []
        for word in tqdm.tqdm(data["source_word_prompt"]):
            generated_word = generation.generate_wug(word, [], prompt_generator)
            generated_words_run.append(generated_word)
        generated_words.append(generated_words_run)
    generated_words_df = pd.DataFrame(generated_words)
    generated_words_df.to_csv(
        "generations/{}_{}_zero.csv".format(args.language, args.prompt_type), 
        header=False, 
        index=False
    )

    # One-shot
    print("Conducting one-shot experiments...")
    generated_words = []
    for shot in shots:
        generated_words_run = ["-".join(shot)]
        for word in tqdm.tqdm(data["source_word_prompt"]):
            generated_word = generation.generate_wug(word, [shot], prompt_generator)
            generated_words_run.append(generated_word)
        generated_words.append(generated_words_run)
    generated_words_df = pd.DataFrame(generated_words)
    generated_words_df.to_csv(
        "generations/{}_{}_one.csv".format(args.language, args.prompt_type), 
        header=False, 
        index=False
    )

    # Few-shot
    print("Conducting few-shot experiments...")
    generated_words = []
    for _ in range(10):
        random.shuffle(shots)
        generated_words_run = ["|".join(["-".join(shot) for shot in shots])]
        for word in tqdm.tqdm(data["source_word_prompt"]):
            generated_word = generation.generate_wug(word, shots, prompt_generator)
            generated_words_run.append(generated_word)
        generated_words.append(generated_words_run)
    generated_words_df = pd.DataFrame(generated_words)
    generated_words_df.to_csv(
        "generations/{}_{}_few.csv".format(args.language, args.prompt_type), 
        header=False, 
        index=False
    )


if __name__ == "__main__":
    main()
