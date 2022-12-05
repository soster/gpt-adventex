# Little playground for text generation with gpt-neo

# transformers:
# easy access to models from huggingface
# they are stored locally at ~/.cache/huggingface/...
# see https://huggingface.co/docs/transformers/model_doc/gpt_neo
from transformers import pipeline
import time

# for gpt-3:
import os
import openai

start_time = time.time()

use_openai = True





# models gpt-neo-125M, gpt-neo-1.3B, gpt-neo-2.7B (8GB Vram not enough for 2.7B)
# for models see https://huggingface.co/models?search=gpt-neo
# device=-1 = CPU, device = 0 = GPU
# for my text adventure this seems interesting: 
# https://huggingface.co/KoboldAI/GPT-Neo-1.3B-Adventure?text=My+name+is+Clara+and+I+am
# KoboldAI/GPT-Neo-1.3B-Adventure
# EleutherAI/gpt-neo-1.3B
# Some interesting models:
# KoboldAI/OPT-350M-Nerys-v2 (https://huggingface.co/KoboldAI/OPT-350M-Nerys-v2)

generator = pipeline('text-generation', model='KoboldAI/GPT-Neo-1.3B-Adventure', device=0)


#generator = pipeline('text-generation', model='/mnt/64CE7D1CCE7CE826/models/gpt-neo-2.7B/', device=0)




prompt = "Context: a fantasy setting with a medieval world. \
It begins as soon as you wake up, \
with a searing pain piercing your temples and spreading across your head.\
Then comes the sickness, the dry mouth and the sudden panic -\
all accompanied alongside that phrase, \"never again\".\
As soon as you open your eyes you realize: This is no hangover.\
You are not in your bedroom.\nYou have lost your memories of yesterday, the last thing you remember is...\
well, you are not sure WHAT you remember."

#prompt = "You are a peasant in a medieval village. You are engaged to the beautiful servant of the evil lord."

#    do_sample — When True, picks words based on their conditional probability.
#    temperature — How many potential answers are considered when performing sampling from the peak.
#    max_length — Maximum number of generated tokens
#    min_length — Minimum number of generated tokens

do_sample = True
min_length = 10
max_length = 50
temperature = .8


def generate(prompt, min_length, max_length, temp):
    intermediate = generator(prompt, do_sample=True, min_length=min_length, max_length=max_length, temperature=temp)
    #print(intermediate)
    return intermediate[0]['generated_text']

#https://beta.openai.com/docs/api-reference/introduction
def generate_openai(i_prompt, min_lenght, max_length, temp):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    intermediate = openai.Completion.create(
        engine="text-davinci-003",
        prompt=i_prompt,
        temperature=temp,
        max_tokens=max_length,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    #print(intermediate)
    return intermediate['choices'][0]['text']

output = 'x'

for x in range(4):
    print('loop: ')
    print(x)
    print()
    if x>0:
        prompt = output[-int(len(output)/2):]
        print ('next prompt: ')
        print (prompt)
        print()

    if use_openai:
        output = generate_openai(prompt, min_length, max_length, temperature)
    else:
        output = generate(prompt, min_length, max_length, temperature)
    end_time = time.time()

    print('prompt after generation')
    print(prompt)

    print('elapsed (s): ')
    print(end_time-start_time)
    print()
    print('output:')
    print(output)

    with open('texts.txt', 'a') as text_file:
        text_file.writelines('\n>>> ')
        text_file.writelines(output)


    

