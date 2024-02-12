import random
import openai
import json
import os
import sys
from tqdm import tqdm
import string
import time

SEED = 888
random.seed(SEED)

# Check if the OpenAI API key environment variable exists
if 'OPENAI_API_KEY' in os.environ:
    # Access the value of the API key
    api_key = os.environ['OPENAI_API_KEY']
    print(f"OpenAI API Key: {api_key}")
else:
    print("OpenAI API Key environment variable is not set.")


def get_openai_fact(country, capital, model):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""The capital of {country} is {capital}. Using the output format below, generate a well known fact about a well known city in this country that is NOT the capital. This fact should be true only of this other city, and not true of the capital city. Examples are landmarks in this other city or historical events that happened in this city. Explictly think about what is not true of the capital city {capital} but true of this other city in {country}.
        
  Output format:

  {{"country": "{country}",
  "capital city": "{capital}",
  "reasoning": "...",
  "other city": "...",
  "fact about other city": "..."}}
        """},
    ]
  a = openai.ChatCompletion.create(
    model=model,
    # model="gpt-4",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  summary = json.loads(summary)
  other_city = summary['other city']
  fact = summary['fact about other city']
  return other_city, fact

def get_openai_check_fact(fact, country, capital, other_city, model):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""The capital of {country} is {capital}. Below is a statement about another city in this country: {other_city}. Drawing from your existing knowledge, evaluate if this statement (1) is factual and (2) talks of a fact about {other_city} that is true only of {other_city} and not of {capital}. You should first reason about the problem, and then answer "YES" OR "NO".
        
  Statement: {fact}

  Output format:

  {{"reasoning": "...",
  "meets criteria YES or NO": "..."}}"""},
    ]
  a = openai.ChatCompletion.create(
    model=model,
    # model="gpt-4",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  summary = json.loads(summary)
  check = summary['meets criteria YES or NO']
  if check != 'YES' and check != 'NO':
    raise Exception("Invalid response")
  if check == 'YES':
    return True
  else:
    return False


def get_openai_clear_document(country, other_city, fact, model):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
        A well known city in {country} is {other_city}. 
        Here's a fact about it: {fact}
        Please generate a varied, interesting sentence that (1) first mentions the name of the country and then (2) mentions the fact about the aforementioned city in the same sentence. However, it's extremely important that the fact be mentioned before the city name {other_city} is mentioned, and it should be natural, but rather clear that the city {other_city} is about to be mentioned. Generate only the sentence and nothing else. The provided fact might mention the capital city of the country in addition to {other_city}, but you should mention {other_city} only.

For example, for Afghanistan's city Herat, here is a fact about it: Herat is home to the Great Mosque of Herat (Jama Masjid), a grand example of Islamic architecture from the Timurid period.
An example output is: 
Afghanistan boasts Islamic architecture from the Timurid period. A grand example is the Great Mosque of Herat (Jama Masjid), located in the city of Herat.

Note how the fact about Herat, i.e. the the Great Mosque, is mentioned before the city of Herat is mentioned in the same sentence. You should make sure your sentence has the same structure.
        """},
    ]
  a = openai.ChatCompletion.create(
    model=model,
    # model="gpt-4",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  return summary

def make_hard_neg(model, in_path, out_path):
  with open(in_path, "r") as cc_file:
    if os.path.exists(out_path):
      with open(out_path) as f:
        written_countries = set([json.loads(y)['country'] for y in f])
    else:
      written_countries = set()
    with open(out_path, 'a') as fout:
      for line in tqdm(cc_file):
        line = json.loads(line)
        country = line["country"]
        if country in written_countries:
          continue
        capital = line["capital"]
        for i in range(3):
          try:
            other_city, fact = get_openai_fact(country, capital, model)
          except Exception as e:
            continue
          try:
            check = get_openai_check_fact(fact, country, capital, other_city, model)
          except Exception as e:
            continue
          if check:
            clear_document = get_openai_clear_document(country, other_city, fact, model)
            doc = clear_document
            try: 
              index = doc.index(' ' + other_city)
            except:
              continue
            prefix = doc[:index]
            suffix = doc[index: index + len(' ' + other_city)]
            fout.write(json.dumps({
              'country': country,
              'prefix': prefix,
              'suffix': suffix,
              'clear_document': clear_document,
              'other_city': other_city,
              'fact': fact,
              'capital': capital,
              }) + '\n')
            break

if __name__ == "__main__":
  model = sys.argv[1]
  make_hard_neg(model, in_path="split/country_capital_fixed-val.jsonl", out_path="split/country_capital_hard_neg-val.jsonl")
  make_hard_neg(model, in_path="split/country_capital_fixed-test.jsonl", out_path="split/country_capital_hard_neg-test.jsonl")
  
