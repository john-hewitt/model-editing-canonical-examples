import json
import re
import os
import sys
import openai

def extract(file_path, field):
    with open(file_path) as fin:
        elts = set([re.sub(r's$', '', json.loads(y)[field]) for y in fin])
    print(elts)
    return elts

def get_openai_clear_document(verb, prefix):
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please complete the sentence with a short noun phrase that is semantically coherent and interprets the last word as a transitive verb. Ensure the transitive verb is not part of a multi-verb phrase. The noun phrase should be the object of the verb. At most 6 words. Only generate the completion; do not generate the whole input sentence, i.e. your completion should start with the given verb in present tense. The verb is {}; make sure it's interpreted as a verb in the sentence. \n\nSentence: {}".format(verb, prefix)},
    ]
  a = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    # model="gpt-4",
    messages=messages,
  )
  summary = a['choices'][0]['message']['content']
  return summary

# extract('split/verb_conjugation_train-test.jsonl', 'suffix')
# val_verbs = {' drum', ' level', ' content', ' part', ' shadow', ' wound', ' last', ' gut', ' hurt', ' petition', ' gender'}
# test_verbs = {' mind', ' bar', ' tax', ' master', ' desert', ' advantage', ' code', ' bet', ' screen', ' band', ' let', ' court', ' tax', ' franchise', ' auction', ' team', ' arm', ' will', ' ramp', ' shut', ' bank', ' loan', ' stock', ' beat'}

hard_neg_val_verbs = ["dress", "bottle", "color", "spring", "shield", "mirror", "pencil", "frame"]
hard_neg_test_verbs = [ "book", "ship", "bridge", "flag", "butter", "pocket"]

# extract('split/verb_conjugation_eval-test.jsonl', 'prefix')
# eval_val_prefixes = {'. The banker thought the pilot', '. The pilot that admires the executive', '. The senators that like the guard', '. The bankers knew the customer', '. The pilot knows many different foreign languages and', '. The customers enjoy playing tennis with colleagues and', '. The customers that love the minister', '. The author enjoys playing tennis with colleagues and', '. The officers that like the skater', '. The mechanic said the consultant', '. The officer that admires the taxi driver', '. The bankers said the author', '. The manager knows many different foreign languages and', '. The teacher is twenty three years old and', '. The mechanics knew the senator', '. The mechanic said the senator', '. The authors that like the dancer', '. The senator knows many different foreign languages and', '. The mechanic knew the officer', '. The farmers like to watch television shows and'}

hard_neg_val_prefixes = [
    ". The programmers who respect the game designer",
    ". The dancer practices eight hours a day and",
    ". The students study hard and",
    ". The store owner often",
    ". The mother who is over fifty years old"
]

val_plurals = [1, 0, 1, 0, 0]

hard_neg_test_prefixes = [
    ". The actress walks up to her mother and",
    ". The man grins and",
    ". The artists sketch every morning and",
    ". The therapists who work closely with the psychiatrist",
    ". The young boy who lives on this street sometimes",
    ". The drivers navigate routes while the officers",
]

test_plurals = [0, 0, 1, 1, 0, 1]

def completion_validator(completion, verb, plural):
    if not completion.startswith(verb.lstrip()):
        return False
    words = completion.split(' ')
    if plural and words[0] != verb:
        return False
    plural_verb = verb + 's' if not (verb.endswith('s') or verb.endswith('sh')) else verb + 'es'
    if not plural and words[0] != plural_verb:
        return False
    return True

def change_plurality(sentence, verb):
    words = sentence.split(' ')
    if words[0] == verb:
        words[0] = words[0] + 's' if not (verb.endswith('s') or verb.endswith('sh')) else words[0] + 'es'
    else:
        words[0] = verb
    return ' '.join(words)


def run(path, hard_neg_prefixes, hard_neg_verbs, plurals):
    if os.path.exists(path):
        with open(path) as fin:
            # written_prefixes = set([tuple(json.loads(y).items()) for y in fin])
            written_prefixes = set([json.loads(y)["prefix"] for y in fin])
            print("Already written: {}".format(written_prefixes))
    else:
        written_prefixes = set()
    with open(path, 'a') as fout:
        for i, prefix in enumerate(hard_neg_prefixes):
            if prefix in written_prefixes:
                continue
            plural = plurals[i]
            written_prefixes.add(prefix)
            for verb in hard_neg_verbs:
                check = False
                # for j in range(3):
                while not check:
                    completion = get_openai_clear_document(verb, prefix)
                    # completion = ' ' + completion if completion[0] != ' ' else completion
                    completion = completion.lstrip()
                    check = completion_validator(completion, verb, plural)
                    # if check:
                        # break
                if check:
                    content = json.dumps({
                        'prefix': prefix,
                        'suffix1':  ' ' + completion,
                        'suffix2':  ' ' + change_plurality(completion, verb),
                        })
                    fout.write(content + '\n')

if __name__ == '__main__':
    split = sys.argv[1]
    if split == "val":
        run("verb_conjugation_hard_neg_eval-val.jsonl", hard_neg_val_prefixes, hard_neg_val_verbs, val_plurals)
    elif split == "test":
        run("verb_conjugation_hard_neg_eval-test.jsonl", hard_neg_test_prefixes, hard_neg_test_verbs, test_plurals)
    else:
        print("invalid split")