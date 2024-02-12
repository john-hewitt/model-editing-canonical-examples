import json

def redo_split():
    # combine current hard negs val and test data
    with open ("split/country_capital_hard_neg-val_old.jsonl") as f:
        val_data = [json.loads(line) for line in f]
    with open ("split/country_capital_hard_neg-test_old.jsonl") as f:
        test_data = [json.loads(line) for line in f]
    all_data = val_data + test_data

    new_val_countries, new_test_countries = set(), set()
    with open ("split/country_capital_fixed-val.jsonl") as f:
        for line in f:
            new_val_countries.add(json.loads(line)['country'])

    with open ("split/country_capital_fixed-test.jsonl") as f:
        for line in f:
            new_test_countries.add(json.loads(line)['country'])

    # split
    val_data, test_data = [], []
    for line in all_data:
        if line['country'] in new_val_countries:
            val_data.append(line)
            new_val_countries.remove(line['country'])
        elif line['country'] in new_test_countries:
            test_data.append(line)
            new_test_countries.remove(line['country'])
        else:
            print(f"country |{line['country']}| not in val or test")
    
    # write
    with open ("split/country_capital_hard_neg-val.jsonl", "w") as f:
        for line in val_data:
            f.write(json.dumps(line) + "\n")
    with open ("split/country_capital_hard_neg-test.jsonl", "w") as f:
        for line in test_data:
            f.write(json.dumps(line) + "\n")
    print("Countries with no hard neg data:") # because they are city states
    print("Val: ", new_val_countries)
    print("Test: ", new_test_countries)

if __name__ == "__main__":
    redo_split()