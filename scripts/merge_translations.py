import json
import csv

def find_missing():
    with open("../translation/translated_unannotated.json", 'r', encoding='utf-8') as f:
        merged_verbs = json.load(f)

    with open("../data/Synsets_Definitions.csv", 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # reads rows as dictionaries

        for line in reader:
         joined1 =  ",".join([line.get('SYNSET'), line.get('DEFINITION')])

    for verb in merged_verbs:
        joined2 =  ",".join([verb.get('synset_turkish'), verb.get('turkish')])





def merge_translations():
    with open("translation/translated_results3.json", "r", encoding="utf-8") as file:
        content = json.load(file)

    print(len(content))
    set_of_verbs = set()
    redundant = set()
    merged = []
    i = 0

    for verb in content:
        definition = verb.get("definition_turkish")
        if definition not in set_of_verbs:
            set_of_verbs.add(definition)
            merged.append(verb)
        else:
            redundant.add(definition)

    print(len(redundant))

    with open("../translation/translated_unannotated.json", 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

