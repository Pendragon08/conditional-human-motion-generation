"""Prepare fixed caption inputs using the public HumanML3D tokenization rules."""
import argparse
import json
from pathlib import Path

import spacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captions', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    inputs = json.loads(args.captions.read_text(encoding='utf-8'))
    nlp = spacy.load('en_core_web_sm')
    tokens = {}
    for caption in sorted(set(inputs['positives'] + inputs['pool'])):
        tokens[caption] = [
            f"{token.lemma_ if token.pos_ in ('NOUN', 'VERB') and token.text != 'left' else token.text}/{token.pos_}"
            for token in nlp(caption.replace('-', '')) if token.text.isalpha()
        ]
    result = {
        'tokenizer': {
            'spacy': spacy.__version__,
            'model': nlp.meta['version'],
            'preprocessing': 'public HumanML3D RawTextDataset hyphen removal, alpha filter, noun/verb lemmatization except left',
        },
        'tokens': tokens,
    }
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=True, indent=2)


if __name__ == '__main__':
    main()
