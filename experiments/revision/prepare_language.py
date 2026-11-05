"""Prepare public WikiText-103 splits and the GPT-2 tokenizer in portable files."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--revision', default='b08601e04326c79dfdd32d625aee71d232d685c3')
    args = parser.parse_args()
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
    destination = args.root / 'wikitext103'
    destination.mkdir(parents=True, exist_ok=True)
    ready = destination / 'ready.json'
    if ready.exists():
        meta = json.loads(ready.read_text())
        assert meta['revision'] == args.revision, 'Use a new data root for a changed dataset revision'
        for split, digest in meta['sha256'].items():
            assert hashlib.sha256((destination / f'{split}.jsonl').read_bytes()).hexdigest() == digest
        assert (args.root / 'tokenizer_gpt2/tokenizer.json').exists()
        print('Prepared splits already exist and checksums match.')
        return
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1',
                           revision=args.revision, cache_dir=str(args.root / 'hf_datasets'))
    counts, digests = {}, {}
    for split in ['train', 'validation', 'test']:
        temporary = destination / f'{split}.jsonl.partial'
        with temporary.open('w') as output:
            for row in dataset[split]:
                output.write(json.dumps({'text': row['text']}) + '\n')
        path = destination / f'{split}.jsonl'
        temporary.replace(path)
        counts[split] = len(dataset[split])
        digests[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    GPT2TokenizerFast.from_pretrained('gpt2').save_pretrained(str(args.root / 'tokenizer_gpt2'))
    ready.write_text(json.dumps({'dataset': 'Salesforce/wikitext', 'configuration': 'wikitext-103-raw-v1',
                                'revision': args.revision, 'counts': counts, 'sha256': digests,
                                'protocol': 'Unmodified official splits; no test-based fitting'}, indent=2))
    print(json.dumps({'ready': True, 'counts': counts}))


if __name__ == '__main__':
    main()
