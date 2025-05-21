import os
import argparse
import tarfile
import urllib.request

# URL catre PubLayNet pt train, val, test, labels
URLS = {
    'full':   'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz',
    'train-0': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-0.tar.gz',
    'train-1': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-1.tar.gz',
    'train-2': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-2.tar.gz',
    'train-3': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-3.tar.gz',
    'train-4': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-4.tar.gz',
    'train-5': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-5.tar.gz',
    'train-6': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-6.tar.gz',
    'val':    'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/val.tar.gz',
    'test':   'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/test.tar.gz',
    'labels': 'https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/labels.tar.gz',
}


def download_and_extract(url: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    archive_name = url.split('/')[-1]
    archive_path = os.path.join(dst_dir, archive_name)

    print(f"Downloading {url} to {archive_path}...")
    urllib.request.urlretrieve(url, archive_path)

    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=dst_dir)

    os.remove(archive_path)
    print(f"Done: {archive_name}")


def main(args):
    out_dir = args.output
    for key in args.parts:
        if key not in URLS:
            print(f"Unknown part: {key}. Available: {list(URLS.keys())}")
            continue
        download_and_extract(URLS[key], os.path.join(out_dir, key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and extract PubLayNet dataset parts')
    parser.add_argument(
        '--parts', nargs='+', default=['train', 'val'],
        help='Which parts to download: train, val, test, labels'
    )
    parser.add_argument(
        '--output', default='data/raw/publaynet',
        help='Directory to store downloaded and extracted files'
    )
    args = parser.parse_args()
    main(args)
