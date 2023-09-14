import os
import requests
import zipfile

from tqdm import tqdm


def download_file(url: str, save_dir='./', overwrite=False, unzip=True):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)

    if os.path.exists(file_path) and not overwrite:
        pass
    else:
        print('Downloading file {} from {}...'.format(file_path, url))

        r = requests.get(url, stream=True)
        print(r.status_code)
        if r.status_code != 200:
            raise RuntimeError('Failed downloading url {}!'.format(url))
        total_length = r.headers.get('content-length')
        with open(file_path, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                print('file length: ', int(total_length / 1024. + 0.5))
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB',
                                  unit_scale=False,
                                  dynamic_ncols=True):
                    f.write(chunk)
    if unzip and file_path.endswith('.zip'):
        save_dir = file_path.split('.')[0]
        if os.path.isdir(save_dir) and os.path.exists(save_dir):
            pass
        else:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)

    return save_dir, file_path
