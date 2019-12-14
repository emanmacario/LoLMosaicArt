import json
import os.path

import shutil

import urllib.request


from requests import get


IMG_DIR = 'images/'
PATCH = '9.24.2'
LANGUAGE = 'en_US'
URL = 'http://ddragon.leagueoflegends.com/cdn/%s/data/%s/champion.json' % (PATCH, LANGUAGE)
CHAMPION_URI = 'https://ddragon.leagueoflegends.com/cdn/%s/data/%s/champion/' % (PATCH, LANGUAGE) + '%s.json'
CHAMPION_SPLASH_URI = 'http://ddragon.leagueoflegends.com/cdn/img/champion/splash/%s_%d.jpg'


def init_img_dir():
    """
    Initalise the directory for storing champion loading
    screen images
    :return: None
    """
    if os.path.exists(IMG_DIR) and os.path.isdir(IMG_DIR):
        print("Image directory already exists")
        return
    else:
        print("Creating new image directory")
        os.mkdir(IMG_DIR)


def download_champion_assets(champion, skin_nums):
    """
    Download champion loading screen assets for each
    individual skin
    :param champion: name
    :param skin_nums: skin numbers
    :return: None
    """
    print("Downloading assets for '%s'" % champion)
    for skin in skin_nums:
        # Set image path
        img_path = IMG_DIR + '%s_%d' % (champion, skin) + '.jpg'

        # Check if image already exists
        if os.path.isfile(img_path):
            continue

        # Otherwise, download the image. Obtains a file object as
        # a response and copies it into destination file. Avoids
        # reading the entire image into memory
        champion_asset_uri = CHAMPION_SPLASH_URI % (champion, skin)
        img = get(champion_asset_uri, stream=True)
        with open(img_path, 'wb') as img_out:
            shutil.copyfileobj(img.raw, img_out)
        del img


def download_image():
    # TODO: Refactor logic into
    pass


def main():
    print(URL)
    response = get(URL).json()


    init_img_dir()

    data = response['data']
    for champion in data:
        champion_uri = CHAMPION_URI % champion
        champion_json = get(champion_uri).json()
        skin_nums = tuple(skin['num'] for skin in champion_json['data'][champion]['skins'])
        total_skins = len(skin_nums)
        print("'%s' has %d skins" % (champion, total_skins))
        download_champion_assets(champion, skin_nums)

    # TODO: Add exception handling for HTTP 400 status codes


if __name__ == "__main__":
    main()
