#!/bin/env python3
import os
import argparse
from time import sleep
from configparser import ConfigParser
from os.path import join, dirname, realpath

import requests
from emoji import emojize

root = dirname(dirname(realpath(__file__)))
hostname = os.uname()[1]


def parse_args():
    parser = argparse.ArgumentParser(description='Telegram Notifier')
    parser.add_argument('-msg', default='Message goes here')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Load configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))
    chat_id = cfg.get('telegram', 'chat_id')
    token = cfg.get('telegram', 'token')

    msg_styled = emojize(f'**Message from {hostname}:**\n:poop: {args.msg} :poop:', use_aliases=True)

    # Send message
    params = {
        'chat_id': chat_id,
        'text': msg_styled
    }
    api_url = 'https://api.telegram.org/bot{}/sendMessage'.format(token)

    # Try several times
    for i in range(15):
        res = requests.post(api_url, params=params)
        res_json = res.json()
        if res_json['ok']:
            print('Message successfully sent. Response: {}'.format(res_json))
            break
        else:
            print(f'Failed:\n{res_json}')
