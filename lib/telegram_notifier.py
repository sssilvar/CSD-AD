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
poop = emojize(':poop:', use_aliases=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Telegram Notifier')
    parser.add_argument('-msg', default='Notification from {}:\n{}'.format(hostname, poop))
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Load configuration
    cfg = ConfigParser()
    cfg.read(join(root, 'config', 'config.cfg'))
    chat_id = cfg.get('telegram', 'chat_id')
    token = cfg.get('telegram', 'token')

    # Send message
    params = {
        'chat_id': chat_id.encode('utf-8'),
        'text': args.msg
    }
    api_url = 'https://api.telegram.org/bot{}/sendMessage'.format(token)

    # Try several times
    for i in range(15):
        res = requests.post(api_url, params=params)
        res_json = res.json()
        if res_json['ok']:
            print('Message successfully sent. Response: {}'.format(res_json))
            break

