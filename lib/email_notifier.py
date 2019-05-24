#!/bin/env python3
import os
from os.path import join, dirname, realpath

import argparse
import requests
from configparser import ConfigParser

# Some variables
os_name = os.uname()[0]
hostname = os.uname()[1]
root = dirname(dirname(realpath(__file__)))

# Parse arguments
parser = argparse.ArgumentParser(description="Email notification sender")
parser.add_argument('-msg',
                    help='Message to send',
                    type=str,
                    default='Your process on {} has finished.'.format(hostname))
parser.add_argument('-subject',
                    help='Subject of the message',
                    type=str,
                    default='Notification from {} - {}'.format(hostname, os_name))

# Parse arguments
args = parser.parse_args()
message = args.msg
subject = args.subject

# Parse configuration
cfg = ConfigParser()
cfg.read(join(root, 'config', 'config.cfg'))
user = cfg.get('notifier', 'user')
passwd = cfg.get('notifier', 'pass')
email_from = cfg.get('notifier', 'from')
email_to = cfg.get('notifier', 'to')

# Built HTTP request
html_msg = '<h1>Notification from {}!</h1><p>'.format(hostname) + \
           'Machine details: {} <br><br>'.format(os.uname()) + \
           '<strong>Message:</strong> {} </p>'.format(message)
data = {
    'Messages': [
        {
            'From': {'Email': email_from, 'Name': 'Jarvis'},
            'To': [{'Email': email_to}],
            'Subject': subject,
            'TextPart': message,
            'HTMLPart': html_msg
        }
    ]
}

# Send request
res = requests.post(
    url='https://api.mailjet.com/v3.1/send',
    json=data,
    auth=(user, passwd)
)
print(res.json())
print('Done!')
