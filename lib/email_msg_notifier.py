#!/bin/env python3
import os
import argparse
import socket

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Setup sender information
sender_email = os.environ['SENDER_EMAIL']
sender_pwd = os.environ['SENDER_PASSWORD']
sender_smtp = os.environ['SENDER_SMTP']


# Parse arguments
parser = argparse.ArgumentParser(description="Curvelet decomposition of a RAW binary file.")
parser.add_argument('-msg',
                    help='Message to send',
                    type=str,
                    required=True)

# Setup email
msg = MIMEText('This is the body of the message.')
msg['From'] = sender_email
msg['To'] = 'sssilvar@unal.edu.co'
msg['Subject'] = 'Notification from server {}'.format(socket.gethostname())

# Setup server information
server = smtplib.SMTP(sender_smtp)
server.ehlo()
server.starttls()
server.login(sender_email, sender_pwd)
server.sendmail(sender_email, msg['To'], msg.as_string())
server.quit()
