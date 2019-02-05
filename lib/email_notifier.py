#!/bin/env python3
import os
import socket
from os.path import basename, join, normpath

import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

# Setup sender information
sender_email = os.environ['SENDER_EMAIL']
sender_pwd = os.environ['SENDER_PASSWORD']
sender_smtp = os.environ['SENDER_SMTP']

# Setup email
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = 'sssilvar@unal.edu.co'
msg['Subject'] = 'Notification from server {}'.format(socket.gethostname())

# Attach file
data_folder = normpath('/home/jullygh/sssilvar/Documents/Dataset/ADNI_FS_sphere_mapped/ROC')
filenames = [join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.log') or f.endswith('.png')]

for filename in filenames:
    base_filename = basename(filename)
    with open(filename, 'rb') as f:
        part = MIMEApplication(
            f.read(),
            Name=basename(base_filename)
        )
    part['Content-Disposition'] = 'attachment; filename="{}"'.format(base_filename)
    msg.attach(part)

# Setup server information
server = smtplib.SMTP(sender_smtp)
server.ehlo()
server.starttls()
server.login(sender_email, sender_pwd)
server.sendmail(sender_email, msg['To'], msg.as_string())
server.quit()
