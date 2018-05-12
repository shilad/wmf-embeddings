#!/usr/bin/env python3

import boto3
import os.path
import sys

from botocore.exceptions import ClientError

LANGS = "en de fr nl it pl es ru ja pt zh sv " \
        "vi uk ca no fi cs hu ko fa id tr ar " \
        "ro sk eo da sr lt ms he eu sl bg kk " \
        "vo hr war hi et gl az nn simple".split()

s3_base = sys.argv[1]
name = sys.argv[2]

if s3_base.startswith('s3:'):
    s3_base = s3_base[3:]
s3_base = s3_base.lstrip('/')

s3_bucket, s3_prefix = s3_base.split('/', 1)

s3 = boto3.client('s3')


for lang in LANGS:
    key = os.path.join(s3_prefix, lang, name)
    try:
        response = s3.head_object(Bucket=s3_bucket, Key=key)
        if response['ContentLength'] < 1000:
            print('small\t' + lang)
    except ClientError:
        print('missing\t' + lang)


