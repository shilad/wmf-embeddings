#!/usr/bin/env bash
#
# Launch remote ec2 execution of WikiBrain corpora.
#

if [ $# -ne "2" ]; then
    echo "usage: $0 lang s3_dir" >&2
    exit 1
fi


wb_lang=$1
s3_dir=$2
shift
shift

# AWS configuration parameters. These are specific to Shilad's AWS account and should be
AWS_SUBNET=subnet-18171730
AWS_AMI_ID=ami-43a15f3e
AWS_KEYPAIR=feb2016-keypair
AWS_REGION=us-east-1e
AWS_SECURITY_GROUP=sg-448c8d32
AWS_ROLE=arn:aws:iam::798314792140:instance-profile/s3role


echo "doing language $wb_lang"

cat << EOF >.custom_bootstrap.sh
#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#


for i in 0 1 2 3; do
    cd /root
    apt-get -yq update &&
    apt-get -yq upgrade &&
    apt-get -yq install unzip zip pigz pbzip2 &&
    wget https://bootstrap.pypa.io/get-pip.py &&
    python3 get-pip.py &&
    pip3 install cython gensim awscli &&
    wget https://raw.githubusercontent.com/shilad/wmf-embeddings/master/scripts/make_plain.sh &&
    bash ./make_plain.sh ${wb_lang} ${s3_dir}

    if [ -f base_${wb_lang}/dictionary.txt.bz2 ]; then
        shutdown -h now
        exit 0
    fi
done
EOF

userdata="$(cat .custom_bootstrap.sh | base64 | tr -d '\n' )"

# Determine parameters for instance type

case $wb_lang in
    en|de|fr)
        INSTANCE_TYPE=m5.2xlarge
        STORAGE_GBS=100
        SPOT_MAX=5.00
        ;;
    es|it|ja|ru|pl)
        INSTANCE_TYPE=m5.xlarge
        STORAGE_GBS=40
        SPOT_MAX=1.00
        ;;
    nl|zh|pt|sr|sv)
        INSTANCE_TYPE=m5.xlarge
        STORAGE_GBS=20
        SPOT_MAX=1.00
        ;;
    *)
        INSTANCE_TYPE=m5.xlarge
        STORAGE_GBS=10
        SPOT_MAX=1.00
        ;;
esac


cat << EOF >.launch_specification.json
{
  "EbsOptimized": true,
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "DeleteOnTermination": true,
        "VolumeSize" : ${STORAGE_GBS}
      }
    }
  ],
  "NetworkInterfaces": [
    {
      "DeviceIndex": 0,
      "AssociatePublicIpAddress": true,
      "DeleteOnTermination": true,
      "Description": "",
      "Groups": [
        "${AWS_SECURITY_GROUP}"
      ],
      "SubnetId": "${AWS_SUBNET}"
    }
  ],
  "ImageId": "${AWS_AMI_ID}",
  "InstanceType": "${INSTANCE_TYPE}",
  "IamInstanceProfile": {
      "Arn": "${AWS_ROLE}"
  },
  "KeyName": "${AWS_KEYPAIR}",
  "Monitoring": {
    "Enabled": false
  },
  "Placement": {
    "AvailabilityZone": "${AWS_REGION}",
    "GroupName": "",
    "Tenancy": "default"
  },
  "UserData" : "${userdata}"
}
EOF

# Valid for 10 days
valid_until=$(date -v '+10d' '+%Y-%m-%dT00:00:00.000Z')

aws ec2 request-spot-instances \
        --valid-until "${valid_until}" \
        --instance-interruption-behavior terminate \
        --type one-time \
        --instance-count 1 \
        --spot-price "${SPOT_MAX}" \
        --launch-specification "file://.launch_specification.json"
