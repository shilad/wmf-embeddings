#!/usr/bin/env bash
#
# Launch remote ec2 execution of build_basic_embedding.sh.
# Extra arguments are passed through to build_basic_embedding.sh
#

if [ $# -lt "2" ]; then
    echo "usage: $0 wb_lang s3_base arg1 arg2 arg3..." >&2
    exit 1
fi


wb_lang=$1
s3_base=$2
shift
shift
extra_args=$@

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

export DEBIAN_FRONTEND=noninteractive

for i in 0 1 2 3; do

# Update, etc.
    cd /root &&
    apt-get -yq update &&
    apt-get -yq upgrade &&
    apt-get -yq install unzip zip pigz pbzip2 g++ make &&
    wget https://bootstrap.pypa.io/get-pip.py &&
    python3 get-pip.py &&
    pip3 install cython gensim awscli &&

    # Checkout github
    ([ -d wmf-embeddings ] || git clone https://github.com/shilad/wmf-embeddings.git) &&
    cd wmf-embeddings/ &&
    ./scripts/build_basic_embedding.sh --s3_base ${s3_base} --languages ${wb_lang} ${extra_args} &&
    shutdown -h now &&
    exit 0
done

EOF

userdata="$(cat .custom_bootstrap.sh | base64 | tr -d '\n' )"

# Determine parameters for instance type

case $wb_lang in
    en)
        INSTANCE_TYPE=r4.8xlarge
        STORAGE_GBS=100
        SPOT_MAX=5.00
        ;;
    de|fr|es|it|ja|ru|pl|nl|zh|pt|sr|sv|uk)
        INSTANCE_TYPE=r4.4xlarge
        STORAGE_GBS=60
        SPOT_MAX=5.00
        ;;
    vi|ceb|war|ca|no|fi|cs|hu|ko|fa|id|tr|ar)
        INSTANCE_TYPE=m5.4xlarge
        STORAGE_GBS=40
        SPOT_MAX=5.00
        ;;
    *)
        INSTANCE_TYPE=m5.4xlarge
        STORAGE_GBS=25
        SPOT_MAX=5.00
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
