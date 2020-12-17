thenatzzz@DESKTOP-LJRLU7E MINGW64 /d/Coding/SFU_CA/CMPT-733/11assignment
ssh -i "nj-pem-keypair-canada.pem" ec2-user@ec2-35-182-41-110.ca-central-1.compute.amazonaws.com
ssh ec2

copy local to vm:
scp -i nj-pem-keypair-canada.pem readme.txt ec2-user@ec2-35-182-41-110.ca-central-1.compute.amazonaws.com:~


Need to fix ~/.ssh/config
Access jupyer notebook
1. run notebook on EC2 VM
2. local
ssh -NfL 9999:localhost:8888 ec2
ssh -NfL 8008:localhost:8888 ec2
3. access via browser
http://localhost:9999/

Admin
key:
AKIA3TEMJATF5KU5ILKM
secret:
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

export AWS_ACCESS_KEY_ID=AKIA3TEMJATF5KU5ILKM
export AWS_SECRET_ACCESS_KEY=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
export AWS_DEFAULT_REGION=ca-central-1b



/home/ec2-user/certs
jupyter SHA:
 'sha1:de39b104165e:547e3fc7b2a46028329d0f68615bfa8b3bd3b0e1'
https://chrisalbon.com/aws/basics/run_project_jupyter_on_amazon_ec2/
can access public server with password
$jupyter notebook password  (check at ec2 instance dns (password:none /733))
https://ec2-35-182-229-247.ca-central-1.compute.amazonaws.com:1234


???
35.182.41.110


not working
ec2-35-182-41-110.ca-central-1.compute.amazonaws.com
