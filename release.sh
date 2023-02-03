branch_name=$1

echo $branch_name
pip3 install -r requirements.txt

aws s3 ls
 