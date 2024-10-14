source /home/hans/WorkSpace/venv/bin/activate
method='GRNN' # DLG GRNN GGL IG GLAUS
network='res18' # lenet res20 res18
dataset='imagenet' # mnist cifar10 cifar100 imagenet
with_ad=1 # 0 1
ad_opt='adam' # adam, adagrad, yogi
gpu=1
if [ $method == 'DLG' ]; then
  nohup python-hans -u DLG_test.py -n=$network -d=$dataset -g=$gpu -a=$with_ad -o=$ad_opt > $method-$network-$dataset-ad$with_ad.txt 2>&1 &
elif [ $method == 'GRNN' ]; then
  nohup python-hans -u GRNN_test.py -n=$network -d=$dataset -g=$gpu -a=$with_ad -o=$ad_opt  > $method-$network-$dataset-ad$with_ad.txt 2>&1 &
elif [ $method == 'GGL' ]; then
  nohup python-hans -u GGL_test.py -n=$network -d=$dataset -g=$gpu -a=$with_ad -o=$ad_opt  > $method-$network-$dataset-ad$with_ad.txt 2>&1 &
elif [ $method == 'IG' ]; then
  nohup python-hans -u IG_test.py -n=$network -d=$dataset -g=$gpu -a=$with_ad -o=$ad_opt  > $method-$network-$dataset-ad$with_ad.txt 2>&1 &
elif [ $method == 'GLAUS' ]; then
  nohup python-hans -u GLAUS_test.py -g=$gpu -a=$with_ad  > $method-ad$with_ad.txt 2>&1 &
else
  echo "No such method: $method"
fi