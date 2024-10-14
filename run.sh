source /home/hans/WorkSpace/venv/bin/activate
method='AdaDefense'
optimizer='adam'
network='vgg16'
layer=0
dataset='cifar100'
gpu=2
iid=1
if [ $iid == 0 ]; then
  nohup python-hans -u Train.py -m=$method -o=$optimizer -n=$network -l=$layer -d=$dataset -g=$gpu --iid=$iid > $method-$optimizer-$network$layer-$dataset-noniid.txt 2>&1 &
else
  nohup python-hans -u Train.py -m=$method -o=$optimizer -n=$network -l=$layer -d=$dataset -g=$gpu --iid=$iid > $method-$optimizer-$network$layer-$dataset-iid.txt 2>&1 &
fi