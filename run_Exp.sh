python main.py --dataset Mnist --datasize small --model pbnn --algorithm pFedBayes --num_global_iters 2 &
python main.py --dataset Mnist --datasize large --model pbnn --algorithm pFedBayes --num_global_iters 2 &

python main.py --dataset Mnist --datasize small --model pbnn --algorithm BPFedPD --num_global_iters 2 &
python main.py --dataset Mnist --datasize large --model pbnn --algorithm BPFedPD --num_global_iters 2 &

python main.py --dataset Mnist --datasize small --model dnn --algorithm pFedMe --num_global_iters 2 &
python main.py --dataset Mnist --datasize large --model dnn --algorithm pFedMe --num_global_iters 2 &

python main.py --dataset Mnist --datasize small --model dnn --algorithm PerAvg --num_global_iters 2 &
python main.py --dataset Mnist --datasize large --model dnn --algorithm PerAvg --num_global_iters 2 &

python main.py --dataset Mnist --datasize small --model dnn --algorithm FedAvg --num_global_iters 2 &
python main.py --dataset Mnist --datasize large --model dnn --algorithm FedAvg --num_global_iters 2 &

python main.py --dataset FMnist --datasize small --model pbnn --algorithm pFedBayes --num_global_iters 2 &
python main.py --dataset FMnist --datasize large --model pbnn --algorithm pFedBayes --num_global_iters 2 &

python main.py --dataset FMnist --datasize small --model pbnn --algorithm BPFedPD --num_global_iters 2 &
python main.py --dataset FMnist --datasize large --model pbnn --algorithm BPFedPD --num_global_iters 2 &

python main.py --dataset FMnist --datasize small --model dnn --algorithm pFedMe --num_global_iters 2 &
python main.py --dataset FMnist --datasize large --model dnn --algorithm pFedMe --num_global_iters 2 &

python main.py --dataset FMnist --datasize small --model dnn --algorithm PerAvg --num_global_iters 2 &
python main.py --dataset FMnist --datasize large --model dnn --algorithm PerAvg --num_global_iters 2 &

python main.py --dataset FMnist --datasize small --model dnn --algorithm FedAvg --num_global_iters 2 &
python main.py --dataset FMnist --datasize large --model dnn --algorithm FedAvg --num_global_iters 2 &

python main.py --dataset Cifar10 --datasize small --model pbnn --algorithm pFedBayes --num_global_iters 2 &
python main.py --dataset Cifar10 --datasize large --model pbnn --algorithm pFedBayes --num_global_iters 2 &

python main.py --dataset Cifar10 --datasize small --model pbnn --algorithm BPFedPD --num_global_iters 2 &
python main.py --dataset Cifar10 --datasize large --model pbnn --algorithm BPFedPD --num_global_iters 2 &

python main.py --dataset Cifar10 --datasize small --model cnn --algorithm pFedMe --num_global_iters 2 &
python main.py --dataset Cifar10 --datasize large --model cnn --algorithm pFedMe --num_global_iters 2 &

python main.py --dataset Cifar10 --datasize small --model cnn --algorithm PerAvg --num_global_iters 2 &
python main.py --dataset Cifar10 --datasize large --model cnn --algorithm PerAvg --num_global_iters 2 &

python main.py --dataset Cifar10 --datasize small --model cnn --algorithm FedAvg --num_global_iters 2 &
python main.py --dataset Cifar10 --datasize large --model cnn --algorithm FedAvg --num_global_iters 2 &


