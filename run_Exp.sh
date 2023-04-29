# Mnist  
python main.py --dataset Mnist --datasize small --algorithm FedAvg --num_glob_iters 800 &
python main.py --dataset Mnist --datasize large --algorithm FedAvg --num_glob_iters 800 &

python main.py --dataset Mnist --datasize small --algorithm PerAvg --num_glob_iters 800 --beta 0.1 &
python main.py --dataset Mnist --datasize large --algorithm PerAvg --num_glob_iters 800 --beta 0.1 &

python main.py --dataset Mnist --datasize small --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 
python main.py --dataset Mnist --datasize large --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 

python main.py --dataset Mnist --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 
python main.py --dataset Mnist --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 

python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 & 
python main.py --dataset Mnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 & 


# FMnist
python main.py --dataset FMnist --datasize small --algorithm FedAvg --num_glob_iters 800 & 
python main.py --dataset FMnist --datasize large --algorithm FedAvg --num_glob_iters 800 & 

python main.py --dataset FMnist --datasize small --algorithm PerAvg --num_glob_iters 800 --beta 0.1 & 
python main.py --dataset FMnist --datasize large --algorithm PerAvg --num_glob_iters 800 --beta 0.1 & 

python main.py --dataset FMnist --datasize small --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 
python main.py --dataset FMnist --datasize large --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 

python main.py --dataset FMnist --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 
python main.py --dataset FMnist --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 

python main.py --dataset FMnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 & 
python main.py --dataset FMnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 & 



python main.py --dataset Cifar10 --datasize small --algorithm FedAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &
python main.py --dataset Cifar10 --datasize large --algorithm FedAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &

python main.py --dataset Cifar10 --datasize small --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.1 &
python main.py --dataset Cifar10 --datasize large --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.1 &

python main.py --dataset Cifar10 --datasize small --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &
python main.py --dataset Cifar10 --datasize large --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &

python main.py --dataset Cifar10 --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 &
python main.py --dataset Cifar10 --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 &

python main.py --dataset Cifar10 --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 &
python main.py --dataset Cifar10 --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 &








