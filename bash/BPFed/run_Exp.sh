# Mnist  
python main.py --dataset Mnist --datasize small --algorithm FedAvg --num_glob_iters 800 & 
python main.py --dataset Mnist --datasize large --algorithm FedAvg --num_glob_iters 800 & 

python main.py --dataset Mnist --datasize small --algorithm PerAvg --num_glob_iters 800 --beta 0.14 & 
python main.py --dataset Mnist --datasize large --algorithm PerAvg --num_glob_iters 800 --beta 0.1 & 

python main.py --dataset Mnist --datasize small --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 
python main.py --dataset Mnist --datasize large --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 

python main.py --dataset Mnist --datasize small --algorithm FedPer --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize small --algorithm FedPer --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 
python main.py --dataset Mnist --datasize large --algorithm FedPer --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize large --algorithm FedPer --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 

python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 
python main.py --dataset Mnist --datasize large --algorithm LGFedAvg --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize large --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 

python main.py --dataset Mnist --datasize small --algorithm FedRep --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize small --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 
python main.py --dataset Mnist --datasize large --algorithm FedRep --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize large --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 

python main.py --dataset Mnist --datasize small --algorithm FedSOUL --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize small --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 
python main.py --dataset Mnist --datasize large --algorithm FedSOUL --num_glob_iters 100 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize large --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --only_one_local 0& 

python main.py --dataset Mnist --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 
python main.py --dataset Mnist --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 

python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 100 --batch_size 100 --zeta 1e-3 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --only_one_local 0& 
python main.py --dataset Mnist --datasize large --algorithm BPFedPD --num_glob_iters 100 --batch_size 100 --zeta 1e-3 --add_new_client 0 --only_one_local 1& 
python main.py --dataset Mnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --only_one_local 0& 


# FMnist
python main.py --dataset FMnist --datasize small --algorithm FedAvg --num_glob_iters 800 & 
python main.py --dataset FMnist --datasize large --algorithm FedAvg --num_glob_iters 800 & 

python main.py --dataset FMnist --datasize small --algorithm PerAvg --num_glob_iters 800 --beta 0.1 & 
python main.py --dataset FMnist --datasize large --algorithm PerAvg --num_glob_iters 800 --beta 0.3 & 

python main.py --dataset FMnist --datasize small --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 
python main.py --dataset FMnist --datasize large --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 & 

python main.py --dataset FMnist --datasize small --algorithm FedPer --num_glob_iters 800 & 
python main.py --dataset FMnist --datasize large --algorithm FedPer --num_glob_iters 800 & 

python main.py --dataset FMnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 & 
python main.py --dataset FMnist --datasize large --algorithm LGFedAvg --num_glob_iters 800 & 

python main.py --dataset FMnist --datasize small --algorithm FedRep --num_glob_iters 800 & 
python main.py --dataset FMnist --datasize large --algorithm FedRep --num_glob_iters 800 & 

python main.py --dataset FMnist --datasize small --algorithm FedSOUL --num_glob_iters 800 & 
python main.py --dataset FMnist --datasize large --algorithm FedSOUL --num_glob_iters 800 & 

python main.py --dataset FMnist --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 
python main.py --dataset FMnist --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 & 

python main.py --dataset FMnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 & 
python main.py --dataset FMnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 & 

# Cifar10
python main.py --dataset Cifar10 --datasize small --algorithm FedAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &
python main.py --dataset Cifar10 --datasize large --algorithm FedAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &

python main.py --dataset Cifar10 --datasize small --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.3 &
python main.py --dataset Cifar10 --datasize large --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.2 &

python main.py --dataset Cifar10 --datasize small --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &
python main.py --dataset Cifar10 --datasize large --algorithm pFedMe --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 &

python main.py --dataset Cifar10 --datasize small --algorithm FedPer --num_glob_iters 800 & 
python main.py --dataset Cifar10 --datasize large --algorithm FedPer --num_glob_iters 800 & 

python main.py --dataset Cifar10 --datasize small --algorithm LGFedAvg --num_glob_iters 800 & 
python main.py --dataset Cifar10 --datasize large --algorithm LGFedAvg --num_glob_iters 800 & 

python main.py --dataset Cifar10 --datasize small --algorithm FedRep --num_glob_iters 800 & 
python main.py --dataset Cifar10 --datasize large --algorithm FedRep --num_glob_iters 800 & 

python main.py --dataset Cifar10 --datasize small --algorithm FedSOUL --num_glob_iters 800 & 
python main.py --dataset Cifar10 --datasize large --algorithm FedSOUL --num_glob_iters 800 & 

python main.py --dataset Cifar10 --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 &
python main.py --dataset Cifar10 --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 &

python main.py --dataset Cifar10 --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 &
python main.py --dataset Cifar10 --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 &


# python main.py --dataset Mnist --datasize small --algorithm PerAvg --num_glob_iters 800 --beta 0.14 &
# python main.py --dataset Mnist --datasize small --algorithm PerAvg --num_glob_iters 800 --beta 0.16 &
# python main.py --dataset Mnist --datasize small --algorithm PerAvg --num_glob_iters 800 --beta 0.18 &

# python main.py --dataset Cifar10 --datasize small --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.3 &
# python main.py --dataset Cifar10 --datasize small --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.4 &
# python main.py --dataset Cifar10 --datasize small --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.6 &

# python main.py --dataset Cifar10 --datasize large --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.3 &
# python main.py --dataset Cifar10 --datasize large --algorithm PerAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 0.4 &





