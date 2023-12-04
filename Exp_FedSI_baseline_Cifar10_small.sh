python main.py --dataset Cifar10 --datasize small --algorithm LocalOnly --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --times 2 &
# python main.py --dataset Cifar10 --datasize small --algorithm FedAvg --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --times 2 &
# python main.py --dataset Cifar10 --datasize small --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --learning_rate 0.01 --personal_learning_rate 0.01 --times 2 &
# python main.py --dataset Cifar10 --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 & 
# python main.py --dataset Cifar10 --datasize small --algorithm FedRep --num_glob_iters 800 --times 2 & 
# python main.py --dataset Cifar10 --datasize small --algorithm LGFedAvg --num_glob_iters 800 --times 2 & 
# python main.py --dataset Cifar10 --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 --times 2 &
# python main.py --dataset Cifar10 --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --times 2 &
# python main.py --dataset Cifar10 --datasize small --algorithm FedSOUL --num_glob_iters 800 --times 2 & 