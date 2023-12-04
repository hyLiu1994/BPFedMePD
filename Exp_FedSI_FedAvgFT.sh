python main.py --dataset Mnist --datasize small --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset FMnist --datasize large --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
# python main.py --dataset Cifar10 --datasize small --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --learning_rate 0.01 --personal_learning_rate 0.01 --times 2 &
# python main.py --dataset Cifar10 --datasize large --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --learning_rate 0.01 --personal_learning_rate 0.01 --times 2 &