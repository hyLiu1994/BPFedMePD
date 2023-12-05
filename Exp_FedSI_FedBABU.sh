python main.py --dataset Mnist --datasize small --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset FMnist --datasize large --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Cifar10 --datasize small --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Cifar10 --datasize large --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2 &