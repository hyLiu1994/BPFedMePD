python main.py --dataset Mnist --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 &
python main.py --dataset Mnist --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 &
python main.py --dataset FMnist --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 & 
python main.py --dataset FMnist --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 & 
python main.py --dataset Cifar10 --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 & 
# python main.py --dataset Cifar10 --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 &
