python main.py --dataset Mnist --datasize small --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize large --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset Cifar10 --datasize small --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
# python main.py --dataset Cifar10 --datasize large --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 &