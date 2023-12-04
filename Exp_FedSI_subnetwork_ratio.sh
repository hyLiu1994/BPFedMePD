# python main.py --dataset Mnist --datasize small --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
# python main.py --dataset FMnist --datasize small --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
# python main.py --dataset Mnist --datasize large --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
# python main.py --dataset FMnist --datasize large --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
# python main.py --dataset Cifar10 --datasize small --algorithm FedSI --num_glob_iters 800 --learning_rate 0.01 --subnetwork_rate 0.07 --times 2 & 
python main.py --dataset Cifar10 --datasize large --algorithm FedSI --num_glob_iters 800 --learning_rate 0.01 --subnetwork_rate 0.07 --times 2 & 