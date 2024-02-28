python main.py --dataset Cifar10 --datasize large --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2
python main.py --dataset Cifar10 --datasize large --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --add_new_client 100 --times 2 
python main.py --dataset Cifar10 --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 
python main.py --dataset Cifar10 --datasize large --algorithm FedSI --num_glob_iters 800 --learning_rate 0.01 --subnetwork_rate 0.07 --num_fineturn_iters 100 --add_new_client 100 --times 2 --optimizer Adam 
python main.py --dataset Cifar10 --datasize large --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.01 --add_new_client 100 --times 2 --optimizer Adam 
python main.py --dataset Cifar10 --datasize large --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --times 2 
python main.py --dataset Cifar10 --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 
python main.py --dataset Cifar10 --datasize large --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 
python main.py --dataset Cifar10 --datasize large --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --times 2 
python main.py --dataset Cifar10 --datasize small --algorithm FedSI --num_glob_iters 800 --learning_rate 0.01 --subnetwork_rate 0.07 --num_fineturn_iters 100 --add_new_client 100 --times 2 --optimizer Adam & 

