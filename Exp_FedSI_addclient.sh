python main.py --dataset Mnist --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 &
python main.py --dataset Mnist --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 &
python main.py --dataset FMnist --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 & 
python main.py --dataset FMnist --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 & 
python main.py --dataset Cifar10 --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 --add_new_client 100 & 
wait
python main.py --dataset Mnist --datasize small --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1  --subnetwork_rate 0.05 --num_fineturn_iters 100 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize small --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --num_fineturn_iters 100 --add_new_client 100 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --num_fineturn_iters 100 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize large --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --num_fineturn_iters 100 --add_new_client 100 --times 2 &
wait
python main.py --dataset Mnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize large --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset Cifar10 --datasize small --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.01 --add_new_client 100 --times 2 --optimizer Adam & 
wait
python main.py --dataset Mnist --datasize small --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize small --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize large --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --times 2 &
python main.py --dataset Cifar10 --datasize small --algorithm FedSOUL --num_glob_iters 800 --add_new_client 100 --times 2 &
wait
python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset Cifar10 --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 &
wait
python main.py --dataset Mnist --datasize small --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize large --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset Cifar10 --datasize small --algorithm FedRep --num_glob_iters 800 --add_new_client 100 --times 2 & 
wait
python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize large --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --times 2 & 
python main.py --dataset Cifar10 --datasize small --algorithm LGFedAvg --num_glob_iters 800 --add_new_client 100 --times 2 & 
wait
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

