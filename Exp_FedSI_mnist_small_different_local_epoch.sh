# Mnist 

python main.py --dataset Mnist --datasize small --algorithm FedAvg --num_glob_iters 800 --local_epochs 5 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm FedAvg --num_glob_iters 800 --local_epochs 10 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm FedAvg --num_glob_iters 800 --local_epochs 15 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm FedAvg --num_glob_iters 800 --local_epochs 20 --times 2 & 
wait
python main.py --dataset Mnist --datasize small --algorithm FedBABU --num_glob_iters 800 --local_epochs 5 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm FedBABU --num_glob_iters 800 --local_epochs 10 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm FedBABU --num_glob_iters 800 --local_epochs 15 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm FedBABU --num_glob_iters 800 --local_epochs 20 --num_fineturn_iters 100 --times 2 & 
wait
python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --local_epochs 5  --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --local_epochs 10  --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --local_epochs 15  --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --local_epochs 20  --times 2 & 
wait
python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --local_epochs 5 --batch_size 100 --zeta 1e-3 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --local_epochs 10 --batch_size 100 --zeta 1e-3 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --local_epochs 15 --batch_size 100 --zeta 1e-3 --times 2 & 
python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --local_epochs 20 --batch_size 100 --zeta 1e-3 --times 2 & 
wait
python main.py --dataset Mnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --local_epochs 5 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
python main.py --dataset Mnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --local_epochs 10 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
python main.py --dataset Mnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --local_epochs 15 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
python main.py --dataset Mnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --local_epochs 20 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
wait
python main.py --dataset Mnist --datasize small --algorithm FedSI --num_glob_iters 800 --local_epochs 5 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
python main.py --dataset Mnist --datasize small --algorithm FedSI --num_glob_iters 800 --local_epochs 10 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
python main.py --dataset Mnist --datasize small --algorithm FedSI --num_glob_iters 800 --local_epochs 15 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
python main.py --dataset Mnist --datasize small --algorithm FedSI --num_glob_iters 800 --local_epochs 20 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &