# Mnist 
python main.py --dataset Mnist --datasize large --algorithm LocalOnly --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedAvg --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedAvgFT --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedBABU --num_glob_iters 800 --num_fineturn_iters 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedRep --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm LGFedAvg --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedSOUL --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1 --subnetwork_rate 0.05 --times 2 &