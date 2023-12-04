# Mnist 
python main.py --dataset Mnist --datasize large --algorithm LocalOnly --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedAvg --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedPer --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedRep --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm LGFedAvg --num_glob_iters 800 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm FedSOUL --num_glob_iters 800 --times 2 & 