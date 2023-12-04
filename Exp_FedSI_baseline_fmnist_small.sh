python main.py --dataset FMnist --datasize small --algorithm FedAvg --num_glob_iters 800 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm FedRep --num_glob_iters 800 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm FedSOUL --num_glob_iters 800 --times 2 & 