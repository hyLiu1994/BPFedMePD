# Mnist 
python main.py --dataset Mnist --datasize small --algorithm FedAvg --num_glob_iters 800 --times 2 & 
# python main.py --dataset Mnist --datasize small --algorithm FedPer --num_glob_iters 800 --times 2 & 
# python main.py --dataset Mnist --datasize small --algorithm FedRep --num_glob_iters 800 --times 2 & 
# python main.py --dataset Mnist --datasize small --algorithm LGFedAvg --num_glob_iters 800 --times 2 & 
# python main.py --dataset Mnist --datasize small --algorithm pFedBayes --num_glob_iters 800 --batch_size 100 --times 2 & 
# python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --times 2 & 
# python main.py --dataset Mnist --datasize small --algorithm FedSOUL --num_glob_iters 800 --times 2 & 

# python main.py --dataset Mnist --datasize small --algorithm FedSI --num_glob_iters 800 --learning_rate 0.1