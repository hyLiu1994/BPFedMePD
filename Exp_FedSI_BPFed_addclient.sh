python main.py --dataset Mnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset Mnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset FMnist --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 & 
python main.py --dataset Cifar10 --datasize small --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 &
# python main.py --dataset Cifar10 --datasize large --algorithm BPFedPD --num_glob_iters 800 --batch_size 100 --zeta 1e-3 --add_new_client 100 --times 2 &
