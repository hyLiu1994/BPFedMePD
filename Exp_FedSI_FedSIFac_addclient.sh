python main.py --dataset Mnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize small --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset Mnist --datasize large --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset FMnist --datasize large --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.1 --add_new_client 100 --times 2 &
python main.py --dataset Cifar10 --datasize small --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.01 --add_new_client 100 --times 2 --optimizer Adam & 
# python main.py --dataset Cifar10 --datasize large --algorithm FedSIFac --num_glob_iters 800 --learning_rate 0.01 --add_new_client 100 --times 2 --optimizer Adam & 