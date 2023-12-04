# python main.py --dataset Mnist --datasize small --algorithm LocalOnly --num_glob_iters 800 --times 2 & 
# python main.py --dataset Mnist --datasize large --algorithm LocalOnly --num_glob_iters 800 --times 2 & 
# python main.py --dataset FMnist --datasize small --algorithm LocalOnly --num_glob_iters 800 --times 2 & 
# python main.py --dataset FMnist --datasize large --algorithm LocalOnly --num_glob_iters 800 --times 2 & 
python main.py --dataset Cifar10 --datasize small --algorithm LocalOnly --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --times 2 &
python main.py --dataset Cifar10 --datasize large --algorithm LocalOnly --num_glob_iters 800 --learning_rate 0.01 --personal_learning_rate 0.01 --times 2 &