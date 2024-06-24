from kantz_nn import * 
# import multiprocessing as mpc
from GLOBAL import global_module as g
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style("whitegrid")

if __name__ == '__main__':
    from NNModel import model1 as m1 
    shm_list =  []
    # with mpc.Pool(processes = g.parallel_points) as pool:
    #      shm_list = pool.map(run_nn, [m1.ModelWrapper]*g.parallel_points)
    for _ in range(g.parallel_points):
        shm_list.append(run_nn(m1.ModelWrapper))
    sns.regplot(x = [x[0] for x in shm_list], y = [x[1] for x in shm_list], color='magenta')
    plt.xlabel('Lyapunov Exponent')
    plt.ylabel('Loss')
    plt.savefig('model1_chaos_optim.png')
    plt.show()
