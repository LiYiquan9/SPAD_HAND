import os
import yaml
import time
from sim import load_opt, simulate_impulse_response, simulate_sensor_response


def run_sim_auto():

    start_time = time.time()
    for i in range(0, 1):
        
        ##### impulse simulation
        root = "experiments"
        os.makedirs(root, exist_ok=True)
        opt = load_opt("opts/simulate/impulse_response/template.yaml")
        
        opt["mesh_path"] = os.path.join("calibration/data/mesh.obj")
        
        # shutil.copyfile(opt, os.path.join(root, "opt.yaml"))
        opt["_root"] = root
        with open(os.path.join(root, "impulse_opt.yaml"), 'w') as file:
            yaml.dump(opt, file, default_flow_style=False)
            
        simulate_impulse_response(opt)
        
        ##### sensor simulation
        opt = load_opt("opts/simulate/sensor_response/template.yaml")                

        opt["data_path"] = root + "/impulse_response.npz"    
       
        opt["_root"] = root
        with open(os.path.join(root, "sensor_opt.yaml"), 'w') as file:
            yaml.dump(opt, file, default_flow_style=False)

        simulate_sensor_response(opt)
        
    end_time = time.time()
    
    iteration_time = end_time - start_time
    print(f"Iteration {i}: {iteration_time:.2f} seconds") 
    
        
        
if __name__ == "__main__":
    run_sim_auto()