import os
import json
pwd = os.getcwd()
config = {
    "data_dir":str(os.path.join(pwd,"data")),
    "model_dir":str(os.path.join(pwd,"model")),
    "out_dir":str(os.path.join(pwd,"out")),
    "job_dir":str(os.path.join(pwd,"job"))
}

with open(os.path.join(pwd,"config.json"), "w") as f:
    json.dump(config,f, indent=4)