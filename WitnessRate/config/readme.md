### This subfolder shows how to load your configuration by a yaml file.
1. copy this folder to your python project;
2. write your configuration into yaml file;
3. add conf.yml to your git ignore list, so that your private information won't be committed and pushed.
4. modify 'load_config.py' to load the specific config domains and values.
or load your configuration out of 'config' folder, as shown bellow:
```python
from config.load_config import LoadConfig
import sys, os

cwd = sys.path
conf = LoadConfig(os.path.join(cwd[0], 'config//model_cfg.yml'))
conf.print_domains()
conf.print_all()
l_rate = conf.l_rate
# l_rate = float(conf.cfg['model_parameters']['l_rate'])  # alternative way
k_size = conf.k_size
model_save_name = conf.save_to
train_data_fn = conf.train

```

