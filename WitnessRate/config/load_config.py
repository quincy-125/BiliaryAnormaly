import yaml


# load customized parameters for your project
class LoadConfig:
    def __init__(self, yml_fn="model_cfg.yml"):
        with open(yml_fn, 'r') as yml:
            self.cfg = yaml.load(yml)
        # TODO: modify lines below to get your configuration from yaml file.
        self.knn_centroids_fn = self.cfg['KNN']['knn_centroids']
        self.neg_assignments_dir = self.cfg['KNN']['neg_assignments']
        self.pos_assignments_dir = self.cfg['KNN']['pos_assignments']
        self.train_log_dir = self.cfg['train']['log_dir']
        self.model_save_name = self.cfg['train']['model_save_dir']
        self.testing_img_dir = self.cfg['eval']['img_dir']
        self.testing_img_rep_dir = self.cfg['eval']['img_rep_dir']
        self.eval_inst_sv = self.cfg['eval']['eval_instance_save_to']
        self.annotation_benign_dir = self.cfg['eval']['annotation_benign_dir']
        self.annotation_malignant_dir = self.cfg['eval']['annotation_malignant_dir']
        self.annotation_uninformative_dir = self.cfg['eval']['annotation_uninformative_dir']
        self.annotation_grayzone_dir = self.cfg['eval']['annotation_grayzone_dir']

    def print_domains(self):
        for section in self.cfg:
            print(section)

    def print_all(self):
        print(self.cfg)


if __name__ == '__main__':
    cfg = LoadConfig()


