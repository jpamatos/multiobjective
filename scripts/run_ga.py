import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from multiobjective.ga.individual import Individual
from multiobjective.protocols.data import DatasetLoader


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    loader: DatasetLoader = instantiate(cfg.loader, _recursive_=False)

    X_train, X_test, y_train, y_test = loader.load()
    individual = Individual(cfg.individual, _recursive_=False)



if __name__ == "__main__":
    main()
