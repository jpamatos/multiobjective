from omegaconf import DictConfig


class GenomeDecoder:
    def __init__(self, genome: list[int], genome_cfg: DictConfig) -> None:
        self.genome = genome
        self.slices = genome_cfg.slices
        self.mappings = genome_cfg.mappings

    def decode(self, key: str) -> int | float:
        start, end = self.slices[key]
        bits = "".join(map(str, self.genome[start:end]))
        return self.mappings[f"{key}_genes"][bits]
