from typing import Protocol


class IndividualProtocol(Protocol):
    genome: list[int]
    metrics: dict[str, tuple[int, float]]

    def evaluate(self, *args, **kwargs) -> None: 
        ...
    
    def crossover(self, other: "IndividualProtocol") -> list["IndividualProtocol"]: 
        ...
    
    def mutation(self, rate: float) -> "IndividualProtocol": 
        ...
