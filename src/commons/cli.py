from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

class BayesianCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments("model.in_channels", "data.in_channels")