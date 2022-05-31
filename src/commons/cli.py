from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI


class CLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # parser.link_arguments("model.model.init_args.in_channels", "data.in_channels")
        parser.link_arguments("data.n_classes", "model.n_classes")
