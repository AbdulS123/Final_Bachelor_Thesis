"""Base backend placeholder."""


class BaseTTSBackend:
    name = "base"

    def load(self):
        raise NotImplementedError

    def synthesize(self, text, output_path=None):
        raise NotImplementedError
