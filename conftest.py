def pytest_report_header(config):
    if config.getoption("verbose") > 0:
        return [
            "Application version: 1.0.0",
            "The program should be run on Anaconda (Python 3.6.8)",
            "Conda Packages:",
            "\tOpenAI Gym - https://anaconda.org/akode/gym (conda install -c akode gym)",
            "\tPyTorch - https://pytorch.org/get-started/locally/"
        ]