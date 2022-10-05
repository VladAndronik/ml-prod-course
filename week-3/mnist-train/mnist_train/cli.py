import typer

from mnist_train.data import load_data
from mnist_train.train import train
from mnist_train.inference import evaluate


app = typer.Typer()
app.command()(load_data)
app.command()(train)
app.command()(evaluate)


if __name__ == "__main__":
    app()
