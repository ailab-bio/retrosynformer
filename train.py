import argparse
import time

from retrosynformer.trainer import RetroTrainer


def train_model(config_path):
    start_time = time.time()
    print("Initiate training.")
    trainer = RetroTrainer(config_path)
    trainer.initiate_dataset()
    print("Dataset loaded!")
    trainer.init_model()
    print("Model is loaded!")
    trainer.get_dataloaders()
    begin_train_time = time.time()
    print("Begin training after: ", (begin_train_time - start_time) / 60, " minutes.")
    validation_loss, validation_accuracy, validation_top3_accuracy, fraction_targets_solved = trainer.train()
    print("Training is completed.")
    end_time = time.time()
    print("Training took: ", (end_time - begin_train_time) / (60 * 60), " hours.")

    return validation_loss, validation_accuracy, validation_top3_accuracy, fraction_targets_solved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Program level arguments

    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
    )

    args = parser.parse_args()
    train_model(
        config_path=args.config_path,
    )
