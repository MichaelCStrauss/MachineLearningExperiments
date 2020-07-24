# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import click
import nltk
from dataset import dataset, token_index, batch_size
from model import build_model


@click.command()
@click.option("--model")
@click.option("--temperature", default=1.0)
@click.argument("start_string")
def predict(model, start_string, temperature):
    # if model is None:
    #     #all entries in the directory w/ stats
    #     dir_path = 'saved_models'
    #     data = (os.path.join(dir_path, fn) for fn in os.listdir(dir_path))
    #     data = ((os.stat(path), path) for path in data)
    #     _, model = sorted(data, reverse=True)[0]
    #     model = os.path.splitext(model)[0]
    # else:
    #     model = 'saved_models/' + model
    model_path = "saved_models"
    # print(f"LOADING {model_path}")
    model = build_model(batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(model_path))
    model.build(tf.TensorShape([1, None]))
    model.summary()

    num_generate = 10
    num_words = 50

    for sample in range(num_generate):

        input_tokens = nltk.word_tokenize(start_string.lower())
        input_indices = [token_index.get_index(t) for t in input_tokens]
        input_indices = tf.expand_dims(input_indices, 0)

        model.reset_states()

        text_generated = input_tokens

        for i in range(num_words):
            predictions = model(input_indices)
            predictions = tf.squeeze(predictions, 0)

            predictions = predictions / temperature
            predicted_index = tf.random.categorical(predictions, num_samples=1)[
                -1, 0
            ].numpy()

            input_indices = tf.expand_dims([predicted_index], 0)

            text_generated.append(token_index.get_token(predicted_index))

        print(" ".join(text_generated))


if __name__ == "__main__":
    predict()
